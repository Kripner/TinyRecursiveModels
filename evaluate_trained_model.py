#!/usr/bin/env python3
"""
Standalone evaluation script for trained HRM models.
Loads a checkpoint and evaluates it on a specified dataset.
Supports both single and multi-GPU evaluation.

Usage:
    # Single GPU
    python evaluate_trained_model.py \
        --checkpoint-path checkpoints/arc-aug-600/run_name/checkpoint.pt \
        --data-path data/arc-aug-1000-test \
        --output-dir eval_results/run_name_1000aug

    # Multi-GPU
    torchrun --nproc-per-node 8 evaluate_trained_model.py \
        --checkpoint-path checkpoints/arc-aug-600/run_name/checkpoint.pt \
        --data-path data/arc-aug-1000-test \
        --output-dir eval_results/run_name_1000aug
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
import torch.distributed as dist
import numpy as np
import wandb

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from pretrain import (
    PretrainConfig,
    create_dataloader,
    create_model,
    create_evaluators,
    evaluate,
    load_checkpoint,
    TrainState
)
from utils.functions import load_model_class


def setup_distributed():
    """Initialize distributed training if in distributed environment."""
    rank = 0
    world_size = 1
    cpu_group = None

    if "LOCAL_RANK" in os.environ:
        # Initialize distributed
        dist.init_process_group(backend="nccl")

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

        # CPU GLOO process group for evaluation
        cpu_group = dist.new_group(backend="gloo")
        assert dist.get_rank(cpu_group) == rank and dist.get_world_size(cpu_group) == world_size

    return rank, world_size, cpu_group


def load_config_from_checkpoint(checkpoint_path: Path) -> PretrainConfig:
    """Load the config from checkpoint directory's config files."""
    checkpoint_dir = checkpoint_path.parent

    # Try different config file locations in order of preference
    config_files = [
        checkpoint_dir / "all_config.yaml",  # Saved by pretrain.py
        checkpoint_dir / ".hydra" / "config.yaml",  # Hydra format
        Path("config/cfg_pretrain.yaml")  # Default fallback
    ]

    config_dict = None
    for config_file in config_files:
        if config_file.exists():
            print(f"Loading config from: {config_file}")
            import yaml
            with open(config_file, 'r') as f:
                config_dict = yaml.safe_load(f)
            break

    if config_dict is None:
        raise ValueError(f"No config file found in {checkpoint_dir} or default location")

    # Convert to PretrainConfig
    config = PretrainConfig(**config_dict)
    return config


def evaluate_checkpoint(
        checkpoint_path: str,
        data_path: str,
        output_dir: str,
        config_overrides: Optional[Dict[str, Any]] = None,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        save_predictions: bool = False
):
    """
    Evaluate a trained model checkpoint on a specified dataset.

    Args:
        checkpoint_path: Path to the model checkpoint
        data_path: Path to the dataset for evaluation
        output_dir: Directory to save evaluation results
        config_overrides: Optional config overrides
        wandb_project: Optional W&B project name
        wandb_run_name: Optional W&B run name
        save_predictions: Whether to save model predictions
    """
    # Setup distributed if needed
    rank, world_size, cpu_group = setup_distributed()

    # Load config from checkpoint
    checkpoint_path = Path(checkpoint_path)
    if rank == 0:
        print(f"Loading config from checkpoint: {checkpoint_path}")

    try:
        config = load_config_from_checkpoint(checkpoint_path)
    except ValueError:
        # Fallback: create a basic config
        if rank == 0:
            print("No .hydra config found, using default config with checkpoint path")
        config = PretrainConfig()

    # Apply overrides
    config.checkpoint_path = str(checkpoint_path.parent)
    config.data_path = data_path

    if config_overrides:
        for key, value in config_overrides.items():
            if key == "arch" and isinstance(value, dict):
                # Handle nested arch config updates (e.g., halt_max_steps)
                for arch_key, arch_value in value.items():
                    if hasattr(config.arch, '__pydantic_extra__'):
                        config.arch.__pydantic_extra__[arch_key] = arch_value
                    else:
                        setattr(config.arch, arch_key, arch_value)
            else:
                setattr(config, key, value)

    # Setup output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize W&B if requested
    if rank == 0 and wandb_project:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name or f"eval_{checkpoint_path.stem}",
            config=OmegaConf.to_container(OmegaConf.create(config.__dict__)),
            dir=str(output_dir)
        )

    # Load dataset
    if rank == 0:
        print(f"Loading evaluation dataset from: {data_path}")

    try:
        eval_loader, eval_metadata = create_dataloader(
            config,
            "test",
            test_set_mode=True,
            epochs_per_iter=1,
            global_batch_size=config.global_batch_size,
            rank=rank,
            world_size=world_size
        )
    except FileNotFoundError as e:
        if rank == 0:
            print(f"Error loading dataset: {e}")
            print("Make sure the dataset exists and has a 'test' split")
        return

    # Create model
    if rank == 0:
        print("Creating model...")

    # Load model - we need to get training metadata for model creation
    # Try to load from the training dataset first
    try:
        train_loader, train_metadata = create_dataloader(
            config,
            "train",
            test_set_mode=False,
            epochs_per_iter=1,
            global_batch_size=config.global_batch_size,
            rank=rank,
            world_size=world_size
        )
    except FileNotFoundError:
        # If no train split, use eval metadata
        if rank == 0:
            print("No train split found, using eval metadata for model creation")
        train_metadata = eval_metadata

    model, _, _ = create_model(config, train_metadata, rank=rank, world_size=world_size)

    # Load checkpoint weights
    if rank == 0:
        print(f"Loading checkpoint weights from: {checkpoint_path}")

        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Handle different checkpoint formats
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Load state dict
        model.load_state_dict(state_dict, strict=True)

        # Get step number if available
        step = checkpoint.get('step', 0)
    else:
        step = 0

    # Broadcast model parameters from rank 0
    if world_size > 1:
        # Broadcast step number
        step_tensor = torch.tensor([step], device='cuda')
        dist.broadcast(step_tensor, src=0)
        step = step_tensor.item()

        # Broadcast model parameters
        with torch.no_grad():
            for param in list(model.parameters()) + list(model.buffers()):
                dist.broadcast(param, src=0)

    # Create evaluators
    if rank == 0:
        print("Creating evaluators...")
    evaluators = create_evaluators(config, eval_metadata)

    # Create a minimal train state for evaluation (match dataclass field order)
    train_state = TrainState(
        model=model,
        optimizers=[],  # Not needed for evaluation
        optimizer_lrs=[],  # Not needed for evaluation
        carry=None,  # Will be initialized during evaluation
        step=step,
        total_steps=step + 1  # Just needs to be > step
    )

    # Set model to eval mode
    model.eval()

    # Run evaluation
    if rank == 0:
        print("Running evaluation...")
        print(f"Dataset has {len(eval_metadata.sets)} test sets")

    # Configure what to save
    if save_predictions:
        config.eval_save_outputs = ["inputs", "preds", "puzzle_identifiers"]

    metrics = evaluate(
        config,
        train_state,
        eval_loader,
        eval_metadata,
        evaluators,
        rank=rank,
        world_size=world_size,
        cpu_group=cpu_group
    )

    # Save results
    if rank == 0 and metrics is not None:
        # Convert metrics to JSON-serializable format
        def convert_to_serializable(obj):
            """Convert numpy types to Python native types for JSON serialization"""
            import numpy as np
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        serializable_metrics = convert_to_serializable(metrics)

        # Save metrics to JSON
        metrics_file = output_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)

        print("\nEvaluation Results:")
        print("=" * 50)
        for key, value in metrics.items():
            if isinstance(value, dict):
                print(f"\n{key}:")
                for subkey, subvalue in value.items():
                    print(f"  {subkey}: {subvalue:.4f}")
            else:
                print(f"{key}: {value:.4f}")

        print(f"\nResults saved to: {output_dir}")

        # Log to W&B if active
        if wandb.run:
            wandb.log(metrics)
            wandb.finish()

    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained HRM model checkpoint")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to the model checkpoint file"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the dataset directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Global batch size for evaluation"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="W&B project name (optional)"
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="W&B run name (optional)"
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save model predictions"
    )
    parser.add_argument(
        "--submission-k",
        type=int,
        default=2,
        help="Number of predictions per puzzle for submission"
    )
    parser.add_argument(
        "--aggregated-voting",
        action="store_true",
        default=True,
        help="Use aggregated voting across augmentations"
    )

    args = parser.parse_args()

    # Config overrides
    config_overrides = {
        "global_batch_size": args.batch_size,
        "evaluators": [
            {
                "name": "ARC",
                "submission_K": args.submission_k,
                "aggregated_voting": args.aggregated_voting,
                "pass_Ks": [1, 2, 5, 10, 100, 1000]
            }
        ]
    }

    evaluate_checkpoint(
        checkpoint_path=args.checkpoint_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        config_overrides=config_overrides,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        save_predictions=args.save_predictions
    )


if __name__ == "__main__":
    main()