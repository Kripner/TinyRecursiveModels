import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

import numpy as np
import torch
from torch import nn

import pydantic
from torch.utils.data import IterableDataset, DataLoader
import arc_agi_core as arc_core

from evaluate_trained_model import load_config_from_checkpoint
from evaluators.arc import ARC, _crop
from models.losses import IGNORE_LABEL_ID
from pretrain import PretrainConfig, EvaluatorConfig
from dataset.common import PuzzleDatasetMetadata
from dataset.build_arc_dataset import inverse_aug, grid_hash, arc_grid_to_np, PuzzleIdSeparator
from utils.functions import load_model_class


class SimplePuzzleDatasetConfig(pydantic.BaseModel):
    seed: int
    data_path: Path
    batch_size: int


class SimplePuzzleDataset(IterableDataset):
    def __init__(self, config: SimplePuzzleDatasetConfig, split: str = "train"):
        super().__init__()
        self.config = config
        self.split = split

        self.metadata = self._load_metadata(config.data_path)

        self.evaluator = None  # To be assigned.

        # State
        self._data = None
        self._iters = 0

    def _load_metadata(self, dataset_path: Path) -> PuzzleDatasetMetadata:
        with open(os.path.join(dataset_path, self.split, "dataset.json"), "r") as f:
            return PuzzleDatasetMetadata(**json.load(f))

    def _lazy_load_dataset(self):
        if self._data is not None:
            return

        field_mmap_modes = {
            "inputs": "r",
            "labels": "r",

            # Keep indices in memory
            "puzzle_identifiers": None,
            "puzzle_indices": None,
            "group_indices": None
        }

        # Load data
        self._data = {}
        for set_name in self.metadata.sets:  # Load subset
            self._data[set_name] = {
                field_name: np.load(
                    os.path.join(self.config.data_path, self.split, f"{set_name}__{field_name}.npy"),
                    mmap_mode=mmap_mode,
                )
                for field_name, mmap_mode in field_mmap_modes.items()
            }

    def _collate_batch(self, batch):
        batch = {k: np.asarray(v).astype(np.int32) for k, v in batch.items()}

        # Convert ignore label IDs
        if self.metadata.ignore_label_id is not None:
            batch["labels"][batch["labels"] == self.metadata.ignore_label_id] = IGNORE_LABEL_ID

        # Pad
        if batch["puzzle_identifiers"].size < self.config.batch_size:
            pad_size = self.config.batch_size - batch["puzzle_identifiers"].size
            pad_values = {
                "inputs": self.metadata.pad_id,
                "labels": IGNORE_LABEL_ID,
                "puzzle_identifiers": self.metadata.blank_identifier_id
            }
            batch = {
                k: np.pad(v, ((0, pad_size),) + ((0, 0),) * (v.ndim - 1), constant_values=pad_values[k])
                for k, v in batch.items()
            }

        return {k: torch.from_numpy(v) for k, v in batch.items()}

    def _iter_test(self):
        for set_i, (set_name, dataset) in enumerate(self._data.items()):  # type: ignore
            total_examples = len(dataset["inputs"])

            # Load examples one by one
            start_index = 0
            while start_index < total_examples:
                # Compute indices
                end_index = min(total_examples, start_index + self.config.batch_size)

                # Get batch of examples, and also puzzle IDs
                puzzle_indices = []
                puzzle_index = np.searchsorted(dataset["puzzle_indices"], start_index, side="right") - 1
                for i in range(start_index, end_index):
                    while puzzle_index + 1 < len(dataset["puzzle_indices"]) and i >= dataset["puzzle_indices"][puzzle_index + 1]:
                        puzzle_index += 1
                    puzzle_indices.append(puzzle_index)

                yield {
                    "inputs": dataset["inputs"][start_index: end_index],
                    "labels": dataset["labels"][start_index: end_index],
                    "puzzle_identifiers": dataset["puzzle_identifiers"][puzzle_indices]
                }

                # Advance to next batch
                start_index += self.config.batch_size

    def __iter__(self):
        self._lazy_load_dataset()

        samples = {"inputs": [], "labels": [], "puzzle_identifiers": []}
        already_seen = set()
        for new_samples in self._iter_test():
            for i in range(len(new_samples["puzzle_identifiers"])):
                identifier = new_samples["puzzle_identifiers"][i]
                assert identifier != self.evaluator.blank_identifier_id
                name = self.evaluator.identifier_map[identifier]
                if PuzzleIdSeparator in name:
                    continue
                already_seen.add(name)
                for k in new_samples.keys():
                    samples[k].append(new_samples[k][i])
            if len(samples["puzzle_identifiers"]) >= self.config.batch_size:
                yield self._collate_batch(samples)
                samples = {"inputs": [], "labels": [], "puzzle_identifiers": []}
        if len(samples["puzzle_identifiers"]) != 0:
            yield self._collate_batch(samples)


def create_dataloader(config: PretrainConfig, split: str, batch_size: int):
    dataset = SimplePuzzleDataset(SimplePuzzleDatasetConfig(
        seed=config.seed,
        data_path=config.data_paths_test[0] if len(config.data_paths_test) > 0 and split == "test" else
        config.data_paths[0],
        batch_size=batch_size,
    ), split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        persistent_workers=True,
    )
    return dataloader, dataset.metadata


def create_model(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata):
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore
        batch_size=config.global_batch_size,
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        causal=False  # Non-autoregressive
    )

    # Instantiate model with loss head
    model_cls = load_model_class(config.arch.name)
    # loss_head_cls = load_model_class(config.arch.loss.name)

    with torch.device("cuda"):
        model: nn.Module = model_cls(model_cfg)
        print(model)
        # model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore

    return model

@dataclass
class HStep:
    out_grid: arc_core.Grid
    L_trace: list[arc_core.Grid]

    def to_dict(self):
        return {
            "out_grid": self.out_grid.to_list(),
            "L_trace": [g.to_list() for g in self.L_trace],
        }

@dataclass
class RolloutStep:
    out_grid: arc_core.Grid
    H_trace: list[HStep]
    q_halt_logit: float
    q_continue_logit: float

    def to_dict(self):
        return {
            "out_grid": self.out_grid.to_list(),
            "H_trace": [s.to_dict() for s in self.H_trace],
            "q_halt_logit": self.q_halt_logit,
            "q_continue_logit": self.q_continue_logit,
        }

@dataclass
class PuzzleRollout:
    puzzle_name: str
    puzzle: arc_core.Task
    input: arc_core.Grid
    rollout: list[RolloutStep]

    def to_dict(self):
        return {
            "puzzle_name": self.puzzle_name,
            "puzzle": self.puzzle.to_dict(),
            "input": self.input.to_list(),
            "rollout": [s.to_dict() for s in self.rollout],
        }


def evaluate(model: nn.Module, eval_loader: torch.utils.data.DataLoader, evaluator: ARC, result_file: Path):
    with torch.inference_mode():
        rollouts = []
        for batch_idx, batch in enumerate(eval_loader):
            print(f"Processing batch {batch_idx} (size {len(batch["puzzle_identifiers"])})")

            # To device
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.device("cuda"):
                carry = model.initial_carry(batch)  # type: ignore

            batch_rollouts = []
            for identifier, input_ in zip(batch["puzzle_identifiers"].detach().cpu().numpy(),
                                          batch["inputs"].detach().cpu().numpy()):
                assert identifier != evaluator.blank_identifier_id
                name = evaluator.identifier_map[identifier]

                puzzle_raw = evaluator.test_puzzles[name]
                puzzle = arc_core.Task(
                    [arc_core.Pair(pair["input"], pair["output"]) for pair in puzzle_raw["train"]],
                    [arc_core.Pair(pair["input"], pair["output"]) for pair in puzzle_raw["test"]],
                )
                input_grid = arc_core.Grid(_crop(input_))
                batch_rollouts.append(PuzzleRollout(name, puzzle, input_grid, rollout=[]))

            while True:
                carry, outputs, batch_traces = model(carry=carry, batch=batch)

                outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)
                outputs = {k: outputs[k].detach().cpu() for k in outputs}

                for i in range(len(batch_rollouts)):
                    result = batch_rollouts[i]
                    pred = arc_core.Grid(_crop(outputs["preds"][i].numpy()))
                    q_halt_logit = outputs["q_halt_logits"][i].item()
                    q_continue_logit = outputs["q_continue_logits"][i].item()

                    logits_to_grid = lambda logits: arc_core.Grid(_crop(torch.argmax(logits, dim=-1).detach().cpu().numpy()))
                    h_trace = []
                    for h_idx in range(len(batch_traces)):
                        h_out_grid = logits_to_grid(batch_traces[h_idx][0][i])
                        l_trace = [
                            logits_to_grid(batch_traces[h_idx][1][l_idx][i])
                            for l_idx in range(len(batch_traces[h_idx][1]))
                        ]

                        h_trace.append(HStep(h_out_grid, l_trace))

                    result.rollout.append(RolloutStep(pred, h_trace, q_halt_logit, q_continue_logit))

                all_finish = carry.halted.all()  # At eval, all 16 steps are performed always.
                if all_finish:
                    break

            rollouts.extend(batch_rollouts)
            result_file.write_text("\n".join(json.dumps(r.to_dict()) for r in rollouts))
            del carry, outputs, all_finish, batch_rollouts, batch_traces

def evaluate_checkpoint(
        checkpoint_path: Path,
        data_path: Path,
        output_dir: Path,
        batch_size: int,
        config_overrides: Optional[dict[str, Any]] = None,
):
    try:
        config = load_config_from_checkpoint(checkpoint_path)
    except ValueError:
        print("No .hydra config found, using default config with checkpoint path")
        config = PretrainConfig()

    # Apply overrides
    config.checkpoint_path = str(checkpoint_path.parent)
    config.data_paths = [str(data_path)]

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

    output_dir.mkdir(parents=True, exist_ok=True)
    result_file = output_dir / "rollouts.jsonl"

    print(f"Loading evaluation dataset from: {data_path}")
    try:
        eval_loader, eval_metadata = create_dataloader(config, "test", batch_size)
    except FileNotFoundError as e:
        print(f"Error loading dataset: {e}")
        print("Make sure the dataset exists and has a 'test' split")
        return

    # Load model - we need to get training metadata for model creation
    # Try to load from the training dataset first
    print(f"Loading train dataset from: {data_path}")
    try:
        train_loader, train_metadata = create_dataloader(config, "train", batch_size)
    except FileNotFoundError:
        # If no train split, use eval metadata
        print("No train split found, using eval metadata for model creation")
        train_metadata = eval_metadata

    print("Creating model...")
    model = create_model(config, train_metadata)

    print(f"Loading checkpoint weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Handle different checkpoint formats
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    state_dict = {
        (k if not k.startswith("_orig_mod.model.") else k[len("_orig_mod.model."):]): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print("Running evaluation...")
    print(f"Dataset has {len(eval_metadata.sets)} test sets")

    evaluator = ARC(str(data_path), eval_metadata)
    train_loader.dataset.evaluator = evaluator
    eval_loader.dataset.evaluator = evaluator
    evaluate(model, eval_loader, evaluator, result_file)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained TRM model checkpoint")
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        required=True,
        help="Path to the model checkpoint file"
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Path to the dataset directory"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="trm_out",
        help="Directory to save evaluation results"
    )
    # parser.add_argument(
    #     "--batch-size",
    #     type=int,
    #     default=512,
    #     help="Global batch size for evaluation"
    # )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Global batch size for evaluation"
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
            EvaluatorConfig(**{
                "name": "arc@ARC",
                "submission_K": args.submission_k,
                "aggregated_voting": args.aggregated_voting,
                "pass_Ks": [1, 2, 5, 10, 100, 1000]
            })
        ]
    }

    evaluate_checkpoint(
        checkpoint_path=args.checkpoint_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        config_overrides=config_overrides,
    )


if __name__ == "__main__":
    main()
