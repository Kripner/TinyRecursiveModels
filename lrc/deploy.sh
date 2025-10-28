#!/usr/bin/env bash

rsync \
   -av --delete \
   --progress \
   --exclude='*.egg-info/' --exclude='*.pyc' --exclude='__pycache__/' --exclude='*.pth' \
   config dataset evaluators kaggle models utils *.py requirements.txt \
   geri:~/trm-fork/
