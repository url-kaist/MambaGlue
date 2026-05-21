#!/usr/bin/env bash
# Two-stage MambaGlue + SuperPoint training via glue-factory.
# Run from the MambaGlue repository root after `pip install -e ".[train]"` and
# after installing glue-factory (see mambaglue/training/README.md).
set -euo pipefail

CONFIG_DIR="mambaglue/training/configs"

# Stage 1: synthetic homographies pretraining
python -m gluefactory.train sp_mambaglue_homog \
    --conf "${CONFIG_DIR}/superpoint+mambaglue_homography.yaml" \
    --mixed_precision bfloat16 \
    "$@"

# Stage 2: MegaDepth fine-tuning, resumed from stage 1 checkpoint
python -m gluefactory.train sp_mambaglue_md \
    --conf "${CONFIG_DIR}/superpoint+mambaglue_megadepth.yaml" \
    --mixed_precision bfloat16 \
    train.load_experiment=sp_mambaglue_homog \
    "$@"
