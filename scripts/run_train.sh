#!/bin/bash

# Activate the conda environment
source activate your_environment_name

# Navigate to the src directory
cd ../src/training

# Run the training script
python train.py --config ../experiments/config.yaml

# Deactivate the conda environment
conda deactivate