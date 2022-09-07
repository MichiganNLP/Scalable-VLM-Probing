#!/usr/bin/env bash

#SBATCH --job-name=clip-svo-probes-feature-importance
#SBATCH --mail-type=BEGIN,END
#SBATCH --cpus-per-task=36
#SBATCH --mem=180G
#SBATCH --time=72:00:00
#SBATCH --account=mihalcea0
#SBATCH --partition=standard

source .venv/bin/activate
python main.py
