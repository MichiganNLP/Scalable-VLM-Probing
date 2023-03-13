#!/usr/bin/env bash

#SBATCH --job-name=clip-svo-probes-feature-importance
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --cpus-per-task=36
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --account=mihalcea98
#SBATCH --partition=standard

module load python/3.10.4
source venv/bin/activate
#python main.py --dependent-variable-name pos_clip_score --no-neg-features --vif
python main.py --dependent-variable-name neg_clip_score --vif
