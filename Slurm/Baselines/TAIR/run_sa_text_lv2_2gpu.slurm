#!/usr/bin/env bash
#SBATCH --job-name=tair_sat_lv2
#SBATCH --partition=suma_a6000
#SBATCH --qos=big_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --time=48:00:00
#SBATCH --output=/scratch2/james2602/TextAwareIR/Slurm/Baselines/TAIR/logs/%x_%j.out
#SBATCH --error=/scratch2/james2602/TextAwareIR/Slurm/Baselines/TAIR/logs/%x_%j.err

set -euo pipefail

cd /scratch2/james2602/TextAwareIR

bash /scratch2/james2602/TextAwareIR/Slurm/Baselines/TAIR/run_sa_text_lv2_2gpu.sh
