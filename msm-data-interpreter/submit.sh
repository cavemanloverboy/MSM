#!/bin/bash

#SBATCH --job-name combineMFT
#SBATCH --output output/combineMFT.out
#SBATCH --cpus-per-task=16
#SBATCH --tasks-per-node=1
#SBATCH --nodes=10
#SBATCH -p kipac,normal,hns
#SBATCH -t 24:00:00

cd /oak/stanford/orgs/kipac/users/pizza/MSM/msm-data-interpreter/
srun cargo +nightly run --release -j 16 --example analyze_bose_star --color always -- ../rust/sim_data/gaussian-overdensity-512