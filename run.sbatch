#!/bin/bash
#SBATCH -N1 --ntasks-per-node=4
#SBATCH --mem=64G
#SBATCH -t5:00:00
#SBATCH -J mof_pfas_optimizer
#SBATCH -o Report-%j.out -e Report_err-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hkansara3@gatech.edu

cd $SLURM_SUBMIT_DIR

source ~/.bashrc

module load anaconda3/2023.03
conda activate exx-sr

python /home/hice1/hkansara3/scratch/EXX_regression/src/main.py > "${SLURM_JOBID}_results.txt"
