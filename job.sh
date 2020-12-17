#!/bin/bash
#SBATCH --mail-user=salari.m1375@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --account=def-mori_cpu
#SBATCH --job-name=MDS
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --time=11:59:00
#SBATCH --mem=100G

cd $SLURM_TMPDIR
cp -r ~/scratch/MDS .
cd MDS

module load python/3.7 cuda/10.0
virtualenv --no-download venv
source venv/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt

python main.py


