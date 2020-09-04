#!/bin/bash
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=45G
#SBATCH --time=10:00:00
#SBATCH -o /network/tmp1/courchea/slurm-%j.out

module load python/3.7
module load cuda/11.0
module load tensorflow/2.2

virtualenv tmp
source tmp/bin/activate
pip install -r requirements.txt

source startvx.sh

sleep 2

xdpyinfo -display :99 >/dev/null 2>&1 && echo "In use" || echo "Free"

python3 train.py
