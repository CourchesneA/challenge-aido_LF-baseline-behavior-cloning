#!/bin/bash
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --time=24:00:00
#SBATCH -o /miniscratch/courchea/jout/slurm-%j.out

module load python/3.7
module load cuda/10.1/cudnn/7.6
module load tensorflow/2.2

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cvmfs/ai.mila.quebec/apps/x86_64/common/cuda/10.1/extras/CUPTI/lib64
export LD_INCLUDE_PATH=$LD_INCLUDE_PATH:/cvmfs/ai.mila.quebec/apps/x86_64/common/cuda/10.1/extras/CUPTI/include

virtualenv tmp
source tmp/bin/activate
pip install -r requirements.txt

#source startvx.sh
#
#sleep 2
#
#xdpyinfo -display :99 >/dev/null 2>&1 && echo "In use" || echo "Free"

python3 train.py
