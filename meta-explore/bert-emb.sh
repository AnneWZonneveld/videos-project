#!/bin/bash

#SBATCH --mail-user=azonneveld@zedat.fu-berlin.de
#SBATCH --job-name=bert-emb                                       
#SBATCH --nodes=1                                    
#SBATCH --ntasks=1                                   
#SBATCH --cpus-per-task=1
#SBATCH --mem=6000                           
#SBATCH --time=5:00:00                            
#SBATCH --qos=standard
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1 # number of GPUs requested

echo "bert version"
python bert-emb.py 