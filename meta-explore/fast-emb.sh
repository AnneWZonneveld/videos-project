#!/bin/bash

#SBATCH --mail-user=azonneveld@zedat.fu-berlin.de
#SBATCH --job-name=fast-emb                                       
#SBATCH --nodes=1                                    
#SBATCH --ntasks=1                                   
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000                              
#SBATCH --time=5:00:00                            
#SBATCH --qos=standard
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1


python fast-emb.py 