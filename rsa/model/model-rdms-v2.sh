#!/bin/bash

#SBATCH --mail-user=azonneveld@zedat.fu-berlin.de
#SBATCH --job-name=model-rdms                              
#SBATCH --nodes=1                                   
#SBATCH --ntasks=1                                   
#SBATCH --cpus-per-task=1
#SBATCH --mem=500                             
#SBATCH --time=30:00:00                            
#SBATCH --qos=standard

python model-rdms-v2.py 