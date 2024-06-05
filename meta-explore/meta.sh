#!/bin/bash

#SBATCH --mail-user=azonneveld@zedat.fu-berlin.de
#SBATCH --job-name=guse-emb                                       
#SBATCH --nodes=1                                    
#SBATCH --ntasks=1                                   
#SBATCH --cpus-per-task=1
#SBATCH --mem=3000                           
#SBATCH --time=5:00:00                            
#SBATCH --qos=standard
                       
echo "meta-explore"
python meta-explore.py 