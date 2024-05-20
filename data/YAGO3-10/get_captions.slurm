#!/bin/bash
#SBATCH --job-name=yago_captions         # Job name
#SBATCH --mail-type=END,FAIL          # Mail events
#SBATCH --mail-user=andrea.papaluca@anu.edu.au     # Where to send mail
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --mem=16gb                    # Job memory request
#SBATCH --output=yago_captions_%j.log   # Standard output and error log
#SBATCH --time=72:00:00               # Time limit hrs:min:sec
pwd; hostname; date

source activate torch
echo "---- Retrieving Captions ----"
python get_captions.py
echo "---- Done ----"
date
