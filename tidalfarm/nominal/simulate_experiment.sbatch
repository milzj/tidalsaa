#!/bin/bash

#SBATCH -J t10                    # Job name
#SBATCH --account=gts-jmilz3                     # charge account
#SBATCH -N 25 --ntasks-per-node=1                  # number of nodes and cores per node
#SBATCH --mem-per-cpu=16gb                        # Memory per core
#SBATCH -qinferno                               # QOS Name
#SBATCH -o output/Report-%x-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=ALL              # Mail preferences
#SBATCH --mail-user=jmilz3@gatech.edu
#SBATCH --time=01:15:00

cd $HOME/tidalsaa/tidalfarm/nominal

module load anaconda3/2022.05.0.1
conda activate tidalsaa
module load gcc

export PYTHONPATH=$HOME/tidalsaa:${PYTHONPATH}

date=$(date '+%d-%b-%Y-%H-%M-%S')

mpiexec -n 25 python simulate_experiment.py $date
