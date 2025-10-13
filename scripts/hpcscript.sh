#!/bin/bash
#SBATCH --output=/users/dancha/jobs/iceberg.%j.out
#SBATCH --error=/users/dancha/jobs/erroriceberg.%j.%N.err
#SBATCH --job-name=iceberg
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=2
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --time=24:30:00
#SBATCH --partition=rocky
#SBATCH --account=rocky
#SBATCH --mail-type=begin,end,fail,requeue
#SBATCH --mail-user=dancha@bas.ac.uk



. /users/dancha/spack/share/spack/setup-env.sh
spack env activate fenicsx
cd /users/dancha/kraken3


mpirun --version


mpirun -n $SLURM_NTASKS python3 scripts/iceberg.py --level 0.01 --l 3 --dt 3 --type relaxation --split dp