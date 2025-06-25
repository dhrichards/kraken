#!/bin/bash
#SBATCH --output=/users/dancha/jobs/hires.%j.%N.out
#SBATCH --error=/users/dancha/jobs/hires.%j.%N.err
#SBATCH --job-name=icebergtest
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=2
#SBATCH --mem=256gb
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --partition=rocky
#SBATCH --account=rocky
#SBATCH --mail-type=begin,end,fail,requeue
#SBATCH --mail-user=dancha@bas.ac.uk

# module purge
# module load mpi/mpich-x86_64

. /users/dancha/spack/share/spack/setup-env.sh
spack env activate fenicsx
cd /users/dancha/kraken3


mpirun --version


mpirun -n $SLURM_NTASKS python3 scripts/iceberg.py