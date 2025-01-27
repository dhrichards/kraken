#!/bin/bash
#SBATCH --output=/users/dancha/jobs/myfirstjob.%j.%N.out
#SBATCH --error=/users/dancha/jobs/myfirstjob.%j.%N.err
#SBATCH --job-name=icebergtest
#SBATCH --mem=4gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --partition=rocky
#SBATCH --account=rocky
#SBATCH --mail-type=begin,end,fail,requeue
#SBATCH --mail-user=dancha@bas.ac.uk

module purge
module load mpi/mpich-x86_64

. /users/dancha/spack/share/spack/setup-env.sh
spack env activate fenicsx2
cd /users/dancha/kraken3/scripts


mpirun --version


srun -n 4 python3 iceberg_hpc.py