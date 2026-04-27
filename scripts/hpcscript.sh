#!/bin/bash
#SBATCH --output=/users/dancha/jobs/iceberg.%j.out
#SBATCH --error=/users/dancha/jobs/erroriceberg.%j.%N.err
#SBATCH --job-name=iceberg
#SBATCH --ntasks=30
#SBATCH --cpus-per-task=2
#SBATCH --mem=100G
#SBATCH --nodes=1
#SBATCH --time=48:30:00
#SBATCH --partition=rocky
#SBATCH --account=rocky
#SBATCH --mail-type=begin,end,fail,requeue
#SBATCH --mail-user=dancha@bas.ac.uk



. /users/dancha/spack/share/spack/setup-env.sh
spack env activate fenicsx
cd /users/dancha/kraken3


mpirun --version


mpirun -n $SLURM_NTASKS python3 scripts/iceberg.py --lstar  0.025 --dt 10 --height 400 --nondim_length 8 --type icebergsymm --cellfactor 2 --no-cracks True --strength_deg 0
#--lstar 0.06 --dt 5.0 --height 175 --nondim_length 40 --type relaxation --cellfactor 3 --nt 500
#--level 0.0 --l 3 --dt 0.02 --type relaxation --psicrit 1.0 --Gc 0.5 --damagemodel AT2higher --cellfactor 1.0 --T -5 --gv_tol 1e-3 --height 300 --tol 5e-6 --nt 500 --refine_z 0.3 --suffix linear