#!/bin/bash
#SBATCH --output=/users/dancha/jobs/iceberg.%j
#SBATCH --error=/users/dancha/jobs/erroriceberg.%j.%N.err
#SBATCH --job-name=iceberg
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --mem=150G
#SBATCH --time=48:30:00
#SBATCH --partition=rocky
#SBATCH --account=rocky
#SBATCH --mail-type=begin,end,fail,requeue
#SBATCH --mail-user=dancha@bas.ac.uk



. /users/dancha/spack/share/spack/setup-env.sh
spack env activate fenicsx
cd /users/dancha/kraken3


# mpirun --bind-to none -n $SLURM_NTASKS python3 scripts/iceberg.py --lstar 0.015 --dt 0.2 --height 450 --nondim_length 6 --type iceberg --cellfactor 2.4 --nt 1200 --Kic 300 --strength_deg 40 --relax_time 2
mpirun --bind-to none -n $SLURM_NTASKS python3 scripts/iceberg.py --lstar 0.005 --dt 0.25 --height 600 --nondim_length 2 --type ssa --cellfactor 2 --cracks 0 --nt 2 --strength_deg 0 --save_bp True
# mpirun --bind-to none -n $SLURM_NTASKS python3 scripts/iceberg.py --lstar  0.005 --dt 2.5 --height 500 --nondim_length 5 --type icebergsymm --cellfactor 1.0 --strength_deg 0 --Ttop -5 --Tbot -5 --nt 1500 --relax_time 400 --lfactor 2 --save_bp True
