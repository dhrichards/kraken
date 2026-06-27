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


# mpirun --bind-to none -n $SLURM_NTASKS python3 scripts/iceberg.py --lstar 0.02 --dt 0.25 --height 600 --nondim_length 2 --type ssa --cellfactor 2 --cracks 0 --nt 2 --strength_deg 0 --save_bp True
# mpirun --bind-to none -n $SLURM_NTASKS python3 scripts/iceberg.py --lstar  0.005 --dt 2.5 --height 510 --nondim_length 10 --type icebergsymm --cellfactor 1.0 --strength_deg 0 --Ttop -5 --Tbot -5 --nt 1500 --relax_time 400 --lfactor 2
# mpirun --bind-to none -n $SLURM_NTASKS python3 scripts/iceshelf.py --lstar  0.005 --dt 2.5 --height 500 --nondim_length 5 --cellfactor 1.0 --strength_deg 0 --Ttop -5 --Tbot -5 --nt 1500 --relax_time 400 --lfactor 2 --suffix variationalsmooth
# mpirun --bind-to none -n $SLURM_NTASKS python3 scripts/degradationtest.py --lstar  0.0025 --cellfactor 2.0
mpirun --bind-to none -n $SLURM_NTASKS python3 scripts/ssa.py --lstar  0.01 --cellfactor 2.0 --height 600 --save_bp True