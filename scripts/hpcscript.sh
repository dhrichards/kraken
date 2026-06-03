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
# mpirun --bind-to none -n $SLURM_NTASKS python3 scripts/iceberg.py --lstar 0.0025 --dt 0.25 --height 400 --nondim_length 0.5 --type ssa --cellfactor 2 --cracks 0 --nt 2 --Kic 400 --strength0 0 strength_deg 0
mpirun --bind-to none -n $SLURM_NTASKS python3 scripts/iceberg.py --lstar  0.005 --dt 2.5 --height 500 --nondim_length 5 --type icebergsymm --cellfactor 1.0 --cracks 0 --strength_deg 0 --Ttop -5 --Tbot -5 --nt 1000 --relax_time 400 --lfactor 3 --smoothc 0.001 --suffix withhist
#--level 0.0 --l 3 --dt 0.02 --type relaxation --psicrit 1.0 --Gc 0.5 --damagemodel AT2higher --cellfactor 1.0 --T -5 --gv_tol 1e-3 --height 300 --tol 5e-6 --nt 500 --refine_z 0.3 --suffix linearY