#!/bin/bash -l

#SBATCH --output=/data/hpcdata/users/dancha/kraken/%j.%N.out
#SBATCH --error=/data/hpcdata/users/dancha/kraken/%j.%N.err
#SBATCH --chdir=/data/hpcdata/users/dancha/kraken/kraken3/

#SBATCH --time=01:00:00
#SBATCH --job-name=k3test
#SBATCH --nodes=1
#SBATCH --partition=short
#SBATCH --account=short
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --mem=128gb

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/data/hpcdata/users/dancha/miniforge3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/data/hpcdata/users/dancha/miniforge3/etc/profile.d/conda.sh" ]; then
        . "/data/hpcdata/users/dancha/miniforge3/etc/profile.d/conda.sh"
    else
        export PATH="/data/hpcdata/users/dancha/miniforge3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

export HOME=/data/hpcdata/users/dancha

echo "START `date +%F\ %T`"

conda activate /data/hpcdata/users/dancha/conda-envs/fenicsx

COMMAND="mpirun -n 2 python /data/hpcdata/users/dancha/kraken/kraken3/scripts/iceberg_hpc.py"

echo "Running $COMMAND"
eval $COMMAND

echo "FINISH `date +%F\ %T`"
