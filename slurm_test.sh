#!/bin/bash -l

#SBATCH --output=/data/hpcdata/users/jambyr/kraken/kraken3/%j.%N.out
#SBATCH --error=/data/hpcdata/users/jambyr/kraken/kraken3/%j.%N.err
#SBATCH --chdir=/data/hpcdata/users/jambyr/kraken/kraken3

#SBATCH --time=7-00:00:00
#SBATCH --job-name=k3test
#SBATCH --nodes=1
#SBATCH --partition=long
#SBATCH --account=long
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=18
#SBATCH --mem=256gb

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/data/hpcdata/users/jambyr/miniforge3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/data/hpcdata/users/jambyr/miniforge3/etc/profile.d/conda.sh" ]; then
        . "/data/hpcdata/users/jambyr/miniforge3/etc/profile.d/conda.sh"
    else
        export PATH="/data/hpcdata/users/jambyr/miniforge3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

export HOME=/data/hpcdata/users/jambyr

echo "START `date +%F\ %T`"

conda activate /data/hpcdata/users/jambyr/miniforge3/envs/fenicsx-env

COMMAND="mpirun -n 2 python scripts/iceberg.py"

echo "Running $COMMAND"
eval $COMMAND

echo "FINISH `date +%F\ %T`"
