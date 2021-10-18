#!/bin/sh -v
#PBS -e /mnt/home/abstrac01/reinis_freibergs/logs
#PBS -o /mnt/home/abstrac01/reinis_freibergs/logs
#PBS -q batch
#PBS -l nodes=1:ppn=12:gpus=1:shared,feature=k40
#PBS -l mem=60gb
#PBS -l walltime=00:10:00


module load conda
eval "$(conda shell.bash hook)"
conda activate conda_env
export LD_LIBRARY_PATH=~/.conda/envs/conda_env/lib:$LD_LIBRARY_PATH

cd /mnt/home/abstrac01/reinis_freibergs
python ./12_2_classification_template.py -learning_rate 1e-3 -is_cuda True


