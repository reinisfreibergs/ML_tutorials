#!/bin/sh -v
#PBS -e /mnt/beegfs2/home/abstrac01/reinis_freibergs/logs
#PBS -o /mnt/beegfs2/home/abstrac01/reinis_freibergs/logs
#PBS -q batch
#PBS -l nodes=1:ppn=8:gpus=1:shared,feature=v100
#PBS -l mem=40gb
#PBS -l walltime=24:00:00


module load conda
eval "$(conda shell.bash hook)"
conda activate conda_env
export LD_LIBRARY_PATH=~/.conda/envs/conda_env/lib:$LD_LIBRARY_PATH

cd /mnt/beegfs2/home/abstrac01/reinis_freibergs
python ./11.1_transformer_template.py -learning_rate 1e-2 &
python ./11.1_transformer_template.py -learning_rate 1e-3 &
python ./11.1_transformer_template.py -learning_rate 1e-4
wait


