#!/bin/bash
#SBATCH -t 0-010:00:00                    # time limit set to 1 hour
#SBATCH --mem=16G                        # reserve 16GB of memory
#SBATCH -J TreeVAE_Run                   # the job name
#SBATCH --mail-type=END,FAIL,TIME_LIMIT  # send notification emails
#SBATCH -n 5                             # use 5 tasks (processes)
#SBATCH --cpus-per-task=1               # use 1 thread per task
#SBATCH -N 1                             # request slots on 1 node
#SBATCH --gres=gpu:v100:1                    # request 1 Volta V100 GPU
#SBATCH --partition=gpuidle              # partition to use (same as before)
#SBATCH --output=logs/test_trained_model_%j_out.txt      # capture output (%j = job id)
#SBATCH --error=logs/test_trained_model_%j_err.txt       # and error streams

# module load anaconda3/latest
# . $ANACONDA_HOME/etc/profile.d/conda.sh

# conda activate treevae_env  # use the same environment as run_main.sbatch  

# Create logs directory if it doesn't exist
mkdir -p logs
   
# launch the python script with SLURM variables    
# python main.py 
# export a unique seed each time, or use $SEED from sbatch  

python test_trained_model.py \
    --model_path "/scratch/bep91wab/projectTreeVAE_Pytorch/treevae/models/experiments/mnist/20250922-030449_a4181" \
    --test_type mnist_only \
    --seed 17   


# conda deactivate
