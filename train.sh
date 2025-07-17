#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1                # 1 computer nodes
#SBATCH --ntasks-per-node=1      # 1 MPI tasks on EACH NODE
#SBATCH --cpus-per-task=8        # 4 OpenMP threads on EACH MPI TASK
#SBATCH --gres=gpu:a100:1             # Using 1 GPU card
#SBATCH --mem=256GB               # Request 50GB memory
#SBATCH --time=0-23:59:00        # Time limit day-hrs:min:sec
#SBATCH --output=log/%j.log   # Standard output
#SBATCH --error=log/%j.err    # Standard error log

# CUDA_VISIBLE_DEVICES=0 
source .venv/bin/activate

python train.py
# python train.py --name fpn --NECK :FPN --lwt_alpha float:10.0 
# python train.py --name loss_cons --loss_components lists:contrastive,lib_loss,p1_bce,p2_lwt,p2_reg
# python train.py --name bank_1 --cse_n_lib int:1
