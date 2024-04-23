#!/bin/bash

#SBATCH --job-name=Llama-8b-ft
#SBATCH --partition=cllm_gpu
#SBATCH --gpus=1
#SBATCH --time=5-00
#SBATCH --constraint a100_80g
#SBATCH --cpus-per-task=1
#SBATCH --mem=80GB
#SBATCH --open-mode append
#SBATCH -o llama-8b.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hainiu.xu@kcl.ac.uk
    
python src/train_bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path /scratch/prj/inf_llmcache/hf_cache/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/1448453bdb895762499deb4176c1dd83b145fac1/
    --dataset openpi_entity_guided \
    --template llama3 \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir ../../../ft_weights/masktom/llama-8b/openpi_entity_guided_weights/ \
    --overwrite_cache \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_steps 100 \
    --learning_rate 5e-5 \
    --num_train_epochs 2 \
    --plot_loss \
    --quantization_bit 4 \
    --fp16 \
    --upcast_layernorm
