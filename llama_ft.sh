python src/train_bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path /scratch/prj/inf_llmcache/hf_cache/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/1448453bdb895762499deb4176c1dd83b145fac1/ \
    --dataset openpi_entity_attr_guided \
    --template llama3 \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir ../../../ft_weights/masktom/llama-8b/openpi_entity_attr/ \
    --overwrite_cache \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_steps 100 \
    --learning_rate 5e-5 \
    --num_train_epochs 3. \
    --plot_loss 

