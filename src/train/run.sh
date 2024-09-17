accelerate launch --config_file /223040263/wanlong/LLM_Retreival/RAG4RE/src/train/configs/sft_zero1.yaml --num_processes 4 --num_machines 1 --machine_rank 0 --deepspeed_multinode_launcher standard /223040263/wanlong/LLM_Retreival/RAG4RE/src/train/train_llm.py --experiment_name EAE_train_no_retrieval --model_path /sds_wangby/models/Meta-Llama-3-8B-Instruct/ --max_ckpts 4 --max_seq_len 4096 --gradient_accumulation_steps 16 --output_dir ./rag_combined --log_dir ./train_logs --n_epochs 6 --train_bsz_per_gpu 2 --eval_bsz_per_gpu 2 --learning_rate 2e-5 --eval_step 120000 --save_step 120000 --gradient_checkpointing > train_combined_with_retrieval.log 2>&1
accelerate launch --config_file /223040263/wanlong/LLM_Retreival/RAG4RE/src/train/configs/sft_zero1.yaml --num_processes 4 --num_machines 1 --machine_rank 0 --deepspeed_multinode_launcher standard /223040263/wanlong/LLM_Retreival/RAG4RE/src/train/train_llm_rag_10.py --experiment_name EAE_train_no_retrieval --model_path /sds_wangby/models/Meta-Llama-3-8B-Instruct/ --max_ckpts 4 --max_seq_len 4096 --gradient_accumulation_steps 16 --output_dir ./rag_10_combined --log_dir ./train_logs --n_epochs 6 --train_bsz_per_gpu 2 --eval_bsz_per_gpu 2 --learning_rate 2e-5 --eval_step 120000 --save_step 120000 --gradient_checkpointing > train_combined_with_retrieval_10.log 2>&1

accelerate launch --config_file /223040263/wanlong/LLM_Retreival/RAG4RE/src/train/configs/sft_zero1.yaml --num_processes 4 --num_machines 1 --machine_rank 0  --deepspeed_multinode_launcher  standard  /223040263/wanlong/LLM_Retreival/RAG4RE/src/train/train_llm_no_rag.py    --experiment_name train_llm_no_rag  --model_path /sds_wangby/models/Meta-Llama-3-8B-Instruct/     --max_ckpts 7      --max_seq_len 1024    --gradient_accumulation_steps 16     --output_dir ./train_llm_no_rag    --log_dir ./train_logs     --n_epochs 10    --train_bsz_per_gpu 2     --eval_bsz_per_gpu 2     --learning_rate 2e-5     --eval_step 120000    --save_step 120000    --gradient_checkpointing  > train_combined_no_retrieval.log 2>&1 &