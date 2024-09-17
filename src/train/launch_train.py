import subprocess

command = [
    "accelerate", "launch",
    "--config_file", "/223040263/wanlong/LLM_Retreival/RAG4RE/src/train/configs/sft_zero1.yaml",
    "--num_processes", "4",
    "--num_machines", "1",
    "--machine_rank", "0",
    "--deepspeed_multinode_launcher", "standard",
    "/223040263/wanlong/LLM_Retreival/RAG4RE/src/train/train_llm.py",
    "--experiment_name", "EAE_train_no_retrieval",
    "--model_path", "/sds_wangby/models/Meta-Llama-3-8B-Instruct/",
    "--max_ckpts", "3",
    "--max_seq_len", "1024",
    "--gradient_accumulation_steps", "2",
    "--output_dir", "./ckpts",
    "--log_dir", "./train_logs",
    "--n_epochs", "3",
    "--train_bsz_per_gpu", "2",
    "--eval_bsz_per_gpu", "2",
    "--learning_rate", "5e-5",
    "--eval_step", "-1",
    "--save_step", "-1",
    "--gradient_checkpointing"
]

with open("train_rams_no_retrieval.log", "w") as log_file:
    process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
    process.communicate()
