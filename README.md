# Source Code for the RAG Pipeline and Baselines for Information Extraction (IE)

## Environment Setup

First, create a conda environment named `RAG4IE` and install the required packages:

```bash
conda create -n RAG4IE python=3.10
conda activate RAG4IE
pip install -r requirements.txt
```

## Preparing the Datasets

**Note:** TACRED is licensed by the Linguistic Data Consortium (LDC), so we cannot directly publish the prompts or the raw results from the experiments conducted with Llama and Mistral, since the responses of these models consist of the prompts in their instruction parts. However, we have published the returned results when Llama and Mistral were integrated. Upon an official request, the data can be accessed on LDC, and the experiments can be easily replicated by following the instructions provided.

## How to Run

Change the paths and configurations under `config.ini` for your experiment.

### 1. Datasets

Place the following datasets under the `data` folder:

- **TACRED**: Licensed by the Linguistic Data Consortium (LDC). Please download it from [here](https://catalog.ldc.upenn.edu/LDC2018T24).
  
- **WikiEvents**: Available on [Github](https://github.com/LWL-cpu/SCPRG-master) and place it under the `data` folder.
  
- **RAMS**: Available on [Github](https://github.com/LWL-cpu/SCPRG-master) and place it under the `data` folder.
  
- **ACE2005**: Licensed by the Linguistic Data Consortium (LDC).

- **SemEval**: Available on [Hugging Face](https://huggingface.co/datasets/sem_eval_2010_task_8) and place it under the `data` folder.

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Compute Embeddings and Similarities

Compute embeddings and similarities for benchmark datasets in advance:

```bash
cd src/data_augmentation/embeddings
python sentence_embeddings.py
python sentence_sim.py
```

### 4. Run the Project

#### a) Inference

Replace all specific paths with `replace_with_your_path`:

```bash
python main.py \
  --test_data_path replace_with_your_path/data/rams/test.jsonlines \
  --train_data_path replace_with_your_path/data/rams/train.jsonlines \
  --similar_sentences_path replace_with_your_path/data/rams/test_rams_similarities.json \
  --dataset rams \
  --prompt_type rag \
  --model_name gpt3.5 \
  --responses_path replace_with_your_path/outputs/rams/gpt3.5_rams_rag_5_doc.json \
  --topk 5 \
  --task EE
```

```bash
python main.py
  --test_data_path replace_with_your_path \
  --train_data_path replace_with_your_path \
  --similar_sentences_path replace_with_your_path \
  --dataset semeval \
  --prompt_type rag \
  --model_name LLaMA3-8b-instruct \
  --responses_path replace_with_your_path \
  --topk 10 \
  --task RE
```
#### b) Training

Similarly, replace specific paths with `replace_with_your_path`:

```bash
accelerate launch \
  --config_file replace_with_your_path/src/train/configs/sft_zero1.yaml \
  --num_processes 4 \
  --num_machines 1 \
  --machine_rank 0 \
  --deepspeed_multinode_launcher standard \
  replace_with_your_path/src/train/train_llm.py \
  --experiment_name EAE_train_no_retrieval \
  --model_path replace_with_your_path/models/Meta-Llama-3-8B-Instruct/ \
  --max_ckpts 3 \
  --max_seq_len 2048 \
  --gradient_accumulation_steps 16 \
  --output_dir ./ckpts \
  --log_dir ./train_logs \
  --n_epochs 3 \
  --train_bsz_per_gpu 2 \
  --eval_bsz_per_gpu 2 \
  --learning_rate 2e-5 \
  --eval_step 100 \
  --save_step 100 \
  --gradient_checkpointing
```
