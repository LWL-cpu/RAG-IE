"""Code for finetune_huatuo"""

from jinja2 import Template
import os
import copy
import json
import torch
import logging
import argparse
from transformers.generation.utils import LogitsProcessorList
from transformers.generation.logits_process import LogitsProcessor

from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import wandb
from accelerate import Accelerator, DeepSpeedPlugin
import transformers
from transformers import set_seed, get_cosine_schedule_with_warmup
import datasets
import shutil
import json
import random
from process_datasets import process_instruction, read_test, calculate_precision_recall_f1, calculate_precision_recall_f1_strict, read_test_label, post_process

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, LlamaForCausalLM
# from models.tokenization_moss import MossTokenizer
os.umask(0)


logger = logging.getLogger(__name__)
logging.basicConfig(level='INFO')

def _tokenize_fn(text, tokenizer, max_seq_length):
    tokenized = tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=max_seq_length,
            truncation=True,
        )
   
    input_ids = tokenized.input_ids
    input_ids_lens = input_ids.ne(tokenizer.pad_token_id).sum().item()
    attention_mask = tokenized.attention_mask

    return dict(
        input_ids=input_ids,
        input_ids_lens=input_ids_lens,
        attention_mask=attention_mask
    )

class EAE_train_dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, dataset, tokenizer, max_length, rag=False):
        self.tokenizer = tokenizer
        self.data = process_instruction(data_dir, dataset, rag)
        self.tokenizer = tokenizer
        self.max_seq_len = max_length
        chat_template = tokenizer.chat_template
        self.template = Template(chat_template)
        self.debug = True

    def __getitem__(self, index):
        return self.data[index]
    
    def get_prompt(self,da):
        q = da["input"]
        a = da['output']

        answer = a + self.tokenizer.eos_token
        # input =  self.template.render(messages=[{"role": "user", "content": q},{"role": "assistant", "content": a}],bos_token=self.tokenizer.bos_token,add_generation_prompt=False)
        query_ids = self.tokenizer.encode(q, add_special_tokens= False)
        answer_ids = self.tokenizer.encode(answer, add_special_tokens= False)
        # query = self.template.render(messages=[{"role": "user", "content": q}],bos_token=self.tokenizer.bos_token,add_generation_prompt=True)
        input_ids = query_ids + answer_ids
        labels = [-100]*len(query_ids) + answer_ids
        assert len(labels) == len(input_ids)
        return {"input_ids": input_ids, "labels": labels}     
    
    def collate_fn(self, batch):
        data = [ self.get_prompt(da) for da in batch]
        input_ids = [item["input_ids"] for item in data]
        labels = [item["labels"] for item in data]
        max_len = max(len(x) for x in input_ids)
        max_len = min(max_len,self.max_seq_len)
        input_ids = [ item[:max_len] + [self.tokenizer.eos_token_id]*(max_len-len(item)) for item in input_ids]
        labels = [item[:max_len] + [-100]*(max_len-len(item)) for item in labels]
        if self.debug:
            print(self.tokenizer.decode(input_ids[-1]))
            print(self.tokenizer.decode([0 if x == -100 else x for x in labels[-1]]))
            self.debug = False
        return {
                "input_ids": torch.LongTensor(input_ids),
                "labels": torch.LongTensor(labels),
            }
            
    def __len__(self):
        return len(self.data)

class EAE_test_dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, dataset, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.data = read_test(data_dir, dataset)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.debug = True

    def __getitem__(self, index):
        item = self.data[index]
        item["index"] = index
        
        return item

    def collate_fn(self, batch):
        batch_input = [item["input"] for item in batch]
        batch_index = [item["index"] for item in batch]
        input_ids = self.tokenizer(batch_input, return_tensors='pt', max_length=self.max_length, padding=True, truncation=True).input_ids
        attention_masks = self.tokenizer(batch_input, return_tensors='pt', max_length=self.max_length, padding=True, truncation=True).attention_mask
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "index": batch_index
        }
        
    def __len__(self):
        return len(self.data)

def get_response(inputs,outputs,tokenizer,num_return):
    responses_list=[]
    # for output in outputs:
    # responses = [tokenizer.decode(output,skip_special_tokens=True) for output in outputs]
    for i, output in enumerate(outputs):
        generated_output = tokenizer.decode(output, skip_special_tokens=True)
        responses_list.append(post_process(generated_output.split('Arguments: ')[-1]))
        # print(generated_output)
    return responses_list

class SFTMetric:
    def __init__(self, device):
        self.n_step = 0
        self.right = torch.Tensor([0]).to(device=device)
        self.total = torch.Tensor([0]).to(device=device)
        self.total_loss = torch.Tensor([0]).to(device=device)
        self.world_size = dist.get_world_size()

    def __call__(self, logits, labels, loss):
        return self.update(logits, labels, loss)

    def update(self, logits, labels, loss):
        self.n_step += 1
        with torch.no_grad():
            shift_preds = logits[..., :-1, :].argmax(dim=-1)
            shift_labels = labels[..., 1:]
            self.right += (shift_preds == shift_labels).masked_fill(shift_labels.eq(-100), 0).sum().item()
            self.total += (shift_labels != -100).sum().item()
            self.total_loss += loss.item()

    def get_metric(self, reset=True):
        dist.all_reduce(self.right, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(self.total, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(self.total_loss, op=torch.distributed.ReduceOp.SUM)

        acc = (self.right / self.total).item()
        loss = self.total_loss.item() / (self.world_size * self.n_step)

        if reset:
            self.n_step = 0
            self.right.fill_(0)
            self.total.fill_(0)
            self.total_loss.fill_(0)
        return acc, loss
    

def train(args):
    # accelerator = Accelerator(mixed_precision='fp16', gradient_accumulation_steps=args.gradient_accumulation_steps)
 
    # accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=args.gradient_accumulation_steps) 
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps) 
    if args.not_shuffle_train_loader:
        accelerator.print('Will not shuffle train data loader.')

    # if accelerator.is_main_process:
    #     wandb.init(project = args.experiment_name, config=args, dir=args.log_dir)
    
    accelerator.print(f'args:\n{args}')

    accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = args.train_bsz_per_gpu
    accelerator.state.deepspeed_plugin.deepspeed_config['train_batch_size'] = args.train_bsz_per_gpu*dist.get_world_size()*accelerator.gradient_accumulation_steps

    left_tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, padding_side='left')
    # tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True, attn_implementation='flash_attention_2')
    
    # special_token_dict = {}
    # if left_tokenizer.pad_token is None:
    #     left_tokenizer.pad_token = '<PAD>'
        # left_tokenizer.pad_token_id = left_tokenizer.eos_token_id
    if left_tokenizer.pad_token is None:
        left_tokenizer.pad_token = '<PAD>'

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # with open(args.data_dir) as f:
    #     data = json.load(f)
    #     train_data = data

    # accelerator.print(f'train_data shuffle: {(not args.not_shuffle_train_loader)}')
    
    train_dataset =EAE_train_dataset(args.train_file, args.train_dataset, left_tokenizer, args.max_seq_len, args.rag)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_bsz_per_gpu, shuffle=True, drop_last=True, collate_fn=train_dataset.collate_fn)

    dataset = EAE_test_dataset(args.test_file, args.test_dataset, left_tokenizer, args.max_seq_len)
    val_dataloader = DataLoader(dataset, batch_size=4, shuffle=False, drop_last=False, collate_fn=dataset.collate_fn)

    args.num_return = 1
    gen_kwargs = {'max_new_tokens':40, 'do_sample':False, 'temperature':1.0 }

    num_training_steps = int(len(train_dataloader) * (args.n_epochs)) // accelerator.gradient_accumulation_steps
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_rates * num_training_steps), num_training_steps=num_training_steps)

    accelerator.print(f'gradient_accumulation_steps:{accelerator.gradient_accumulation_steps} data_dir:{args.data_dir} lr:{args.learning_rate} num_training_steps:{num_training_steps}')
    
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, val_dataloader, lr_scheduler)

    if args.checkpoint_path:
        if os.path.isfile(os.path.join(args.checkpoint_path, "scheduler.bin")) and \
           os.path.isfile(os.path.join(args.checkpoint_path, "training_state.pt")):
            accelerator.load_state(args.checkpoint_path)
            # 保存半精度模型
            # if accelerator.is_main_process:
            #     output_dir = os.path.join(args.checkpoint_path, 'tfmr')
            #     model.save_pretrained(output_dir,state_dict=accelerator.get_state_dict(model))
            #     left_tokenizer.save_pretrained(output_dir)
            #     copy_files = []
            #     for item in os.listdir(args.model_path):
            #         if os.path.exists(os.path.join(output_dir,item)):
            #             continue
            #         if item.startswith("pytorch_model") and item.endswith(".bin"):
            #             continue
            #         if item.endswith(".index.json") or item.endswith(".safetensors"):
            #             continue
            #         s = os.path.join(args.model_path, item)
            #         if os.path.isfile(s):
            #             shutil.copy(s, os.path.join(output_dir,item))
            #         copy_files.append(item)
            #     accelerator.print(f'huggingface model save in {output_dir}, copy file:{copy_files}')
            training_state = torch.load(os.path.join(args.checkpoint_path, "training_state.pt"))
            start_epoch = training_state["epoch"]
            start_step = training_state["step"]+1
            global_step = training_state["global_step"]
            accelerator.print(f"Checkpoint Loaded at {start_epoch} epoch, {start_step} step and {global_step} global step")
            accelerator.print(f"Loading trained model :{args.checkpoint_path}")
        else:
            raise ValueError(f"Checkpoint not found at: {args.checkpoint_path}")
    else:
        start_epoch = 0
        start_step = 0
        global_step = 0

    if args.save_step <= 0:
        args.save_step=len(train_dataloader) // 15
        accelerator.print(f'Save step setted to {args.save_step}')
    if args.eval_step <= 0:
        args.eval_step=len(train_dataloader) // 30
        accelerator.print(f'Eval step setted to {args.eval_step}')

    best_score = 0
    save_score = 0
    # global_step = 0
    metric = SFTMetric(device=torch.cuda.current_device())

    #Code for saving checkpoints
    def save_checkpoint(epoch, step, global_step):
        save_dir = os.path.join(args.output_dir, f"checkpoint-{epoch}-{global_step}")
        if accelerator.is_main_process:
            checkpoint_files = os.listdir(args.output_dir)
            checkpoint_files = [file for file in checkpoint_files if file.startswith("checkpoint-")]
            num_checkpoints = len(checkpoint_files)
            if args.max_ckpts>0:
                if num_checkpoints >= args.max_ckpts:
                    checkpoint_files.sort(key=lambda x: os.path.getctime(os.path.join(args.output_dir, x)))
                    oldest_checkpoint = checkpoint_files[0]
                    shutil.rmtree(os.path.join(args.output_dir, oldest_checkpoint))        
            os.makedirs(save_dir, exist_ok=True)
            output_dir = os.path.join(save_dir, 'tfmr')
            if accelerator.state.deepspeed_plugin.zero_stage!=3:
                model.save_pretrained(output_dir,state_dict=accelerator.get_state_dict(model))
            left_tokenizer.save_pretrained(output_dir)
            copy_files = []
            for item in os.listdir(args.model_path):
                if os.path.exists(os.path.join(output_dir,item)):
                    continue
                if item.startswith("pytorch_model") and item.endswith(".bin"):
                    continue
                if item.endswith(".index.json") or item.endswith(".safetensors"):
                    continue
                s = os.path.join(args.model_path, item)
                if os.path.isfile(s):
                    shutil.copy(s, os.path.join(output_dir,item))
                copy_files.append(item)
            print(f'huggingface model save in {output_dir}, copy file:{copy_files}')

        if accelerator.state.deepspeed_plugin.zero_stage==3:
            unwrap_model = accelerator.unwrap_model(model)
            unwrap_model.save_pretrained(os.path.join(save_dir, f'tfmr'),is_main_process=accelerator.is_main_process,save_function=accelerator.save,state_dict=accelerator.get_state_dict(model))
            
        accelerator.wait_for_everyone()
        accelerator.save({"epoch": epoch, "step": step, "global_step": global_step}, os.path.join(save_dir, "training_state.pt"))
        accelerator.print(f'checkpoint checkpoint-{epoch}-{global_step} is saved...')
    # 注意
    accelerator.print(accelerator.deepspeed_config)
    model.train()
    for epoch in range(start_epoch, args.n_epochs):
        train_dataloader_iterator = tqdm(enumerate(train_dataloader), total=len(train_dataloader)) if accelerator.is_main_process else enumerate(train_dataloader)
        for batch_cnt, batch in train_dataloader_iterator:
            if epoch==start_epoch and batch_cnt<start_step:
                continue

            if batch_cnt == 1 and epoch == 0:
                torch.cuda.empty_cache()

            input_ids=batch['input_ids']
            labels=batch['labels']
            # attention_mask = batch["attention_mask"]
            output = model(input_ids=input_ids, labels=labels, return_dict=True,use_cache=False)
            loss = output.loss

            metric(output.logits, labels, loss)
            acc, train_loss = metric.get_metric()
            accelerator.backward(loss)
            if (global_step+1) % accelerator.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            global_step += 1

            if accelerator.is_main_process:
                train_dataloader_iterator.set_postfix(epoch=epoch, current_step=batch_cnt, total_step=len(train_dataloader), skip=accelerator.optimizer_step_was_skipped, loss=round(train_loss, 3), acc=round(acc, 3), length=len(input_ids[0]), lr=lr_scheduler.get_last_lr()[0])

            # if global_step % 3 == 0 and accelerator.is_main_process:
            #     wandb.log({
            #         'skip': int(accelerator.optimizer_step_was_skipped),
            #         'loss': train_loss,
            #         'acc': acc,
            #         'lr': lr_scheduler.get_last_lr()[0]
            #     }, step=global_step)

            # if global_step > args.eval_step:
            if global_step % args.eval_step == 0:
                torch.cuda.empty_cache()
                model.eval() 
                dataloader_iterator = tqdm(val_dataloader, total=len(val_dataloader)) if accelerator.is_main_process else val_dataloader
                responses = []
                indices = []

                for batch in dataloader_iterator:
                    input_ids = batch["input_ids"]
                    attention_mask = batch["attention_mask"]
                    batch_indices = batch["index"]
                    # lebels = batch["labels"]
                    outputs = accelerator.unwrap_model(model).generate(input_ids, attention_mask=attention_mask, **gen_kwargs)

                    response = get_response(input_ids,outputs,left_tokenizer,args.num_return)
                    responses.extend(response)
                    indices.extend(batch_indices)
                labels = read_test_label(args.test_file, args.test_dataset)
                sorted_responses = ["[]"] * (max(indices) + 1)
                accelerator.wait_for_everyone()

                if accelerator.is_main_process:
                    for i, index in enumerate(indices):     
                        try:
                            sorted_responses[index] = responses[i]
                        except:
                            print("no_having")
                    
                    
                    precision, recall, f1_score, mention_exact_match = calculate_precision_recall_f1(predictions=sorted_responses, labels=labels)
                    print("Precision: " + str(precision))
                    print(" Recall: " + str(recall))
                    print(" F1 Score: " + str(f1_score))
                    print(" Mention Exact Match: " + str(mention_exact_match))

                    precision, recall, f1_score, mention_exact_match = calculate_precision_recall_f1_strict(predictions=sorted_responses, labels=labels)

                    print("Strict Precision: " + str(precision))
                    print("Strict  Recall: " + str(recall))
                    print("Strict  F1 Score: " + str(f1_score))
                    print("Strict  Mention Exact Match: " + str(mention_exact_match))
                        
                    

                model.train()           

            # if global_step % args.save_step == 22 and best_score > save_score:
            if global_step % args.save_step == 0:
            # if global_step % args.save_step == 40:
                accelerator.print(f'save model with best score {best_score}')
                save_score = best_score
                accelerator.wait_for_everyone()
                save_checkpoint(epoch, batch_cnt, global_step)
            
        accelerator.wait_for_everyone()
        save_checkpoint(epoch, batch_cnt, global_step)
        # Reset start_step to 0 for the next epoch if we resumed from a checkpoint
        start_step = 0


    # wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args of sft')
    # Experiment Args
    parser.add_argument('--experiment_name', type=str, default="EAE_train")
    parser.add_argument('--checkpoint_path',default=None, type=str)

    # Model Args
    parser.add_argument('--model_path', default='/sds_wangby/models/Meta-Llama-3-8B-Instruct/', type=str)
    
    # Data Args
    parser.add_argument('--not_shuffle_train_loader', action='store_true')
    parser.add_argument('--data_dir', default='', type=str)
    parser.add_argument('--output_dir', default='./ckpts', type=str)
    parser.add_argument('--max_ckpts', default=5, type=int)
    parser.add_argument('--log_dir', default='./train_logs', type=str)
    parser.add_argument('--train_file', default='/223040263/wanlong/LLM_Retreival/RAG4RE/data/rams/train.jsonlines', type=str)
    parser.add_argument('--train_dataset', default='rams', type=str)
    parser.add_argument('--test_file', default='/223040263/wanlong/LLM_Retreival/RAG4RE/data/rams/test.jsonlines', type=str)
    parser.add_argument('--test_dataset', default='rams', type=str)
    # Training Args
    parser.add_argument('--max_seq_len', default=2048, type=int)
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument('--gradient_accumulation_steps', default=4, type=int)
    parser.add_argument('--train_bsz_per_gpu', default=1, type=int)
    parser.add_argument('--eval_bsz_per_gpu', default=4, type=int)
    parser.add_argument('--weight_decay', default=0.1, type=float)
    parser.add_argument('--learning_rate', default=5e-5, type=float)
    parser.add_argument('--warmup_rates', default=0.05, type=float)
    parser.add_argument('--n_epochs', default=4, type=int)

    # Other Args
    parser.add_argument('--save_step', default=10, type=int)
    parser.add_argument('--eval_step', default=10, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--rag', default=True, type=bool)

    args = parser.parse_args()
    args.log_dir = os.path.join(args.log_dir,args.experiment_name)
    args.output_dir = os.path.join(args.output_dir,args.experiment_name)


    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)
    train(args)           
'''
 nohup accelerate launch --config_file /223040263/wanlong/LLM_Retreival/RAG4RE/src/train/configs/sft_zero1.yaml --num_processes 4 --num_machines 1 --machine_rank 0  --deepspeed_multinode_launcher  standard  /223040263/wanlong/LLM_Retreival/RAG4RE/src/train/train_llm.py    --experiment_name EAE_train_no_retrieval   --model_path /sds_wangby/models/Meta-Llama-3-8B-Instruct/     --max_ckpts 3      --max_seq_len 2048    --gradient_accumulation_steps 4     --output_dir ./ckpts    --log_dir ./train_logs     --n_epochs 4    --train_bsz_per_gpu 8     --eval_bsz_per_gpu 8     --learning_rate 2e-5     --eval_step 10    --save_step 10    --gradient_checkpointing    > train_rams_no_retrieval.log 2>&1 &

'''