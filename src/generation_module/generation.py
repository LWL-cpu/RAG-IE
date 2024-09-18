"""Generates responses for prompts using a language model defined in Hugging Face."""
"""Created by: Sefika"""
import json
import requests
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import T5Tokenizer, T5ForConditionalGeneration
import random
import os
from openai import OpenAI
import time
from retrying import retry
class GPT:
    def __init__(self, model_name='gpt-4-turbo') -> None:
        self.key_ind = 0
        self.max_wrong_time = 5
        self.model_name = model_name
        self.url = "your-url"
        self.keys = 'your-keys'
        os.environ["OPENAI_API_KEY"] = self.keys
        os.environ["OPENAI_BASE_URL"] = self.url

        assert len(self.keys) > 0, 'have no key'
        self.wrong_time = [0] * len(self.keys)
        print(f'keys: {self.keys}')
        print(f'use model of {self.model_name}')

    def init_api_keys(self):
        self.keys = []
        with open('gpt_key.txt', encoding="utf-8", mode="r") as fr:
            for l in fr:
                cols = l.split('---')
                if len(cols[0]) < 45 or len(cols[0]) > 55:
                    continue
                if len(cols) == 1:
                    cols.append('None')
                self.keys.append((cols[0], cols[1]))
        assert len(self.keys) > 0, 'have no key'
        print(f'keys: {self.keys}')
        self.wrong_time = [0] * len(self.keys)
        random.shuffle(self.keys)

    def get_api_key(self):
        self.key_ind = (self.key_ind + 1) % len(self.keys)
        return self.keys[self.key_ind]

    def gpt4_call(self, prompt):
        client = OpenAI()
        while True:
            try:
                # 初始化OpenAI客户端
                completion = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                response = completion.choices[0].message.content
                # 如果返回的消息中包含错误
                if 'error' in response:
                    # 增加当前键对应的错误次数
                    self.wrong_time[self.key_ind] += 1
                    # 如果错误次数超过过最大允许的错误次数
                    if self.wrong_time[self.key_ind] > self.max_wrong_time:
                        # 打印错误响应
                        print(response)
                        # 打印错误的键
                        print(f'Wrong key: {self.keys[self.key_ind]}')
                        # 断言失败，并输出错误信息
                        assert False, str(response)
                # 返回消息内容
                return response
            except RateLimitError as e:
                print(f'Rate limit error: {e}. Retrying...')
                time.sleep(5)  # 等待5秒后重试
                self.key_ind = (self.key_ind + 1) % len(self.keys)  # 轮换到下一个 API 密钥

    def call(self, content, args={}, showkeys=False):
        api_key, organization = self.get_api_key()
        if showkeys:
            print(api_key, organization)
        if organization == 'None':
            organization = ''

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "OpenAI-Organization": organization,
        }

        parameters = {
            "model": self.model_name,
            "messages": [{'role': 'user', 'content': content}],
            **args,
        }

        response = requests.post(
            self.url,
            headers=headers,
            json=parameters
        )
        response = json.loads(response.content.decode("utf-8"))
        if 'error' in response:
            self.wrong_time[self.key_ind] += 1
            if self.wrong_time[self.key_ind] > self.max_wrong_time:
                print(response)
                print(f'del {self.keys[self.key_ind]}')
                # del self.keys[self.key_ind]
                # del self.wrong_time[self.key_ind]
            assert False, str(response)
        return response['choices'][0]['message']['content']

    def test(self):
        for _ in range(len(self.keys)):
            try:
                print(self.call('你好', showkeys=True))
            except Exception as e:
                print(e)


def call_gpt_with_retry(gpt_instance, content, args={}, showkeys=False, max_retries=3):
    attempts = 0
    while attempts < max_retries:
        output = gpt_instance.gpt4_call(content)
        try:
            if 'error' in output:
                continue
            else:
                return output

        except json.JSONDecodeError:
            attempts += 1
            print(f"Attempt {attempts}/{max_retries} failed: Retrying...")
    
    return output
class LLM(object):
    
    def __init__(self, model_id="google/flan-t5-xl", prompt_type = "RAG"):
        """
        Initialize the LLM model
        Args:
            model_id (str, optional): model name from Hugging Face. Defaults to "google/flan-t5-xl".
        """
        self.maxmem={i:f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB' for i in range(4)}
        self.prompt_type = prompt_type
        self.model_id = model_id
        self.maxmem['cpu']='300GB'
        if model_id=="google/flan-t5-xl":
            self.model, self.tokenizer = self.get_model(model_id)
        elif model_id == "gpt4":
            self.model = GPT(model_name='gpt-4o-2024-05-13')
        elif model_id == "gpt3.5" or model_id == "chatgpt" or model_id == "ChatGPT":
            self.model = GPT(model_name='gpt-3.5-turbo')
        else: 
            self.model, self.tokenizer = self.get_model_decoder(model_id)
        
        
    def get_model(self, model_id="google/flan-t5-xl"):
        """_summary_

        Args:
            model_id (str, optional): LLM name at HuggingFace . Defaults to "google/flan-t5-xl".

        Returns:
            model: model from Hugging Face
            tokenizer: tokenizer of this model
        """
        tokenizer = T5Tokenizer.from_pretrained(model_id)
 
        model = T5ForConditionalGeneration.from_pretrained(model_id,
                                                    load_in_8bit=False, 
                                                    torch_dtype=torch.float16,
                                                    max_memory=self.maxmem)
        return model,tokenizer
    
    def get_prediction(self, prompt, task="RE", length=80):
        """_summary_

        Args:
            model : loaded model
            tokenizer: loaded tokenizer
            prompt (str): prompt to generate response 
            length (int, optional): Response length. Defaults to 30.

        Returns:
            response (str): response from the model
        """
        if self.model_id == "gpt4" or self.model_id == "gpt3.5" or self.model_id == "ChatGPT":
            response = call_gpt_with_retry(self.model, prompt)
            # print()
            return response

        else:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            inputs = self.tokenizer(prompt, max_length=4096, padding=True, truncation=True, add_special_tokens=True, return_tensors="pt").input_ids.to('cuda:0')

            outputs = self.model.generate(inputs, max_new_tokens=length)

            responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            responses_new = []

            for item in responses:
                if task == "RE":
                    responses_new.append(item.split("Answer:")[-1])
                else:
                    responses_new.append(item.split("Arguments:")[-1])

            return responses_new
    
    def get_model_decoder(self, model_id="/sds_wangby/models/Meta-Llama-3-8B-Instruct"):
        """loades the model from Hugging Face such llama and mistral

        Args:
            model_id (str, optional): _description_. Defaults to "meta-llama/Llama-2-7b-chat-hf".

        Returns:
            model: loaded model
            tokenizer: loaded tokenizer
        """

        if self.prompt_type == "memory":
            from src.models.modeling_gemma import GemmaForCausalLM
            tokenizer = AutoTokenizer.from_pretrained('gemma-2b', use_fast=True,
        split_special_tokens=False,
        padding_side="left", )
            model = GemmaForCausalLM.from_pretrained('/data/code/lwl/RAG4RE/gemma-2b-memory/gemma-2b/',
                                                         load_in_8bit=False,
                                                         torch_dtype=torch.bfloat16,
                                                         max_memory=self.maxmem).to('cuda:0')
            model = init_adapter(model, True, )
            state_dict = torch.load('/data/code/lwl/DeepKE-main/example/llm/InstructKGC/gemma-2b-v2/all_params/model_epoch_5.996989463120923.pth', map_location='cuda:0')
            # new_state_dict = {}
            # for key in state_dict:
            #     state_dict[key] = state_dict[key].to(torch.bfloat16)


            model.load_state_dict(state_dict, strict=True)
            model.eval()
        elif self.prompt_type == "memory_unsupervised":
            from src.models.modeling_gemma import GemmaForCausalLM
            tokenizer = AutoTokenizer.from_pretrained('gemma-2b')
            model = GemmaForCausalLM.from_pretrained('/data/code/lwl/RAG4RE/gemma-2b-memory/gemma-2b/',
                                                     load_in_8bit=False,
                                                     torch_dtype=torch.bfloat16,
                                                     max_memory=self.maxmem).to('cuda:0')
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
            model = AutoModelForCausalLM.from_pretrained(model_id,
                                                         load_in_8bit=False,
                                                         torch_dtype=torch.bfloat16,
                                                        max_memory=self.maxmem).to('cuda:0')
        return model, tokenizer
