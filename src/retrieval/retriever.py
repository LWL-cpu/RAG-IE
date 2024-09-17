import os
import sys
import configparser
from data_augmentation.embeddings.sentence_embeddings import *
from data_augmentation.prompt_generation.prompt_generation import generate_prompts_EE, generate_prompts_RE
from generation_module.generation import LLM
import configparser
import re
import json
from collections import defaultdict
from utils import read_json, write_json
from collections import Counter
import string
import numpy as np
import random
from sklearn.metrics import precision_recall_fscore_support
PACKAGE_PARENT = '.'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
PREFIX_PATH = "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[:-1]) + "/"
import tqdm

def process_prompt_gpt(prompt, llm_instance, i):
    response = llm_instance.get_prediction(prompt)
    if len(response[0]) > 800:
        input_length = len(prompt)
        response = response[0].split('Arguments: ')[-1]
    print(f"generate {i} th example")
    return i, post_process(response)

def read_tacred(path):
    new_data = []
    data = read_json(path)
    for item in data:
        new_item = {}
        new_item["sentence"] = " ".join(item["token"])
        new_item["subj"] = " ".join(item["token"][item["subj_start"]: item["subj_end"]+1])
        new_item["obj"] = " ".join(item["token"][item["obj_start"]: item["obj_end"]+1])
        new_item["label"] = item["relation"]
        new_data.append(new_item)
    return new_data

def benchmark_data_augmentation_call_EE(args):
    test_data_path = args.test_data_path
    train_data_path = args.train_data_path
    similar_sentences_path = args.similar_sentences_path
    
    similar_sentences = read_json(similar_sentences_path)
    print(len(similar_sentences))
    test_data = read_test(test_data_path, test_data_path.split('/')[-2])
    train_data_ = read_train(train_data_path, train_data_path.split('/')[-2])
    train_data = [item[1] for item in train_data_]


    dataset = args.dataset
    prompt_type = args.prompt_type
    model_name = args.model_name

    if prompt_type == "rag":
        print(f"RAG {args.topk} documents!")
        output_responses_path = args.responses_path
        prompts = generate_prompts_EE(test_data, similar_sentences, dataset, None, prompt_type, args.topk)
    elif prompt_type == "random_rag":
        print(f"Random RAG {args.topk} documents!")
        for item in similar_sentences:
            random_docs = random.sample(train_data, args.topk)
            item["similar_sentence"] = random_docs
        output_responses_path = args.responses_path
        prompts = generate_prompts_EE(test_data, similar_sentences, dataset, None, prompt_type, args.topk)
    elif prompt_type == "no_diversity_rag":
        print(f"No_diversity RAG {args.topk} documents!")
        for i, item in enumerate(similar_sentences):
            test_type = test_data[i][0]
            similar_sentence = item["similar_sentence"]
            event_types = item["event_type"]
            similar_sentence_new = []
            event_types_new = []
            for ii, it in enumerate(similar_sentence):
                if test_type == event_types[ii]:
                    similar_sentence_new.append(it)
                    event_types_new.append(event_types[ii])
            if len(similar_sentence_new) < 10:
                needed_items = 10 - len(similar_sentence_new)
                additional_items = random.sample(similar_sentence, needed_items)
                similar_sentence_new.extend(additional_items)
            similar_sentences[i]["similar_sentence"] = similar_sentence_new
            similar_sentences[i]["event_type"] = event_types_new
        output_responses_path = args.responses_path
        prompts = generate_prompts_EE(test_data, similar_sentences, dataset, None, prompt_type, args.topk)
    elif prompt_type == "diversity_rag":
        print(f"Diversity RAG {args.topk} documents!")
        for i, item in enumerate(similar_sentences):
            test_type = test_data[i][0]
            similar_sentence = item["similar_sentence"]
            event_types = item["event_type"]
            similar_sentence_new = []
            other_type_sentence = []
            event_types_new = []
            diversity_num = int(args.topk * 0.4)
            for ii, it in enumerate(similar_sentence):
                if test_type != event_types[ii]:
                    other_type_sentence.append(it)
            for ii in range(args.topk - diversity_num):
                similar_sentence_new.append(similar_sentence[ii])
            for ii in range(diversity_num):
                similar_sentence_new.extend(random.sample(other_type_sentence[args.topk:], diversity_num))
          
            similar_sentences[i]["similar_sentence"] = similar_sentence_new
        output_responses_path = args.responses_path
        prompts = generate_prompts_EE(test_data, similar_sentences, dataset, None, prompt_type, args.topk)
        print()
    else:
        output_responses_path = args.responses_path
        prompts = generate_prompts_EE(test_data, None, dataset, None, prompt_type)
    
    llm_instance = LLM(model_name, prompt_type)
    
    responses = []
    # from transformers import AutoTokenizer
    # from jinja2 import Template
    # tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side='left')
    # template = Template(tokenizer.chat_template)
    # prompts = [template.render(messages=[{"role": "user", "content": prompt}],bos_token=tokenizer.bos_token,add_generation_prompt=True) for prompt in prompts]

    if model_name == "gpt4" or model_name == "gpt3.5" or model_name == "ChatGPT": 
        import concurrent.futures
        responses = [None] * len(prompts)
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(process_prompt_gpt, prompt, llm_instance, i)
                for i, prompt in enumerate(prompts)
            ]

        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing prompts"):
            i, result = future.result()
            responses[i] = result  # 根据索引i将结果放入responses中
    else:    
        batch_size = 16
        num_batches = (len(prompts) + batch_size - 1) // batch_size  # Calculate total number of batches
        progress_bar = tqdm.tqdm(total=num_batches, desc="Processing prompts in batches")
        
        for i in range(0, len(prompts), batch_size):
             # Get the current batch of prompts
            batch_prompts = prompts[i:i + batch_size]
        
            # Obtain predictions for the current batch
            batch_responses = llm_instance.get_prediction(batch_prompts,task="EE")
        
            for response in batch_responses:
                # Assuming `response` is a list or similar structure where the first item is the text
                responses.append(post_process(response))
        
            # Optional: Update progress bar manually if using tqdm
            # This updates the progress bar based on the number of batches processed
            # tqdm.tqdm.update(i // batch_size + 1)
            progress_bar.update(1)
        progress_bar.close() 
    labels = read_test_label(test_data_path, test_data_path.split('/')[-2])
    precision, recall, f1_score, mention_exact_match = calculate_precision_recall_f1(predictions=responses, labels=labels)
    print("Precision: " + str(precision))
    print(" Recall: " + str(recall))
    print(" F1 Score: " + str(f1_score))
    print(" Mention Exact Match: " + str(mention_exact_match))

    precision, recall, f1_score, mention_exact_match = calculate_precision_recall_f1_strict(predictions=responses, labels=labels)

    print("Strict Precision: " + str(precision))
    print("Strict  Recall: " + str(recall))
    print("Strict  F1 Score: " + str(f1_score))
    print("Strict  Mention Exact Match: " + str(mention_exact_match))
    
    # write_json(output_prompts_path, prompts)
    write_json(output_responses_path, responses)
def extract_unique_relations(train_data):
    """Extracts a list of unique relations from a list of data items.

    Args:
        train_data (list): A list of dictionaries, each containing a 'relation' key.

    Returns:
        list: A list of unique relations.
    """
    # Using a set to avoid duplicates
    relations_set = set()

    # Iterate over each item in the train_data
    for item in train_data:
        # Add the 'relation' value to the set
        if 'label' in item:  # Check if the 'relation' key exists
            relations_set.add(item['label'].split(":")[-1])

    # Convert the set to a list before returning
    unique_relations = list(relations_set)
    return unique_relations
def benchmark_data_augmentation_call_RE(args):
    test_data_path = args.test_data_path
    train_data_path = args.train_data_path
    relations_path = args.relation_path
    similar_sentences_path = args.similar_sentences_path
    test_label_path = args.test_label_path
    train_label_path = args.train_label_path
    similar_sentences = read_json(similar_sentences_path)
    
    if args.dataset == "semeval":
        relations = read_json(relations_path)
        relations_ = relations['relation']['names']
        relations = [item.split('(')[0] for item in relations_]
        train_labels_ = read_json(train_label_path)
        train_labels = [relations[item] for item in train_labels_]
        test_labels = read_json(test_label_path)
        labels = [relations[item] for item in test_labels]
        test_data = read_json(test_data_path)
        train_data = read_json(train_data_path)
        
    else:

        train_data = read_tacred(train_data_path)
        test_data = read_tacred(test_data_path)
        labels = [item['label'].split(":")[-1] for item in test_data]
        relations = extract_unique_relations(train_data)

    dataset = args.dataset
    prompt_type = args.prompt_type
    model_name = args.model_name


    if prompt_type == "rag":
        print(f"RAG {args.topk} documents!")
        output_responses_path = args.responses_path

        for item in similar_sentences:
            tmp_sents = item["similar_sentence"]
            tmp_labels = item["label"]
            new_sents_list = []
            if dataset != "semeval":
                subjs = item["subj"]
                objs = item["obj"]     
            for i, sent in enumerate(tmp_sents):
                if dataset == "semeval":
                    relation_prefix = "Relation between <e1> and <e2> is "
                    new_sent = sent + " The " + relation_prefix + tmp_labels[i].split('(')[0] 
                else:
                    relation_prefix = "Relation between head entity {} and tail entity {} is ".format(subjs[i], objs[i])
                    new_sent = sent['sentence'] + " The " + relation_prefix + tmp_labels[i].split(":")[-1] 
                new_sents_list.append(new_sent)
            item["similar_sentence"] = new_sents_list
        
        prompts = generate_prompts_RE(test_data, relations, similar_sentences,  dataset, prompt_type, args.topk)
    elif prompt_type == "random_rag":
        print(f"Random RAG {args.topk} documents!")
        train_data_new = []
        for i, item in enumerate(train_data):
            if dataset == "semeval":
                relation_prefix = "Relation between <e1> and <e2> is "
                new_sent = item + relation_prefix + train_labels[i].split('(')[0]
            else:
                relation_prefix = "Relation between head entity {} and tail entity {} is ".format(item["subj"], item["obj"])
                new_sent = item["sentence"] + relation_prefix + item['label'].split(":")[-1] 
            train_data_new.append(new_sent)
        for item in similar_sentences:
            item["similar_sentence"] = random.sample(train_data_new, args.topk)

        output_responses_path = args.responses_path
        prompts = generate_prompts_RE(test_data, relations, similar_sentences,  dataset, prompt_type, args.topk)
    else:
        output_responses_path = args.responses_path
        prompts = generate_prompts_RE(test_data, relations, similar_sentences,  dataset, prompt_type, args.topk)
    
    llm_instance = LLM(model_name, prompt_type)
    
    responses = []
    # from transformers import AutoTokenizer
    # from jinja2 import Template
    # tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side='left')
    # template = Template(tokenizer.chat_template)
    # prompts = [template.render(messages=[{"role": "user", "content": prompt}],bos_token=tokenizer.bos_token,add_generation_prompt=True) for prompt in prompts]

    if model_name == "gpt4" or model_name == "gpt3.5" or model_name == "ChatGPT": 
        import concurrent.futures
        responses = [None] * len(prompts)
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(process_prompt_gpt, prompt, llm_instance, i)
                for i, prompt in enumerate(prompts)
            ]

        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing prompts"):
            i, result = future.result()
            responses[i] = result  # 根据索引i将结果放入responses中
    else:   
        batch_size = 16
        num_batches = (len(prompts) + batch_size - 1) // batch_size  # Calculate total number of batches
        progress_bar = tqdm.tqdm(total=num_batches, desc="Processing prompts in batches")
        
        for i in range(0, len(prompts), batch_size):
             # Get the current batch of prompts
            batch_prompts = prompts[i:i + batch_size]
        
            # Obtain predictions for the current batch
            batch_responses = llm_instance.get_prediction(batch_prompts, length=40, task="RE")
        
            
            for response in batch_responses:
                # Assuming `response` is a list or similar structure where the first item is the text
                processed_response = white_space_fix(response)
                responses.append(processed_response)
        
            # Optional: Update progress bar manually if using tqdm
            # This updates the progress bar based on the number of batches processed
            # tqdm.tqdm.update(i // batch_size + 1)
            progress_bar.update(1)
        progress_bar.close() 
    labels_new = [item.lower().strip() for item in labels]
    responses_new = [item.lower().strip() for item in responses]
    
    precision = precision_score(labels_new, responses_new, average='macro')
    # 计算召回率
    recall = recall_score(labels_new, responses_new, average='macro')
    # 计算 F1 分数
    f1 = f1_score(labels_new, responses_new, average='macro')
  
    print("Strict Precision: " + str(precision))
    print("Strict  Recall: " + str(recall))
    print("Strict  F1 Score: " + str(f1))

    acc = relaxed_acc(labels_new, responses_new)

    print("Relaxed  ACC: " + str(acc))
    
    # write_json(output_prompts_path, prompts)
    write_json(output_responses_path, responses)
def post_process(model_output):
    # 使用更宽松的正则表达式匹配role和argument
    pattern = r'(?P<role>"type":\s*"[^"]*")[^}]*?(?P<argument>"argument":\s*"[^"]*")'
    matches = re.finditer(pattern, model_output, re.IGNORECASE)

    result = []
    for match in matches:
        role = re.search(r'"type":\s*"([^"]*)"', match.group('role')).group(1).strip()
        argument = re.search(r'"argument":\s*"([^"]*)"', match.group('argument')).group(1).strip()
        result.append({
                'role': role,
                'argument': argument
            })

    return json.dumps(result, ensure_ascii=False, indent=2)

from sklearn.metrics import f1_score, precision_score, recall_score

def relaxed_acc(labels, predictions):
    relaxed_predictions = [int(label in prediction) for label, prediction in zip(labels, predictions)]
    # 计算准确率：正确预测的数量除以总数
    accuracy = sum(relaxed_predictions) / len(labels)
    return accuracy
def s_f1_score(labels, predictions):
    relaxed_predictions = [int(label == prediction) for label, prediction in zip(labels, predictions)]
    relaxed_labels = [1] * len(labels)  # 所有的label都被认为是正类
    return f1_score(relaxed_labels, relaxed_predictions, average='macro'), precision_score(relaxed_labels, relaxed_predictions, average='macro'), recall_score(relaxed_labels, relaxed_predictions, average='macro')
def white_space_fix(text: str):
        return " ".join(text.split())
def normalize_text(input_string) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text: str):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str):
        return " ".join(text.split())

    def remove_punc(text: str):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(input_string))))

# def normalize_text(text):
#     # 移除常见的前置词和特殊字符，转换为小写
#     prepositions = {"the", "a", "an"}
#     words = [word for word in re.split(r'\W+', text.lower()) if word and word not in prepositions]
#     return ' '.join(words)

# def calculate_partial_match_score(pred, label):
#     # Calculate intersection and union of words for Jaccard similarity
#     pred_set = set(pred.split(' '))
#     label_set = set(label.split(' '))
#     intersection = pred_set.intersection(label_set)
#     union = pred_set.union(label_set)
#     if not union:
#         return 0
#     return len(intersection) / len(union)  # Jaccard similarity

# def compute_exact_match(label_str, pred_str):
#     return normalize_text(pred_str) == normalize_text(label_str)

def calculate_precision_recall_f1(predictions, labels):
    true_positive = 0
    false_positive = 0
    false_negative = 0
    EM = []

    for i in range(len(predictions)):
        prediction = json.loads(predictions[i])
        label = labels[i]

        # Normalize the predictions and labels to be lowercase
        pred_counter = Counter([(item['role'].lower(), normalize_text(item['argument'].lower())) for item in prediction])
        label_counter = Counter([(item['role'].lower(), normalize_text(item['text'].lower())) for item in label])
        
        preds = {item['role'].lower(): normalize_text(item['argument'].lower()) for item in prediction}
        label_s = {item['role'].lower(): normalize_text(item['text'].lower()) for item in label}

        for role, text in label_s.items():
            if role in preds and text in preds[role]:
                EM.append(1)
                true_positive += 1
            else:
                EM.append(0)
                false_negative += 1

        # Calculate false positives
        for role, text in preds.items():
            if role not in label_s or text not in label_s.get(role, ''):
                false_positive += 1

    # Calculate precision and recall
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0
    mention_exact_match = np.mean(EM) * 100

    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1_score, mention_exact_match


def calculate_precision_recall_f1_strict(predictions, labels):
    true_positive = 0
    false_positive = 0
    false_negative = 0
    EM = []
    for i in range(len(predictions)):
        prediction = json.loads(predictions[i])
        label = labels[i]
        # Normalize the predictions and labels to be lowercase
        pred_counter = Counter([(item['role'].lower(), normalize_text(item['argument'].lower())) for item in prediction])
        label_counter = Counter([(item['role'].lower(), normalize_text(item['text'].lower())) for item in label])
        preds = list(pred_counter.keys())
        label_s = list(label_counter.keys())
        for item in label_s:
            if item in preds:
                EM.append(1)
            else:
                EM.append(0)

        # Calculate true positives
        true_positive += sum((pred_counter & label_counter).values())

        # Calculate false positives and false negatives
        false_positive += sum((pred_counter - label_counter).values())
        false_negative += sum((label_counter - pred_counter).values())

    # Calculate precision and recall
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0
    mention_exact_match = np.mean(EM) * 100

    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1_score, mention_exact_match



def read_json(path):
    """ Read a json file from the given path."""
    with open(path, 'r') as f:
        data = json.load(f)
    return data



