import os
import sys
import json

import re
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


PACKAGE_PARENT = '.'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
PREFIX_PATH = "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[:-1]) + "/"
from prompt_templates import *

def read_json(path):
    """Read json file"""

    with open(path, 'r') as f:
        data = json.load(f)

    return data

def rams_format(test_data,  similar_sentences, train_dataset=None, type="rag", topk=1):
    """Regenerate prompt for tacred and its variants like tacrev, re-tacred

    Args:
        test_data (list): list of test data
        relations (list): list of relations (target labels)
        similar_sentences (list): list of similar sentence with corresponding test data
        type (str, optional): prompt type. Defaults to "rag".

    Returns:
        list: list of regenerated prompts
    """
    
    prompts = []
    
    for index, instance in enumerate(test_data):

        if type == "simple":
            prompt = get_zero_shot_template(instance[1])
        else:
            context = similar_sentences[index]
            prompt = get_zero_shot_template_rag(instance[1], context['similar_sentence'], topk)
        prompts.append(prompt)

    print("Number Prompts:{0}".format(len(prompts)))

    return prompts

def wiki_format(test_data,  similar_sentences, train_dataset=None, type="rag", topk=1):
    """Regenerate prompt for tacred and its variants like tacrev, re-tacred

    Args:
        test_data (list): list of test data
        relations (list): list of relations (target labels)
        similar_sentences (list): list of similar sentence with corresponding test data
        type (str, optional): prompt type. Defaults to "rag".

    Returns:
        list: list of regenerated prompts
    """
    
    prompts = []
    
    for index, instance in enumerate(test_data):

        if type == "simple":
            prompt = get_zero_shot_template(instance[1])
        else:
            context = similar_sentences[index]
            prompt = get_zero_shot_template_rag(instance[1], context['similar_sentence'], topk)
        prompts.append(prompt)

    print("Number Prompts:{0}".format(len(prompts)))
    return prompts

def ace_format(test_data,  similar_sentences, train_dataset=None, type="rag", topk=1):
    """Regenerate prompt for tacred and its variants like tacrev, re-tacred

    Args:
        test_data (list): list of test data
        relations (list): list of relations (target labels)
        similar_sentences (list): list of similar sentence with corresponding test data
        type (str, optional): prompt type. Defaults to "rag".

    Returns:
        list: list of regenerated prompts
    """
    
    prompts = []
    
    for index, instance in enumerate(test_data):

        if type == "simple":
            prompt = get_zero_shot_template(instance[1])
        else:
            context = similar_sentences[index]
            prompt = get_zero_shot_template_rag(instance[1], context['similar_sentence'], topk)
        prompts.append(prompt)

    print("Number Prompts:{0}".format(len(prompts)))
    return prompts

def generate_prompts_RE(sentences, relations, similar_sentences, dataset="tacred", prompt_type="rag", topk=1):
    """Regenerate the user query along with similar sentence.

    Args:
        sentences (list): list of sentences or dataset
        relations (list): list of relations
        similar_sentences (list): list of similar sentences
        dataset (str, optional): dataset name. Defaults to "tacred".
        prompt_type (str, optional): approach type. Defaults to "rag".

    Returns:
        list of prompts: list of regenerated prompts
    """

    prompts = []

    if dataset == "semeval":
        
        if prompt_type == "simple":
            prompts = semeval_format(sentences, relations, similar_sentences,  prompt_type, topk=topk)
        else:
            prompts = semeval_format(sentences, relations, similar_sentences,  prompt_type, topk=topk)
    else:

        if prompt_type == "simple":
            prompts = tacred_format(sentences, relations, similar_sentences,  prompt_type, topk=topk)
        else:
            prompts = tacred_format(sentences, relations, similar_sentences,  prompt_type, topk=topk)
    
    return prompts
    

def generate_prompts_EE(sentences, similar_sentences, dataset="rams", train_dataset=None, prompt_type="rag", topk=1):
    """Regenerate the user query along with similar sentence.

    Args:
        sentences (list): list of sentences or dataset
        relations (list): list of relations
        similar_sentences (list): list of similar sentences
        dataset (str, optional): dataset name. Defaults to "tacred".
        prompt_type (str, optional): approach type. Defaults to "rag".

    Returns:
        list of prompts: list of regenerated prompts
    """

    prompts = []
    if dataset == "rams":
        prompts = rams_format(sentences, similar_sentences, train_dataset, prompt_type, topk)
        return prompts
    elif dataset == "wiki":
        prompts = wiki_format(sentences, similar_sentences, train_dataset, prompt_type, topk)
        return prompts
    elif dataset == "ace2005":
        prompts = ace_format(sentences, similar_sentences, train_dataset, prompt_type, topk)
        return prompts

def semeval_format(test_data, relations, similar_sentences, prompt_type="simple", topk=1):
    """Regenerate prompt for semeval dataset

    Args:
        test_data (list): list of test sentences along with e1 and e2
        relations (list): target relation label indexes
        similar_sentences (list): list of similar sentence with corresponding test data
        labels (list): the list  of target label names
        prompt_type (str, optional): prompt type. Defaults to "simple".

    Returns:
        list: the list of regenerated prompts
    """
    
    # relation_names = list(set(relations))
    # relations = ", ".join([relation for relation in relation_names])
    prompts = []

    for index, line in enumerate(test_data):

        # label = labels[index]
        sentence = line
        context = similar_sentences[index]

        e1_index = sentence.find("<e1>")
        e2_index = sentence.find("<e2>")

        if e1_index < e2_index:
            head_name = re.findall("<e1>(.*?)</e1>", sentence, re.DOTALL)
            tail_name = re.findall("<e2>(.*?)</e2>", sentence, re.DOTALL)
            head = "e1"
            tail = "e2"
        else:
            # print("e2")
            head_name = re.findall("<e2>(.*?)</e2>", sentence, re.DOTALL)
            tail_name = re.findall("<e1>(.*?)</e1>", sentence, re.DOTALL)
            head = "e2"
            tail = "e1"

        head_name = " ".join(head_name)
        tail_name = " ".join(tail_name)
        
        if prompt_type == "simple":
            prompt = semeval_prompt_template(sentence, relations, head, tail, head_name, tail_name)
        
        else:
            # context = context[index]
            prompt = semeval_prompt_template_rag(sentence, relations, head, tail, head_name, tail_name, context['similar_sentence'], topk)
            
        prompts.append(prompt)

    print("Number of Prompts:{0}".format(len(prompts)))

    return prompts

def tacred_format(test_data, relations, similar_sentences, type="rag",  topk=1):
    """Regenerate prompt for tacred and its variants like tacrev, re-tacred

    Args:
        test_data (list): list of test data
        relations (list): list of relations (target labels)
        similar_sentences (list): list of similar sentence with corresponding test data
        type (str, optional): prompt type. Defaults to "rag".

    Returns:
        list: list of regenerated prompts
    """
    
    prompts = []

    
    for index, line in enumerate(test_data):
        
        sentence = line['sentence']
        head = line['subj']
        tail = line['obj']

        if type == "simple":
            prompt = get_zero_shot_template_tacred(sentence, relations, head, tail)
        else:
            context = similar_sentences[index]
            prompt = get_zero_shot_template_tacred_rag(sentence, relations, head, tail, context['similar_sentence'], topk)
        prompts.append(prompt)

    print("Number Prompts:{0}".format(len(prompts)))

    return prompts