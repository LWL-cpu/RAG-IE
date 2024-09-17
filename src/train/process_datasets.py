import jsonlines
import copy
import csv
from itertools import chain
import json
import random
import os
import sys
import configparser
import re
import json
from collections import defaultdict
from collections import Counter
import string
import numpy as np

def read_test_label(input_file, dataset):
    lines = read_jsonlines(input_file)
    examples_out = []
    if dataset == "rams":
        examples = create_example_rams(lines)
        for example in examples:
            examples_out.append(example.args)
    if dataset == "WikiEvent":
        examples = create_example_wikievent(lines)
        for example in examples:
            examples_out.append(example.args)
    if dataset == "ace_eeqa":
        examples = create_example_ace(lines)
        for example in examples:
            examples_out.append(example.args)
    return examples_out

def read_jsonlines(input_file):
    lines = []
    with jsonlines.open(input_file) as reader:
        for obj in reader:
            lines.append(obj)
    return lines

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
    labels = []

    
    for index, line in enumerate(test_data):
        
        sentence = line['sentence']
        head = line['subj']
        tail = line['obj']
        label = line['label'].split(":")[-1]

        if type == "simple":
            prompt = get_zero_shot_template_tacred(sentence, relations, head, tail)
        else:
            context = similar_sentences[index]
            prompt = get_zero_shot_template_tacred_rag(sentence, relations, head, tail, context['similar_sentence'], topk)
        prompts.append(prompt)
        labels.append(label)

    print("Number Prompts:{0}".format(len(prompts)))

    return prompts, labels

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
    
    relation_names = list(set(relations))
    relations = ", ".join([relation for relation in relation_names])
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

    return prompts







class Event:
    def __init__(self, doc_id, sent_id, sent, event_type, event_trigger, event_args, full_text, first_word_locs=None):
        self.doc_id = doc_id
        self.sent_id = sent_id
        self.sent = sent
        self.type = event_type
        self.trigger = event_trigger
        self.args = event_args

        self.full_text = full_text
        self.first_word_locs = first_word_locs

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "doc id: {}\n".format(self.doc_id)
        s += "sent id: {}\n".format(self.sent_id)
        s += "text: {}\n".format(" ".join(self.sent))
        s += "event_type: {}\n".format(self.type)
        s += "trigger: {}\n".format(self.trigger['text'])
        for arg in self.args:
            s += "arg {}: {} ({}, {})\n".format(arg['role'], arg['text'], arg['start'], arg['end'])
        s += "----------------------------------------------\n"
        return s

def read_roles(role_path):
    template_dict = {}
    role_dict = {}
    with open(role_path, "r", encoding='utf-8') as f:
         csv_reader = csv.reader(f)
         for line in csv_reader:
            event_type_arg, template = line
            template_dict[event_type_arg] = template

            event_type, arg = event_type_arg.split('_')
            if event_type not in role_dict:
                role_dict[event_type] = []
            role_dict[event_type].append(arg)
    return template_dict, role_dict
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
def create_example_wikievent(lines):
    W = 250
    assert(W%2==0)
    all_args_num = 0
    invalid_arg_num = 0

    examples = []
    for line in lines:
        entity_dict = {entity['id']:entity for entity in line['entity_mentions']}
        events = line["event_mentions"]
        if not events:
            continue
        doc_key = line["doc_id"]
        full_text = line['tokens']
        sent_length = len(full_text)

        curr_loc = 0
        first_word_locs = []
        for sent in line["sentences"]:
            first_word_locs.append(curr_loc)
            curr_loc += len(sent[0])

        for event in events:
            event_type = event['event_type']
            cut_text = full_text
            event_trigger = event['trigger']

            offset, min_s, max_e = 0, 0, W+1
            if sent_length > W+1:
                if event_trigger['end'] <= W//2:     # trigger word is located at the front of the sents
                    cut_text = full_text[:(W+1)]
                elif event_trigger['start'] >= sent_length-W/2:   # trigger word is located at the latter of the sents
                    offset = sent_length - (W+1)
                    min_s += offset
                    max_e += offset
                    event_trigger['start'] -= offset
                    event_trigger['end'] -= offset 
                    cut_text = full_text[-(W+1):]
                else:
                    offset = event_trigger['start'] - W//2
                    min_s += offset
                    max_e += offset
                    event_trigger['start'] -= offset
                    event_trigger['end'] -= offset 
                    cut_text = full_text[offset:(offset+W+1)]
            event_trigger['offset'] = offset
                        
            event_args = list()
            for arg_info in event['arguments']:
                all_args_num += 1

                evt_arg = dict()
                arg_entity = entity_dict[arg_info['entity_id']]
                evt_arg['start'] = arg_entity['start']
                evt_arg['end'] = arg_entity['end']
                evt_arg['text'] = arg_info['text']
                evt_arg['role'] = arg_info['role']
                if evt_arg['start']<min_s or evt_arg['end']>max_e:
                    invalid_arg_num += 1
                else:
                    evt_arg['start'] -= offset
                    evt_arg['end'] -= offset 
                    event_args.append(evt_arg)
            examples.append(Event(doc_key, None, cut_text, event_type, event_trigger, event_args, full_text, first_word_locs))

    print("{} examples collected. {} dropped.".format(len(examples), invalid_arg_num))
    return examples
def create_example_ace(lines):
    examples = []
    for doc_idx, line in enumerate(lines):
        if not line['event']:
            continue
        events = line['event']
        offset = line['s_start']
        full_text = copy.deepcopy(line['sentence'])
        text = line['sentence']
        for event_idx, event in enumerate(events):
            event_type = event[0][1]
            event_trigger = dict()
            start = event[0][0] - offset; end = start+1
            event_trigger['start'] = start; event_trigger['end'] = end
            event_trigger['text'] = " ".join(text[start:end])
            event_trigger['offset'] = offset

            event_args = list()
            for arg_info in event[1:]:
                arg = dict()
                start = arg_info[0]-offset; end = arg_info[1]-offset+1
                role = arg_info[2]
                arg['start'] = start; arg['end'] = end
                arg['role'] = role; arg['text'] = " ".join(text[start:end])
                event_args.append(arg)

            examples.append(Event(doc_idx, event_idx, text, event_type, event_trigger, event_args, full_text))
            
    print("{} examples collected.".format(len(examples)))
    return examples

def create_example_rams(lines):
        # maximum doc length is 543 in train (max input ids 803), 394 in dev, 478 in test
        # too long, so we use a window to cut the sentences.
    W = 250
    invalid_arg_num = 0
    assert (W % 2 == 0)
    all_args_num = 0

    examples = []
    for line in lines:
        if len(line["evt_triggers"]) == 0:
            continue
        doc_key = line["doc_key"]
        events = line["evt_triggers"]

        full_text = copy.deepcopy(list(chain(*line['sentences'])))
        cut_text = list(chain(*line['sentences']))
        sent_length = sum([len(sent) for sent in line['sentences']])

        text_tmp = []
        first_word_locs = []
        for sent in line["sentences"]:
            first_word_locs.append(len(text_tmp))
            text_tmp += sent
            # 这里相当于切割句子在W之内。同时保证触发词一定在句子中
        for event_idx, event in enumerate(events):
            event_trigger = dict()
            event_trigger['start'] = event[0]
            event_trigger['end'] = event[1] + 1
            event_trigger['text'] = " ".join(full_text[event_trigger['start']:event_trigger['end']])
            event_type = event[2][0][0]

            offset, min_s, max_e = 0, 0, W + 1
            event_trigger['offset'] = offset
            if sent_length > W + 1:
                if event_trigger['end'] <= W // 2:  # trigger word is located at the front of the sents
                    cut_text = full_text[:(W + 1)]
                else:  # trigger word is located at the latter of the sents
                    offset = sent_length - (W + 1)
                    min_s += offset
                    max_e += offset
                    event_trigger['start'] -= offset
                    event_trigger['end'] -= offset
                    event_trigger['offset'] = offset
                    cut_text = full_text[-(W + 1):]
                # 把所有的参数标签整理好
            event_args = list()
            for arg_info in line["gold_evt_links"]:
                if arg_info[0][0] == event[0] and arg_info[0][1] == event[1]:  # match trigger span
                    all_args_num += 1

                    evt_arg = dict()
                    evt_arg['start'] = arg_info[1][0]
                    evt_arg['end'] = arg_info[1][1] + 1
                    evt_arg['text'] = " ".join(full_text[evt_arg['start']:evt_arg['end']])
                    evt_arg['role'] = arg_info[2].split('arg', maxsplit=1)[-1][2:]
                    if evt_arg['start'] < min_s or evt_arg['end'] > max_e:
                        invalid_arg_num += 1
                    else:
                        evt_arg['start'] -= offset
                        evt_arg['end'] -= offset
                        event_args.append(evt_arg)

            if event_idx > 0:
                examples.append(
                    Event(doc_key + str(event_idx), None, cut_text, event_type, event_trigger, event_args,
                            full_text, first_word_locs))
            else:
                examples.append(Event(doc_key, None, cut_text, event_type, event_trigger, event_args, full_text,
                                        first_word_locs))

    # print("{} examples collected. {} arguments dropped.".format(len(examples), invalid_arg_num))
    return examples


def process_instruction(input_file, dataset, rag, topk=10):
    # Process semeval dataset
    outs = []
    relations = read_json("/223040263/wanlong/LLM_Retreival/RAG4RE/data/semeval/original_data/relations.json")
    relations_ = relations['relation']['names']
    relations = [item.split('(')[0] for item in relations_]
            
    
    similar_sentences = read_json("/223040263/wanlong/LLM_Retreival/RAG4RE/data/semeval/processed_data/sentence_sim_train.json")
    test_data = read_json("/223040263/wanlong/LLM_Retreival/RAG4RE/data/semeval/processed_data/test_sentences.json")
    train_data = read_json("/223040263/wanlong/LLM_Retreival/RAG4RE/data/semeval/processed_data/train_sentences.json")
    train_labels_ = read_json("/223040263/wanlong/LLM_Retreival/RAG4RE/data/semeval/processed_data/train_labels.json")
    train_labels = [relations[item] for item in train_labels_]
    test_labels_ = read_json("/223040263/wanlong/LLM_Retreival/RAG4RE/data/semeval/processed_data/test_labels.json")
    test_labels = [relations[item] for item in test_labels_]
    for item in similar_sentences:
        tmp_sents = item["similar_sentence"]
        tmp_labels = item["label"]
        new_sents_list = []
        for i, sent in enumerate(tmp_sents):
            new_sent = sent + " The Relation between <e1> and <e2> is " + tmp_labels[i].split("(")[0]
            new_sents_list.append(new_sent)
        item["similar_sentence"] = new_sents_list


    total_examples = len(train_data)
    # ratios = [0.2, 0, 0, 0.05, 0.15, 0.1, 0.2, 0.1, 0.2]
    ratios = [0.5, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625]
    counts = [int(total_examples * ratio) for ratio in ratios]

    # 使用切片和累积偏移量分割示例
    semeval_prompts = []
    offset = 0
    test_data_new = []
    for i, item in enumerate(test_data):
        new_sent = item + " The Relation between <e1> and <e2> is " + test_labels[i]
        test_data_new.append(new_sent)
    from copy import deepcopy
    similar_sentences_random = deepcopy(similar_sentences)
    for item in similar_sentences_random:
        item["similar_sentence"] = random.sample(test_data_new, 30)

    indices = list(range(len(train_data)))
    random.shuffle(indices)
    train_data_random = [train_data[idx] for idx in indices]
    train_labels_random = [train_labels[idx] for idx in indices]
    similar_sentences_random_ = [similar_sentences_random[idx] for idx in indices]
    similar_sentences_ = [similar_sentences[idx] for idx in indices]

    train_data = train_data_random
    train_labels = train_labels_random
    similar_sentences_random = similar_sentences_random_ 
    similar_sentences = similar_sentences_
            
    no_rag_examples = train_data[offset : offset + counts[0]]
    semeval_prompts.extend(semeval_format(no_rag_examples, relations, similar_sentences[offset:], "simple"))
    offset += counts[0]

    similarity_examples_1 = train_data[offset : offset + counts[1]]
    semeval_prompts.extend(semeval_format(similarity_examples_1, relations, similar_sentences[offset:], "rag", 1))
    offset += counts[1]

    random_examples_1 = train_data[offset : offset + counts[2]]
    semeval_prompts.extend(semeval_format(random_examples_1, relations, similar_sentences_random[offset:], "rag", 1))
    offset += counts[2]

    similarity_examples_5 = train_data[offset : offset + counts[3]]
    semeval_prompts.extend(semeval_format(similarity_examples_5, relations, similar_sentences[offset:], "rag", 5))
    offset += counts[3]

    random_examples_5 = train_data[offset : offset + counts[4]]
    semeval_prompts.extend(semeval_format(random_examples_5, relations, similar_sentences_random[offset:], "rag", 5))
    offset += counts[4]

    similarity_examples_10 = train_data[offset : offset + counts[5]]
    semeval_prompts.extend(semeval_format(similarity_examples_10, relations, similar_sentences[offset:], "rag", 10))
    offset += counts[5]

    random_examples_10 = train_data[offset : offset + counts[6]]
    semeval_prompts.extend(semeval_format(random_examples_10, relations, similar_sentences_random[offset:], "rag", 10))
    offset += counts[6]

    similarity_examples_15 = train_data[offset : offset + counts[7]]
    semeval_prompts.extend(semeval_format(similarity_examples_15, relations, similar_sentences[offset:], "rag", 15))
    offset += counts[7]
      
    random_examples_15 = train_data[offset : offset + counts[8]]
    semeval_prompts.extend(semeval_format(random_examples_15, relations, similar_sentences_random[offset:], "rag", 15))
            
    for i, item in enumerate(semeval_prompts):
        outs.append({"input": item, "output": train_labels[i]})





    # Process Tacred dataset
    similar_sentences = read_json("/223040263/wanlong/LLM_Retreival/RAG4RE/data/tacred/processed_data/train_tacred_similarities.json")
    test_data = read_json("/223040263/wanlong/LLM_Retreival/RAG4RE/data/tacred/processed_data/test_data.json")
    train_data = read_json("/223040263/wanlong/LLM_Retreival/RAG4RE/data/tacred/processed_data/train_data.json")
    relations_ = extract_unique_relations(train_data)
    relations = [item.split(":")[-1] for item in relations_]



    for item in similar_sentences:
        tmp_sents = item["similar_sentence"]
        tmp_labels = item["label"]
        subjs = item["subj"]
        objs = item["obj"]  
        new_sents_list = []
        for i, sent in enumerate(tmp_sents):
            new_sent = sent["sentence"] + " The relation between head entity {} and tail entity {} is ".format(subjs[i], objs[i]) + tmp_labels[i].split(":")[-1]
            new_sents_list.append(new_sent)
        item["similar_sentence"] = new_sents_list


    total_examples = len(train_data)
    # ratios = [0.2, 0, 0, 0.05, 0.15, 0.05, 0.25, 0.1, 0.2]
    ratios = [0.5, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625]
    counts = [int(total_examples * ratio) for ratio in ratios]

    # 使用切片和累积偏移量分割示例
    tacred_prompts = []
    tacred_labels = []
    offset = 0
    test_data_new = []
    for i, item in enumerate(test_data):
        subjs = item["subj"]
        objs = item["obj"]  
        label = item['label'].split(":")[-1]
        new_sent = item["sentence"] + " The relation between head entity {} and tail entity {} is ".format(subjs, objs) + label
        test_data_new.append(new_sent)
    from copy import deepcopy
    similar_sentences_random = deepcopy(similar_sentences)
    for item in similar_sentences_random:
        item["similar_sentence"] = random.sample(test_data_new, 30)

    indices = list(range(len(train_data)))
    random.shuffle(indices)
    train_data_random = [train_data[idx] for idx in indices]
    similar_sentences_random_ = [similar_sentences_random[idx] for idx in indices]
    similar_sentences_ = [similar_sentences[idx] for idx in indices]

    train_data = train_data_random
    similar_sentences_random = similar_sentences_random_ 
    similar_sentences = similar_sentences_


            
    no_rag_examples = train_data[offset : offset + counts[0]]
    prompts, labels = tacred_format(no_rag_examples, relations, similar_sentences[offset:], "simple")
    tacred_prompts.extend(prompts)
    tacred_labels.extend(labels)
    offset += counts[0]

    similarity_examples_1 = train_data[offset : offset + counts[1]]
    prompts, labels = tacred_format(similarity_examples_1, relations, similar_sentences[offset:], "rag", 1)
    tacred_prompts.extend(prompts)
    tacred_labels.extend(labels)
    offset += counts[1]

    random_examples_1 = train_data[offset : offset + counts[2]]
    prompts, labels = tacred_format(random_examples_1, relations, similar_sentences_random[offset:], "rag", 1)
    tacred_prompts.extend(prompts)
    tacred_labels.extend(labels)
    offset += counts[2]

    similarity_examples_5 = train_data[offset : offset + counts[3]]
    prompts, labels = tacred_format(similarity_examples_5, relations, similar_sentences[offset:], "rag", 5)
    tacred_prompts.extend(prompts)
    tacred_labels.extend(labels)
    offset += counts[3]

    random_examples_5 = train_data[offset : offset + counts[4]]
    prompts, labels = tacred_format(random_examples_5, relations, similar_sentences_random[offset:], "rag", 5)
    tacred_prompts.extend(prompts)
    tacred_labels.extend(labels)
    offset += counts[4]

    similarity_examples_10 = train_data[offset : offset + counts[5]]
    prompts, labels = tacred_format(similarity_examples_10, relations, similar_sentences[offset:], "rag", 10)
    tacred_prompts.extend(prompts)
    tacred_labels.extend(labels)
    offset += counts[5]

    random_examples_10 = train_data[offset : offset + counts[6]]
    prompts, labels = tacred_format(random_examples_10, relations, similar_sentences_random[offset:], "rag", 10)
    tacred_prompts.extend(prompts)
    tacred_labels.extend(labels)
    offset += counts[6]

    similarity_examples_15 = train_data[offset : offset + counts[7]]
    prompts, labels = tacred_format(similarity_examples_15, relations, similar_sentences[offset:], "rag", 15)
    tacred_prompts.extend(prompts)
    tacred_labels.extend(labels)
    offset += counts[7]
      
    random_examples_15 = train_data[offset : offset + counts[8]]
    prompts, labels = tacred_format(random_examples_15, relations, similar_sentences_random[offset:], "rag", 15)
    tacred_prompts.extend(prompts)
    tacred_labels.extend(labels)
            
    for p, l in zip(tacred_prompts, tacred_labels):
        outs.append({"input": p, "output": l})
    return outs

def process_instruction_only_rams(input_file, dataset, rag):
    lines = read_jsonlines(input_file)
    examples = create_example_rams(lines)
    similar_sentences = read_json('/223040263/wanlong/LLM_Retreival/RAG4RE/data/rams/train_rams_similarities.json')
    roles_dict = read_roles('/223040263/wanlong/LLM_Retreival/RAG4RE/data/dset_meta/description_rams.csv')
    indices = list(range(len(examples)))
    random.shuffle(indices)
    examples_random = [examples[idx] for idx in indices]
    similar_sentences_ = [similar_sentences[idx] for idx in indices]
    examples = examples_random
    similar_sentences = similar_sentences_

    # 计算每个比例对应的数量
    total_examples = len(examples)
    ratios = [1, 0, 0, 0, 0, 0, 0, 0, 0]
    # ratios = [0.5, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625]
    counts = [int(total_examples * ratio) for ratio in ratios]

    # 使用切片和累积偏移量分割示例
    offset = 0
    no_rag_examples = examples[offset : offset + counts[0]]
    offset += counts[0]

    similarity_examples_1 = examples[offset : offset + counts[1]]
    offset += counts[1]

    random_examples_1 = examples[offset : offset + counts[2]]
    offset += counts[2]

    similarity_examples_5 = examples[offset : offset + counts[3]]
    offset += counts[3]

    random_examples_5 = examples[offset : offset + counts[4]]
    offset += counts[4]

    similarity_examples_10 = examples[offset : offset + counts[5]]
    offset += counts[5]

    random_examples_10 = examples[offset : offset + counts[6]]
    offset += counts[6]

    similarity_examples_15 = examples[offset : offset + counts[7]]
    offset += counts[7]
      
    random_examples_15 = examples[offset :]

            
    outs = []

    offset = len(outs)
    for example in no_rag_examples:
        examples_out = {}
        event_type = example.type
        roles = roles_dict[1][event_type]
        trigger = example.trigger['text']
        instruction = """Task description: Given a document and an event, you need to identify all arguments of this event, and classify the role of this argument. Limit responses to arguments only.  Please directly answer in JSON format of [{"type": <role>, "argument": <argument>}, {"type": <role>, "argument": <argument>}, ...].\n""" +   \
                    "Question:" + '\n'
        input = ("Document: " + " ".join(example.sent) + "\n" + "Event: " + event_type + "\n" + "Trigger: " + trigger + "\n" + "Possible roles: " + ", ".join(roles) + "\n") + "Arguments:"

        args = example.args
        arguments_list = []
        for role in roles:
            for item in args:
                if role == item['role']:
                    arguments_list.append({"type": role, "argument": item['text']})
        arguments_json = json.dumps(arguments_list, indent=2)
        examples_out['input'] = instruction + input
        examples_out['output'] = arguments_json
        outs.append(examples_out)

            
    offset = len(outs)
    for i, example in enumerate(similarity_examples_1):
        examples_out = {}
        event_type = example.type
        retrieval_ctxs = similar_sentences[i+offset]["similar_sentence"]
        retrieval_ctxs = "\n".join(retrieval_ctxs[:1])
        roles = roles_dict[1][event_type]
        trigger = example.trigger['text']
        instruction = """Task description: Given a document and an event, you need to identify all arguments of this event, and classify the role of this argument. Limit responses to arguments only.  Please directly answer in JSON format of [{"type": <role>, "argument": <argument>}, {"type": <role>, "argument": <argument>}, ...].\n""" + "Examples:" + retrieval_ctxs + "\nExamples end here.\n" + \
                    "Question:" + '\n'
        input = ("Document: " + " ".join(example.sent) + "\n" + "Event: " + event_type + "\n" + "Trigger: " + trigger + "\n" + "Possible roles: " + ", ".join(roles) + "\n") + "Arguments:"
        args = example.args
        arguments_list = []
        for role in roles:
            for item in args:
                if role == item['role']:
                    arguments_list.append({"type": role, "argument": item['text']})
        arguments_json = json.dumps(arguments_list, indent=2)
        examples_out['input'] = instruction + input
        examples_out['output'] = arguments_json
        outs.append(examples_out)

    offset = len(outs)
    for i, example in enumerate(random_examples_1):
        examples_out = {}
        event_type = example.type
        retrieval_ctxs = similar_sentences[i+offset]["similar_sentence"]
        retrieval_ctxs = "\n".join(random.sample(retrieval_ctxs, 1))
        roles = roles_dict[1][event_type]
        trigger = example.trigger['text']
        instruction = """Task description: Given a document and an event, you need to identify all arguments of this event, and classify the role of this argument. Limit responses to arguments only.  Please directly answer in JSON format of [{"type": <role>, "argument": <argument>}, {"type": <role>, "argument": <argument>}, ...].\n""" + "Examples:" + retrieval_ctxs + "\nExamples end here.\n" + \
                    "Question:" + '\n'
        input = ("Document: " + " ".join(example.sent) + "\n" + "Event: " + event_type + "\n" + "Trigger: " + trigger + "\n" + "Possible roles: " + ", ".join(roles) + "\n") + "Arguments:"
        args = example.args
        arguments_list = []
        for role in roles:
            for item in args:
                if role == item['role']:
                    arguments_list.append({"type": role, "argument": item['text']})
        arguments_json = json.dumps(arguments_list, indent=2)
        examples_out['input'] = instruction + input
        examples_out['output'] = arguments_json
        outs.append(examples_out)

            
    offset = len(outs)
    for i, example in enumerate(similarity_examples_5):
        examples_out = {}
        event_type = example.type
        retrieval_ctxs = similar_sentences[i+offset]["similar_sentence"]
        retrieval_ctxs = "\n".join(retrieval_ctxs[:5])
        roles = roles_dict[1][event_type]
        trigger = example.trigger['text']
        instruction = """Task description: Given a document and an event, you need to identify all arguments of this event, and classify the role of this argument. Limit responses to arguments only.  Please directly answer in JSON format of [{"type": <role>, "argument": <argument>}, {"type": <role>, "argument": <argument>}, ...].\n""" + "Examples:" + retrieval_ctxs + "\nExamples end here.\n" + \
                    "Question:" + '\n'
        input = ("Document: " + " ".join(example.sent) + "\n" + "Event: " + event_type + "\n" + "Trigger: " + trigger + "\n" + "Possible roles: " + ", ".join(roles) + "\n") + "Arguments:"
        args = example.args
        arguments_list = []
        for role in roles:
            for item in args:
                if role == item['role']:
                    arguments_list.append({"type": role, "argument": item['text']})
        arguments_json = json.dumps(arguments_list, indent=2)
        examples_out['input'] = instruction + input
        examples_out['output'] = arguments_json
        outs.append(examples_out)

    offset = len(outs)
    for i, example in enumerate(random_examples_5):
        examples_out = {}
        event_type = example.type
        retrieval_ctxs = similar_sentences[i+offset]["similar_sentence"]
        retrieval_ctxs = "\n".join(random.sample(retrieval_ctxs, 5))
        roles = roles_dict[1][event_type]
        trigger = example.trigger['text']
        instruction = """Task description: Given a document and an event, you need to identify all arguments of this event, and classify the role of this argument. Limit responses to arguments only.  Please directly answer in JSON format of [{"type": <role>, "argument": <argument>}, {"type": <role>, "argument": <argument>}, ...].\n""" + "Examples:" + retrieval_ctxs + "\nExamples end here.\n" + \
                    "Question:" + '\n'
        input = ("Document: " + " ".join(example.sent) + "\n" + "Event: " + event_type + "\n" + "Trigger: " + trigger + "\n" + "Possible roles: " + ", ".join(roles) + "\n") + "Arguments:"
        args = example.args
        arguments_list = []
        for role in roles:
            for item in args:
                if role == item['role']:
                    arguments_list.append({"type": role, "argument": item['text']})
        arguments_json = json.dumps(arguments_list, indent=2)
        examples_out['input'] = instruction + input
        examples_out['output'] = arguments_json
        outs.append(examples_out)

    offset = len(outs)
    for i, example in enumerate(similarity_examples_10):
        examples_out = {}
        event_type = example.type
        retrieval_ctxs = similar_sentences[i+offset]["similar_sentence"]
        retrieval_ctxs = "\n".join(retrieval_ctxs[:10])
        roles = roles_dict[1][event_type]
        trigger = example.trigger['text']
        instruction = """Task description: Given a document and an event, you need to identify all arguments of this event, and classify the role of this argument. Limit responses to arguments only.  Please directly answer in JSON format of [{"type": <role>, "argument": <argument>}, {"type": <role>, "argument": <argument>}, ...].\n""" + "Examples:" + retrieval_ctxs + "\nExamples end here.\n" + \
                    "Question:" + '\n'
        input = ("Document: " + " ".join(example.sent) + "\n" + "Event: " + event_type + "\n" + "Trigger: " + trigger + "\n" + "Possible roles: " + ", ".join(roles) + "\n") + "Arguments:"
        args = example.args
        arguments_list = []
        for role in roles:
            for item in args:
                if role == item['role']:
                    arguments_list.append({"type": role, "argument": item['text']})
        arguments_json = json.dumps(arguments_list, indent=2)
        examples_out['input'] = instruction + input
        examples_out['output'] = arguments_json
        outs.append(examples_out)

    offset = len(outs)
    for i, example in enumerate(random_examples_10):
        examples_out = {}
        event_type = example.type
        retrieval_ctxs = similar_sentences[i+offset]["similar_sentence"]
        retrieval_ctxs = "\n".join(random.sample(retrieval_ctxs, 10))
        roles = roles_dict[1][event_type]
        trigger = example.trigger['text']
        instruction = """Task description: Given a document and an event, you need to identify all arguments of this event, and classify the role of this argument. Limit responses to arguments only.  Please directly answer in JSON format of [{"type": <role>, "argument": <argument>}, {"type": <role>, "argument": <argument>}, ...].\n""" + "Examples:" + retrieval_ctxs + "\nExamples end here.\n" + \
                    "Question:" + '\n'
        input = ("Document: " + " ".join(example.sent) + "\n" + "Event: " + event_type + "\n" + "Trigger: " + trigger + "\n" + "Possible roles: " + ", ".join(roles) + "\n") + "Arguments:"
        args = example.args
        arguments_list = []
        for role in roles:
            for item in args:
                if role == item['role']:
                    arguments_list.append({"type": role, "argument": item['text']})
        arguments_json = json.dumps(arguments_list, indent=2)
        examples_out['input'] = instruction + input
        examples_out['output'] = arguments_json
        outs.append(examples_out)

    offset = len(outs)
    for i, example in enumerate(similarity_examples_15):
        examples_out = {}
        event_type = example.type
        retrieval_ctxs = similar_sentences[i+offset]["similar_sentence"]
        retrieval_ctxs = "\n".join(retrieval_ctxs[:15])
        roles = roles_dict[1][event_type]
        trigger = example.trigger['text']
        instruction = """Task description: Given a document and an event, you need to identify all arguments of this event, and classify the role of this argument. Limit responses to arguments only.  Please directly answer in JSON format of [{"type": <role>, "argument": <argument>}, {"type": <role>, "argument": <argument>}, ...].\n""" + "Examples:" + retrieval_ctxs + "\nExamples end here.\n" + \
                    "Question:" + '\n'
        input = ("Document: " + " ".join(example.sent) + "\n" + "Event: " + event_type + "\n" + "Trigger: " + trigger + "\n" + "Possible roles: " + ", ".join(roles) + "\n") + "Arguments:"
        args = example.args
        arguments_list = []
        for role in roles:
            for item in args:
                if role == item['role']:
                    arguments_list.append({"type": role, "argument": item['text']})
        arguments_json = json.dumps(arguments_list, indent=2)
        examples_out['input'] = instruction + input
        examples_out['output'] = arguments_json
        outs.append(examples_out)

    offset = len(outs)
    for i, example in enumerate(random_examples_15):
        examples_out = {}
        event_type = example.type
        retrieval_ctxs = similar_sentences[i+offset]["similar_sentence"]
        retrieval_ctxs = "\n".join(random.sample(retrieval_ctxs, 15))
        roles = roles_dict[1][event_type]
        trigger = example.trigger['text']
        instruction = """Task description: Given a document and an event, you need to identify all arguments of this event, and classify the role of this argument. Limit responses to arguments only.  Please directly answer in JSON format of [{"type": <role>, "argument": <argument>}, {"type": <role>, "argument": <argument>}, ...].\n""" + "Examples:" + retrieval_ctxs + "\nExamples end here.\n" + \
                    "Question:" + '\n'
        input = ("Document: " + " ".join(example.sent) + "\n" + "Event: " + event_type + "\n" + "Trigger: " + trigger + "\n" + "Possible roles: " + ", ".join(roles) + "\n") + "Arguments:"
        args = example.args
        arguments_list = []
        for role in roles:
            for item in args:
                if role == item['role']:
                    arguments_list.append({"type": role, "argument": item['text']})
        arguments_json = json.dumps(arguments_list, indent=2)
        examples_out['input'] = instruction + input
        examples_out['output'] = arguments_json
        outs.append(examples_out)
    return outs

def read_json(path):
    """ Read a json file from the given path."""
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def read_test(input_file, dataset):
    lines = read_jsonlines(input_file)
    out = []
    if dataset == "rams":
        examples = create_example_rams(lines)
        roles_dict = read_roles('/223040263/wanlong/LLM_Retreival/RAG4RE/data/dset_meta/description_rams.csv')
    elif dataset == "WikiEvent":
        examples = create_example_wikievent(lines)
        roles_dict = read_roles('/223040263/wanlong/LLM_Retreival/RAG4RE/data/dset_meta/description_wikievent.csv')
    elif dataset == "ace_eeqa":
        examples = create_example_ace(lines)
        roles_dict = read_roles('/223040263/wanlong/LLM_Retreival/RAG4RE/data/dset_meta/description_ace.csv')
    for example in examples:
        examples_out = {}
        event_type = example.type
        roles = roles_dict[1][event_type]
        trigger = example.trigger['text']
        instruction = """Task description: Given a document and an event, you need to identify all arguments of this event, and classify the role of this argument. Limit responses to arguments only.  Please directly answer in JSON format of [{"type": <role>, "argument": <argument>}, {"type": <role>, "argument": <argument>}, ...].\n""" +  "Examples:" + '\n' + \
                         'Document: Transportation officials are urging carpool and teleworking as options to combat an expected flood of drivers on the road . ( Paul Duggan ) -- A Baltimore prosecutor accused a police detective of “ sabotaging ” investigations related to the death of Freddie Gray , accusing him of fabricating notes to suggest that the state ’s medical examiner believed the manner of death was an accident rather than a homicide . The heated exchange came in the chaotic sixth day of the trial of Baltimore Officer Caesar Goodson Jr. , who drove the police van in which Gray suffered a fatal spine injury in 2015 . ( Derek Hawkins and Lynh Bui )\nEvent: life.die.deathcausedbyviolentevents\nTrigger: homicide\nPossible roles: instrument, killer, victim, place\nArguments: [{"type": "killer", "argument": "Officer Caesar Goodson Jr."}, {"type": "victim", "argument": "Freddie Gray"}, {"type": "place", "argument": "Baltimore"}]\nExamples end here.\n' + \
                         "Question:" + '\n'
        input = ("Document: " + " ".join(example.full_text) + "\n" + "Event: " + event_type + "\n" + "Trigger: " + trigger + "\n" + "Possible roles: " + ", ".join(roles) + "\n") + "Arguments:"

        args = example.args
        arguments_list = []
        for role in roles:
            for item in args:
                if role == item['role']:
                    arguments_list.append({"type": role, "argument": item['text']})
        arguments_json = json.dumps(arguments_list, indent=2)
        examples_out['input'] = instruction + input
        examples_out['output'] = arguments_json
        out.append(examples_out)
    return out

import numpy as np




def process_instruction_only_rams_wo_label(input_file, dataset, rag):
    lines = read_jsonlines(input_file)
    examples = create_example_rams(lines)
    similar_sentences = read_json('/223040263/wanlong/LLM_Retreival/RAG4RE/data/rams/train_rams_similarities.json')
    roles_dict = read_roles('/223040263/wanlong/LLM_Retreival/RAG4RE/data/dset_meta/description_rams.csv')
    indices = list(range(len(examples)))
    random.shuffle(indices)
    examples_random = [examples[idx] for idx in indices]
    similar_sentences_ = [similar_sentences[idx] for idx in indices]
    examples = examples_random
    similar_sentences = similar_sentences_

    # 计算每个比例对应的数量
    total_examples = len(examples)
    ratios = [0, 0, 0, 0, 0, 1, 0, 0, 0]
    # ratios = [0.5, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625]
    counts = [int(total_examples * ratio) for ratio in ratios]

    # 使用切片和累积偏移量分割示例
    offset = 0
    no_rag_examples = examples[offset : offset + counts[0]]
    offset += counts[0]

    similarity_examples_1 = examples[offset : offset + counts[1]]
    offset += counts[1]

    random_examples_1 = examples[offset : offset + counts[2]]
    offset += counts[2]

    similarity_examples_5 = examples[offset : offset + counts[3]]
    offset += counts[3]

    random_examples_5 = examples[offset : offset + counts[4]]
    offset += counts[4]

    similarity_examples_10 = examples[offset : offset + counts[5]]
    offset += counts[5]

    random_examples_10 = examples[offset : offset + counts[6]]
    offset += counts[6]

    similarity_examples_15 = examples[offset : offset + counts[7]]
    offset += counts[7]
      
    random_examples_15 = examples[offset :]

            
    outs = []

    offset = len(outs)
    for example in no_rag_examples:
        examples_out = {}
        event_type = example.type
        roles = roles_dict[1][event_type]
        trigger = example.trigger['text']
        instruction = """Task description: Given a document and an event, you need to identify all arguments of this event, and classify the role of this argument. Limit responses to arguments only.  Please directly answer in JSON format of [{"type": <role>, "argument": <argument>}, {"type": <role>, "argument": <argument>}, ...].\n""" +   \
                    "Question:" + '\n'
        input = ("Document: " + " ".join(example.sent) + "\n" + "Event: " + event_type + "\n" + "Trigger: " + trigger + "\n" + "Possible roles: " + ", ".join(roles) + "\n") + "Arguments:"

        args = example.args
        arguments_list = []
        for role in roles:
            for item in args:
                if role == item['role']:
                    arguments_list.append({"type": role, "argument": item['text']})
        arguments_json = json.dumps(arguments_list, indent=2)
        examples_out['input'] = instruction + input
        examples_out['output'] = arguments_json
        outs.append(examples_out)

            
    offset = len(outs)
    for i, example in enumerate(similarity_examples_1):
        examples_out = {}
        event_type = example.type
        retrieval_ctxs = similar_sentences[i+offset]["similar_sentence"]
        retrieval_ctxs = "\n".join(retrieval_ctxs[:1])
        roles = roles_dict[1][event_type]
        trigger = example.trigger['text']
        instruction = """Task description: Given a document and an event, you need to identify all arguments of this event, and classify the role of this argument. Limit responses to arguments only.  Please directly answer in JSON format of [{"type": <role>, "argument": <argument>}, {"type": <role>, "argument": <argument>}, ...].\n""" + "Examples:" + retrieval_ctxs + "\nExamples end here.\n" + \
                    "Question:" + '\n'
        input = ("Document: " + " ".join(example.sent) + "\n" + "Event: " + event_type + "\n" + "Trigger: " + trigger + "\n" + "Possible roles: " + ", ".join(roles) + "\n") + "Arguments:"
        args = example.args
        arguments_list = []
        for role in roles:
            for item in args:
                if role == item['role']:
                    arguments_list.append({"type": role, "argument": item['text']})
        arguments_json = json.dumps(arguments_list, indent=2)
        examples_out['input'] = instruction + input
        examples_out['output'] = arguments_json
        outs.append(examples_out)

    offset = len(outs)
    for i, example in enumerate(random_examples_1):
        examples_out = {}
        event_type = example.type
        retrieval_ctxs = similar_sentences[i+offset]["similar_sentence"]
        retrieval_ctxs = "\n".join(random.sample(retrieval_ctxs, 1))
        roles = roles_dict[1][event_type]
        trigger = example.trigger['text']
        instruction = """Task description: Given a document and an event, you need to identify all arguments of this event, and classify the role of this argument. Limit responses to arguments only.  Please directly answer in JSON format of [{"type": <role>, "argument": <argument>}, {"type": <role>, "argument": <argument>}, ...].\n""" + "Examples:" + retrieval_ctxs + "\nExamples end here.\n" + \
                    "Question:" + '\n'
        input = ("Document: " + " ".join(example.sent) + "\n" + "Event: " + event_type + "\n" + "Trigger: " + trigger + "\n" + "Possible roles: " + ", ".join(roles) + "\n") + "Arguments:"
        args = example.args
        arguments_list = []
        for role in roles:
            for item in args:
                if role == item['role']:
                    arguments_list.append({"type": role, "argument": item['text']})
        arguments_json = json.dumps(arguments_list, indent=2)
        examples_out['input'] = instruction + input
        examples_out['output'] = arguments_json
        outs.append(examples_out)

            
    offset = len(outs)
    for i, example in enumerate(similarity_examples_5):
        examples_out = {}
        event_type = example.type
        retrieval_ctxs = similar_sentences[i+offset]["similar_sentence"]
        retrieval_ctxs = "\n".join(retrieval_ctxs[:5])
        roles = roles_dict[1][event_type]
        trigger = example.trigger['text']
        instruction = """Task description: Given a document and an event, you need to identify all arguments of this event, and classify the role of this argument. Limit responses to arguments only.  Please directly answer in JSON format of [{"type": <role>, "argument": <argument>}, {"type": <role>, "argument": <argument>}, ...].\n""" + "Examples:" + retrieval_ctxs + "\nExamples end here.\n" + \
                    "Question:" + '\n'
        input = ("Document: " + " ".join(example.sent) + "\n" + "Event: " + event_type + "\n" + "Trigger: " + trigger + "\n" + "Possible roles: " + ", ".join(roles) + "\n") + "Arguments:"
        args = example.args
        arguments_list = []
        for role in roles:
            for item in args:
                if role == item['role']:
                    arguments_list.append({"type": role, "argument": item['text']})
        arguments_json = json.dumps(arguments_list, indent=2)
        examples_out['input'] = instruction + input
        examples_out['output'] = arguments_json
        outs.append(examples_out)

    offset = len(outs)
    for i, example in enumerate(random_examples_5):
        examples_out = {}
        event_type = example.type
        retrieval_ctxs = similar_sentences[i+offset]["similar_sentence"]
        retrieval_ctxs = "\n".join(random.sample(retrieval_ctxs, 5))
        roles = roles_dict[1][event_type]
        trigger = example.trigger['text']
        instruction = """Task description: Given a document and an event, you need to identify all arguments of this event, and classify the role of this argument. Limit responses to arguments only.  Please directly answer in JSON format of [{"type": <role>, "argument": <argument>}, {"type": <role>, "argument": <argument>}, ...].\n""" + "Examples:" + retrieval_ctxs + "\nExamples end here.\n" + \
                    "Question:" + '\n'
        input = ("Document: " + " ".join(example.sent) + "\n" + "Event: " + event_type + "\n" + "Trigger: " + trigger + "\n" + "Possible roles: " + ", ".join(roles) + "\n") + "Arguments:"
        args = example.args
        arguments_list = []
        for role in roles:
            for item in args:
                if role == item['role']:
                    arguments_list.append({"type": role, "argument": item['text']})
        arguments_json = json.dumps(arguments_list, indent=2)
        examples_out['input'] = instruction + input
        examples_out['output'] = arguments_json
        outs.append(examples_out)

    offset = len(outs)
    for i, example in enumerate(similarity_examples_10):
        examples_out = {}
        event_type = example.type
        retrieval_ctxs = similar_sentences[i+offset]["similar_sentence"]
        retrieval_ctxs = "\n".join(retrieval_ctxs[:10])
        roles = roles_dict[1][event_type]
        trigger = example.trigger['text']
        instruction = """Task description: Given a document and an event, you need to identify all arguments of this event, and classify the role of this argument. Limit responses to arguments only.  Please directly answer in JSON format of [{"type": <role>, "argument": <argument>}, {"type": <role>, "argument": <argument>}, ...].\n""" + "Examples:" + retrieval_ctxs + "\nExamples end here.\n" + \
                    "Question:" + '\n'
        input = ("Document: " + " ".join(example.sent) + "\n" + "Event: " + event_type + "\n" + "Trigger: " + trigger + "\n" + "Possible roles: " + ", ".join(roles) + "\n") + "Arguments:"
        args = example.args
        arguments_list = []
        for role in roles:
            for item in args:
                if role == item['role']:
                    arguments_list.append({"type": role, "argument": item['text']})
        arguments_json = json.dumps(arguments_list, indent=2)
        examples_out['input'] = instruction + input
        examples_out['output'] = arguments_json
        outs.append(examples_out)

    offset = len(outs)
    for i, example in enumerate(random_examples_10):
        examples_out = {}
        event_type = example.type
        retrieval_ctxs = similar_sentences[i+offset]["similar_sentence"]
        retrieval_ctxs = "\n".join(random.sample(retrieval_ctxs, 10))
        roles = roles_dict[1][event_type]
        trigger = example.trigger['text']
        instruction = """Task description: Given a document and an event, you need to identify all arguments of this event, and classify the role of this argument. Limit responses to arguments only.  Please directly answer in JSON format of [{"type": <role>, "argument": <argument>}, {"type": <role>, "argument": <argument>}, ...].\n""" + "Examples:" + retrieval_ctxs + "\nExamples end here.\n" + \
                    "Question:" + '\n'
        input = ("Document: " + " ".join(example.sent) + "\n" + "Event: " + event_type + "\n" + "Trigger: " + trigger + "\n" + "Possible roles: " + ", ".join(roles) + "\n") + "Arguments:"
        args = example.args
        arguments_list = []
        for role in roles:
            for item in args:
                if role == item['role']:
                    arguments_list.append({"type": role, "argument": item['text']})
        arguments_json = json.dumps(arguments_list, indent=2)
        examples_out['input'] = instruction + input
        examples_out['output'] = arguments_json
        outs.append(examples_out)

    offset = len(outs)
    for i, example in enumerate(similarity_examples_15):
        examples_out = {}
        event_type = example.type
        retrieval_ctxs = similar_sentences[i+offset]["similar_sentence"]
        retrieval_ctxs = "\n".join(retrieval_ctxs[:15])
        roles = roles_dict[1][event_type]
        trigger = example.trigger['text']
        instruction = """Task description: Given a document and an event, you need to identify all arguments of this event, and classify the role of this argument. Limit responses to arguments only.  Please directly answer in JSON format of [{"type": <role>, "argument": <argument>}, {"type": <role>, "argument": <argument>}, ...].\n""" + "Examples:" + retrieval_ctxs + "\nExamples end here.\n" + \
                    "Question:" + '\n'
        input = ("Document: " + " ".join(example.sent) + "\n" + "Event: " + event_type + "\n" + "Trigger: " + trigger + "\n" + "Possible roles: " + ", ".join(roles) + "\n") + "Arguments:"
        args = example.args
        arguments_list = []
        for role in roles:
            for item in args:
                if role == item['role']:
                    arguments_list.append({"type": role, "argument": item['text']})
        arguments_json = json.dumps(arguments_list, indent=2)
        examples_out['input'] = instruction + input
        examples_out['output'] = arguments_json
        outs.append(examples_out)

    offset = len(outs)
    for i, example in enumerate(random_examples_15):
        examples_out = {}
        event_type = example.type
        retrieval_ctxs = similar_sentences[i+offset]["similar_sentence"]
        retrieval_ctxs = "\n".join(random.sample(retrieval_ctxs, 15))
        roles = roles_dict[1][event_type]
        trigger = example.trigger['text']
        instruction = """Task description: Given a document and an event, you need to identify all arguments of this event, and classify the role of this argument. Limit responses to arguments only.  Please directly answer in JSON format of [{"type": <role>, "argument": <argument>}, {"type": <role>, "argument": <argument>}, ...].\n""" + "Examples:" + retrieval_ctxs + "\nExamples end here.\n" + \
                    "Question:" + '\n'
        input = ("Document: " + " ".join(example.sent) + "\n" + "Event: " + event_type + "\n" + "Trigger: " + trigger + "\n" + "Possible roles: " + ", ".join(roles) + "\n") + "Arguments:"
        args = example.args
        arguments_list = []
        for role in roles:
            for item in args:
                if role == item['role']:
                    arguments_list.append({"type": role, "argument": item['text']})
        arguments_json = json.dumps(arguments_list, indent=2)
        examples_out['input'] = instruction + input
        examples_out['output'] = arguments_json
        outs.append(examples_out)
    return outs

def read_json(path):
    """ Read a json file from the given path."""
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def read_test(input_file, dataset):
    lines = read_jsonlines(input_file)
    out = []
    if dataset == "rams":
        examples = create_example_rams(lines)
        roles_dict = read_roles('/223040263/wanlong/LLM_Retreival/RAG4RE/data/dset_meta/description_rams.csv')
    elif dataset == "WikiEvent":
        examples = create_example_wikievent(lines)
        roles_dict = read_roles('/223040263/wanlong/LLM_Retreival/RAG4RE/data/dset_meta/description_wikievent.csv')
    elif dataset == "ace_eeqa":
        examples = create_example_ace(lines)
        roles_dict = read_roles('/223040263/wanlong/LLM_Retreival/RAG4RE/data/dset_meta/description_ace.csv')
    for example in examples:
        examples_out = {}
        event_type = example.type
        roles = roles_dict[1][event_type]
        trigger = example.trigger['text']
        instruction = """Task description: Given a document and an event, you need to identify all arguments of this event, and classify the role of this argument. Limit responses to arguments only.  Please directly answer in JSON format of [{"type": <role>, "argument": <argument>}, {"type": <role>, "argument": <argument>}, ...].\n""" +  "Examples:" + '\n' + \
                         'Document: Transportation officials are urging carpool and teleworking as options to combat an expected flood of drivers on the road . ( Paul Duggan ) -- A Baltimore prosecutor accused a police detective of “ sabotaging ” investigations related to the death of Freddie Gray , accusing him of fabricating notes to suggest that the state ’s medical examiner believed the manner of death was an accident rather than a homicide . The heated exchange came in the chaotic sixth day of the trial of Baltimore Officer Caesar Goodson Jr. , who drove the police van in which Gray suffered a fatal spine injury in 2015 . ( Derek Hawkins and Lynh Bui )\nEvent: life.die.deathcausedbyviolentevents\nTrigger: homicide\nPossible roles: instrument, killer, victim, place\nArguments: [{"type": "killer", "argument": "Officer Caesar Goodson Jr."}, {"type": "victim", "argument": "Freddie Gray"}, {"type": "place", "argument": "Baltimore"}]\nExamples end here.\n' + \
                         "Question:" + '\n'
        input = ("Document: " + " ".join(example.full_text) + "\n" + "Event: " + event_type + "\n" + "Trigger: " + trigger + "\n" + "Possible roles: " + ", ".join(roles) + "\n") + "Arguments:"

        args = example.args
        arguments_list = []
        for role in roles:
            for item in args:
                if role == item['role']:
                    arguments_list.append({"type": role, "argument": item['text']})
        arguments_json = json.dumps(arguments_list, indent=2)
        examples_out['input'] = instruction + input
        examples_out['output'] = arguments_json
        out.append(examples_out)
    return out

import numpy as np

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

import re
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
def semeval_prompt_template_rag(sentence, relation, head, tail, head_name, tail_name, context, topk):
    """ Get rag template
    Args:
        sentence: input sentence
        relation: relation type
    return: rag template
    """
    context = "\n".join(context[:topk])
    template_zero_shot = """Problem Definition: Relation extraction is to identify the relationship between two entities in a sentence.\n Please directly answer the relation_type of the following question.\n""" +\
                        """ Examples:"""+str(context)+ """\n""" + \
                        """ Question : What is the relation type between """+head+""" and """+tail+""" entities according to given relationships below in the following sentence, considering example sentence and its relationship?\n""" +\
                        """ Query Sentence:""" + str(sentence)+ """\n""" +\
                        """ e1: """ + head_name + """. \n""" +\
                        """ e2 : """ + tail_name + """. \n""" +\
                        """ Relation types: """ + str(relation) + """. \n""" +\
                        """ Output format: relation_type.\n"""  + \
                        " Please directly answer the relation_type\n"  + \
                        '''Answer:'''
    return template_zero_shot
def semeval_prompt_template(sentence, relation, head, tail, head_name, tail_name):
    """ Get zero shot template
    Args:
        sentence: input sentence
        relation: relation type
    return: zero shot template
    """
    template_zero_shot = """Problem Definition: Relation extraction is to identify the relationship between two entities in a sentence.\n Please directly answer the relation_type of the following question.\n""" +\
    """Question : What is the relation type between """+head+""" and """+tail+""" entities in the following sentence?\n""" +\
                        """ Sentence:""" + str(sentence)+ """\n""" +\
                        """ e1: """ + head_name + """. \n""" +\
                        """ e2 : """ + tail_name + """. \n""" +\
                        """ Relation types: """ + str(relation) + """. \n""" +\
                        """ output format: relation_type""" +  " Please directly answer the relation_type\n" +  \
                        '''Answer:'''
    return template_zero_shot


def get_zero_shot_template_tacred(sentence, relation, head, tail):
    """ Get zero shot template
    Args:
        sentence: input sentence
        relation: relation type
    return: zero shot template
    """
    template_zero_shot ="""Problem Definition: Relation extraction is to identify the relationship between two entities in a sentence.\n""" +\
                        """Question : What is the relation type between head and tail entities in the following sentence?\n""" +\
                        """ Sentence:""" + str(sentence)+ """\n""" +\
                        """ Head entity: """ + head + """. \n""" +\
                        """ Tail entity: """ + tail + """. \n""" +\
                        """ Relation types: """ + str(relation) + """. \n""" +\
                        """ Output format: relation_type\n""" +  "Please directly answer the relation_type between the head and tail entities from the following relation list \n" +  \
                        """ Relation types: """ + str(relation) + """. \n""" +\
                        '''Answer: '''
    return template_zero_shot

def get_zero_shot_template_tacred_rag(sentence, relation, head, tail, context, topk):
    """ Get rag template
    Args:
        sentence: input sentence
        relation: relation type
    return: rag template
    """
    context = "\n".join(context[:topk])
    template_zero_shot = """Problem Definition: Relation extraction is to identify the relationship between two entities in a sentence.\n""" +\
                        """ Examples: """+ str(context)+ """\n""" +\
                        """ Question : What is the relation type between tail and head entities according to given relationships below in the following sentence?\n""" +\
                        """ Query Sentence:""" + str(sentence)+ """\n""" +\
                        """ Head entity: """ + head + """. \n""" +\
                        """ Tail entity: """ + tail + """. \n""" +\
                        """ Output format: relation_type\n""" + \
                          "Please directly answer the relation_type between the head and tail entities from the following relation list \n" +  \
                        """ Relation types: """ + str(relation) + """. \n""" +\
                        '''Answer:'''
    return template_zero_shot