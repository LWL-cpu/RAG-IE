""" This script is used to compute the sentence embeddings for the sentences in the dataset."""
"""Created by: Sefika"""
import os
import sys
import json
from sentence_transformers import SentenceTransformer
import numpy as np
import configparser
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
PREFIX_PATH = "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[:-2]) + "/"

import jsonlines
import copy
import csv
from itertools import chain
# 替换为你的OpenAI API密钥
def read_jsonlines(input_file):
    lines = []
    with jsonlines.open(input_file) as reader:
        for obj in reader:
            lines.append(obj)
    return lines


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

    print("{} examples collected. {} arguments dropped.".format(len(examples), invalid_arg_num))
    return examples

def process_rams(path):
    lines = read_jsonlines(path)
    examples = create_example_rams(lines)
    for example in examples:
        print()

# result = process_rams()

def read_json(path):
    """ Read json file"""
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def write_json(path, data):
    """ Write json file"""
    
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
        
    with open(path, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        
def compute_sentence(data):
    """Compute the sentence embeddings for the sentences in the dataset
    Args:
        data (list): list of sentences
    Returns:
        list: list of sentence embeddings
    """
    sent_embeddings = []
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("The embeddings will be compted for {0} sentences".format(len(data)))

    for i, sent in enumerate(data):
        clean_sent = clean_sentence(sent)
        embeddings = model.encode(clean_sent)
        sent_embeddings.append(embeddings)
        print("Processed sentence: ", i)

    print("The embeddings were completed for {0} sentences".format(len(sent_embeddings)))

    return sent_embeddings

def clean_sentence(sent):
    """Clean the sentence from the entity tags"""
    sent = sent.replace("<e1>", "")
    sent = sent.replace("</e1>", "")
    sent = sent.replace("<e2>", "")
    sent = sent.replace("</e2>", "")

    return sent

def write_embeddings(embeddings, output_file):
    np.save(output_file, embeddings)


def read_train_rag(input_file, dataset):
    lines = read_jsonlines(input_file)
    examples_out = []
    if dataset == "rams":
        examples = create_example_rams(lines)
        roles_dict = read_roles('/223040263/wanlong/LLM_Retreival/RAG4RE/data/dset_meta/description_rams.csv')
        for example in examples:
            event_type = example.type
            roles = roles_dict[1][event_type]
            trigger = example.trigger['text']
            all_content = "Document: " + " ".join(example.full_text) + "\n" + "Event: " + event_type + "\n" + "Trigger: " + trigger + "\n" + "Possible roles: " + ", ".join(roles) + "\n"
            args = example.args
            arguments = []
            for role in roles:
                for item in args:
                    if role == item['role']:
                        arguments.append( "( " + "type: " + role + ", argument: " + item['text'] + " )")
                    # else:
                    #     arguments.append( "( " + "type: " + role + ", argument: " + "None" + " )")
            arguments = ", ".join(arguments)
            arguments = "Arguments: " + arguments
            all_content = all_content + arguments
            examples_out.append(all_content)
    return examples_out

def read_train(input_file, dataset):
    lines = read_jsonlines(input_file)
    examples_out = []
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
        event_type = example.type
        roles = roles_dict[1][event_type]
        trigger = example.trigger['text']
        all_content =  "Document: " + " ".join(example.sent) + "\n" + "Event: " + event_type + "\n" + "Trigger: " + trigger + "\n" + "Possible roles: " + ", ".join(roles) + "\n"
        args = example.args
        arguments_list = []
        if len(roles) == 0:
            continue
        for role in roles:
            for item in args:
                if role == item['role']:
                    arguments_list.append({"type": role, "argument": item['text']})
        arguments_json = json.dumps(arguments_list, indent=2)
        arguments = "Output: " + arguments_json
        all_content = all_content + arguments
        event_type = example.type
        roles = roles_dict[1][event_type]
        examples_out.append([event_type, all_content])     
    
    return examples_out

def process_instruction(input_file, dataset):
    lines = read_jsonlines(input_file)
    out = []
    if dataset == "rams":
        examples = create_example_rams(lines)
        roles_dict = read_roles('/223040263/wanlong/LLM_Retreival/RAG4RE/data/dset_meta/description_rams.csv')
        for example in examples:
            examples_out = {}
            event_type = example.type
            roles = roles_dict[1][event_type]
            trigger = example.trigger['text']
            instruction = ("Document: " + " ".join(example.full_text) + "\n" + "Event: " + event_type + "\n" + "Trigger: " + trigger + "\n" + "Possible roles: " + ", ".join(roles) + "\n") + "Output: "

            args = example.args
            arguments = []
            for role in roles:
                for item in args:
                    if role == item['role']:
                        arguments.append( "( " + "type: " + role + ", argument: " + item['text'] + " )")
                    # else:
                    #     arguments.append( "( " + "type: " + role + ", argument: " + "None" + " )")
            arguments = ", ".join(arguments)

            examples_out['instruction'] = """Task description: Given a document and an event, you need to identify all arguments of this event, and classify the role of this argument. Limit responses to arguments only.  Please answer in JSON format of [{"type": <role>, "argument": <argument>}, {"type": <role>, "argument": <argument>}, ...].\n""" + \
                         "Examples:" + '\n' + \
                         'Document: Transportation officials are urging carpool and teleworking as options to combat an expected flood of drivers on the road . ( Paul Duggan ) -- A Baltimore prosecutor accused a police detective of “ sabotaging ” investigations related to the death of Freddie Gray , accusing him of fabricating notes to suggest that the state ’s medical examiner believed the manner of death was an accident rather than a homicide . The heated exchange came in the chaotic sixth day of the trial of Baltimore Officer Caesar Goodson Jr. , who drove the police van in which Gray suffered a fatal spine injury in 2015 . ( Derek Hawkins and Lynh Bui )\nEvent: life.die.deathcausedbyviolentevents\nTrigger: homicide\nPossible roles: instrument, killer, victim, place\nArguments: [{"type": "killer", "argument": "Officer Caesar Goodson Jr."}, {"type": "victim", "argument": "Freddie Gray"}, {"type": "place", "argument": "Baltimore"}]\nExamples end here.\n' + \
                         "Question: " + '\n' + instruction
            examples_out['output'] = arguments
            out.append(examples_out)
    return out

def read_test(input_file, dataset):
    lines = read_jsonlines(input_file)
    examples_out = []
    if dataset == "rams":
        examples = create_example_rams(lines)
        roles_dict = read_roles('/223040263/wanlong/LLM_Retreival/RAG4RE/data/dset_meta/description_rams.csv')
    elif dataset == "WikiEvent":
        examples = create_example_wikievent(lines)
        roles_dict = read_roles('/223040263/wanlong/LLM_Retreival/RAG4RE/data/dset_meta/description_wikievent.csv')
    elif dataset == "ace_eeqa":
        examples = create_example_ace(lines)
        roles_dict = read_roles('/223040263/wanlong/LLM_Retreival/RAG4RE/data/dset_meta/description_ace.csv')
    elif dataset == "tacred":
        examples = read_json(input_file)
    if dataset == "tacred":  
        examples_out = []
        for example in examples:
            relation_type = example["relation"]
            tokens = " ".join(example["token"])
            # sub_s = example["subj_start"]
            # sub_e = example["subj_end"]
            # obj_s = example["obj_start"]
            # obj_e = example["obj_end"]
            # subject_entity = token[sub_s:sub_e+1]
            # object_entity = token[obj_s:obj_e+1]
            examples_out.append(tokens)
        return examples_out

    else:
        for example in examples:
            event_type = example.type
            roles = roles_dict[1][event_type]
            trigger = example.trigger['text']
            all_content = "Document: " + " ".join(
                example.sent) + "\n" + "Event: " + event_type + "\n" + "Trigger: " + trigger + "\n" + "Possible roles: " + ", ".join(
                roles) + "\n"
            event_type = example.type
            examples_out.append([event_type, all_content])
    return examples_out

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
if __name__ == "__main__":

    print("PREFIX_PATH", PREFIX_PATH)

    config = configparser.ConfigParser()
    config.read(PREFIX_PATH+"config.ini")

    input_file = config["EMBEDDING"]["input_embedding_path"]
    output_file = config["EMBEDDING"]["output_embedding_path"]
    dataset_type = input_file.split('/')[-1].split('.')[0]
    dataset = input_file.split('/')[-2]

    if "tacred" in input_file or "semeval" in input_file:
        data = read_json(input_file)
        data_new = []
        for example in data:
            if "relation" in example:
                relation_type = example["relation"]
            elif "label" in example:
                relation_type = example["label"]
            else:
                relation_type = "NA"
            # tokens = " ".join(example["token"])
            tokens = " The relation between head and tail entities is " + relation_type
            # sub_s = example["subj_start"]
            # sub_e = example["subj_end"]
            # obj_s = example["obj_start"]
            # obj_e = example["obj_end"]
            # subject_entity = token[sub_s:sub_e+1]
            # object_entity = token[obj_s:obj_e+1]
            data_new.append(tokens)
    else:
        if "train" in dataset_type:
            data = read_train(input_file, dataset)
        else:
            data = read_test(input_file, dataset)
        data_new = [item[1] for item in data]
    embeddings = compute_sentence(data_new)
    write_embeddings(embeddings, output_file)