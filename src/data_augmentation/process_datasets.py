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
            instruction = """Task description: Given a document and an event, you need to identify all arguments of this event, and classify the role of this argument. Limit responses to arguments only.  Please answer in JSON format of [{"type": <role>, "argument": <argument>}, {"type": <role>, "argument": <argument>}, ...].\nQuestion:\n"""
            input = ("Document: " + " ".join(example.full_text) + "\n" + "Event: " + event_type + "\n" + "Trigger: " + trigger + "\n" + "Possible roles: " + ", ".join(roles) + "\n") + "Arguments: "

            args = example.args
            arguments_list = []
            for role in roles:
                for item in args:
                    if role == item['role']:
                        arguments_list.append({"type": role, "argument": item['text']})
            arguments_json = json.dumps(arguments_list, indent=2)

            examples_out['instruction'] = instruction
            examples_out['input'] = input
            examples_out['output'] = arguments_json
            out.append(examples_out)
    return out