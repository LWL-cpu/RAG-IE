import os
import sys
import json
import numpy as np
from numpy.linalg import norm
from sentence_embeddings import *
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
PREFIX_PATH = "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[:-2]) + "/"


import configparser

def read_json(path):
    """ Read json file"""
    with open(path, 'r') as f:
        data = json.load(f)
    return data

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

def write_json(path, data):
    """ Write json file"""
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
        
    with open(path, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        


def compute_similarity(test_data, train_data, train_embeddings, test_embeddings):
    """Compute Consine similarity between test and train embeddings

    Args:
        test_data (list): list of sentences
        train_data (list): list of sentences
        train_embeddings (list): list of sentence embeddings
        test_embeddings (list): list of sentence embeddings

    Returns:
        list: list of similarity scores along with similar sentence and train data index
    """

    similarities = []

    for test_index, _ in enumerate(test_data):
        test_emb = test_embeddings[test_index]
        train_similarities = []
        # test_data_event_type = _[0]

        for train_index, train_line in enumerate(train_data):

            train_emb = train_embeddings[train_index]
            train_data_event_type = train_line[0]
            sim = np.dot(test_emb,train_emb)/(norm(test_emb)*norm(train_emb))

                
            context = train_line

            train_similarities.append({"train":train_index, "event_type": train_data_event_type, "simscore": sim, "sentence":context[1]})

        train_similarities = sorted(train_similarities, key=lambda x: x["simscore"], reverse=True)
            
        similarities.append({"test":test_index, "similar_sentence":[train_similarities[i]['sentence'] for i in range(20)],"train_idex":[train_similarities[i]['train'] for i in range(20)], "simscore":[float(train_similarities[i]['simscore']) for i in range(20)], "event_type":[str(train_similarities[i]['event_type']) for i in range(20)]})

        print("test index: ", test_index)

    return similarities

def semeval_similarity(test_data, train_data, train_embeddings, test_embeddings, train_label_path, relations_path):
    """Compute Consine similarity between test and train embeddings

    Args:
        test_data (list): list of sentences
        train_data (list): list of sentences
        train_embeddings (list): list of sentence embeddings
        test_embeddings (list): list of sentence embeddings

    Returns:
        list: list of similarity scores along with similar sentence and train data index
    """
    similarities = []
    train_labels = read_json(train_label_path)
    relations = read_json(relations_path)
    relations = relations['relation']['names']
    labels = [relations[item] for item in train_labels]
    for test_index, _ in enumerate(test_data):
        test_emb = test_embeddings[test_index]
        train_similarities = []
        # test_data_event_type = _[0]
        for train_index, train_line in enumerate(train_data):
            train_emb = train_embeddings[train_index]
            label = labels[train_index]
            # train_data_event_type = train_line[0]
            sim = np.dot(test_emb,train_emb)/(norm(test_emb)*norm(train_emb))             
            context = train_line
            train_similarities.append({"train":train_index, "simscore": sim, "sentence":context, "label": label})
            # train_similarities.append({"train":train_index, "event_type": train_data_event_type, "simscore": sim, "sentence":context[1]})
        train_similarities = sorted(train_similarities, key=lambda x: x["simscore"], reverse=True)
        similarities.append({"test":test_index, "similar_sentence":[train_similarities[i]['sentence'] for i in range(20)],"train_idex":[train_similarities[i]['train'] for i in range(50)], "simscore":[float(train_similarities[i]['simscore']) for i in range(50)], "label":[str(train_similarities[i]['label']) for i in range(50)]})   
        # similarities.append({"test":test_index, "similar_sentence":[train_similarities[i]['sentence'] for i in range(len(train_similarities))],"train_idex":[train_similarities[i]['train'] for i in range(len(train_similarities))], "simscore":[float(train_similarities[i]['simscore']) for i in range(len(train_similarities))], "event_type":[str(train_similarities[i]['event_type']) for i in range(len(train_similarities))]})
        print("test index: ", test_index)
    return similarities

import torch
# def tacred_similarity(test_data, train_data, train_embeddings, test_embeddings):
#     """Compute Consine similarity between test and train embeddings

#     Args:
#         test_data (list): list of sentences
#         train_data (list): list of sentences
#         train_embeddings (list): list of sentence embeddings
#         test_embeddings (list): list of sentence embeddings

#     Returns:
#         list: list of similarity scores along with similar sentence and train data index
#     """

#     similarities = []

#     for test_index, _ in enumerate(test_data):
#         test_emb = test_embeddings[test_index]
#         train_similarities = []
#         # test_data_event_type = _[0]

#         for train_index, train_line in enumerate(train_data):

#             train_emb = train_embeddings[train_index]
#             label = train_line["label"]
#             # train_data_event_type = train_line[0]
#             sim = np.dot(test_emb,train_emb)/(norm(test_emb)*norm(train_emb))

                
#             context = train_line
#             train_similarities.append({"train":train_index, "simscore": sim, "sentence":context, "label": label})
#             # train_similarities.append({"train":train_index, "event_type": train_data_event_type, "simscore": sim, "sentence":context[1]})

#         train_similarities = sorted(train_similarities, key=lambda x: x["simscore"], reverse=True)
#         similarities.append({"test":test_index, "similar_sentence":[train_similarities[i]['sentence'] for i in range(50)],"train_idex":[train_similarities[i]['train'] for i in range(50)], "simscore":[float(train_similarities[i]['simscore']) for i in range(50)], "label":[str(train_similarities[i]['label']) for i in range(50)]})
            
#         # similarities.append({"test":test_index, "similar_sentence":[train_similarities[i]['sentence'] for i in range(len(train_similarities))],"train_idex":[train_similarities[i]['train'] for i in range(len(train_similarities))], "simscore":[float(train_similarities[i]['simscore']) for i in range(len(train_similarities))], "event_type":[str(train_similarities[i]['event_type']) for i in range(len(train_similarities))]})

#         print("test index: ", test_index)

#     return similarities
def tacred_similarity(test_data, train_data, train_embeddings, test_embeddings):
    """Compute Consine similarity between test and train embeddings

    Args:
        test_data (list): list of sentences
        train_data (list): list of sentences
        train_embeddings (list): list of sentence embeddings
        test_embeddings (list): list of sentence embeddings

    Returns:
        list: list of similarity scores along with similar sentence and train data index
    """
    # Convert lists to numpy arrays for vectorized operations
    train_embeddings = np.array(train_embeddings)
    test_embeddings = np.array(test_embeddings)

    # Precompute norms for train and test embeddings
    train_norms = np.linalg.norm(train_embeddings, axis=1)
    test_norms = np.linalg.norm(test_embeddings, axis=1)

    similarities = []

    # Vectorized similarity calculation
    cos_sim_matrix = np.dot(test_embeddings, train_embeddings.T) / np.outer(test_norms, train_norms)

    for test_index in range(len(test_data)):
        # Get top 50 similar indices for each test case
        sim_indices = np.argsort(-cos_sim_matrix[test_index])[:50]
        sim_scores = cos_sim_matrix[test_index][sim_indices]

        similar_sentences = [train_data[idx] for idx in sim_indices]
        subjs = [train_data[idx]["subj"] for idx in sim_indices]
        objs = [train_data[idx]["obj"] for idx in sim_indices]
        labels = [train_data[idx]["label"] for idx in sim_indices]

        similarities.append({
            "test": test_index,
            "similar_sentence": similar_sentences,
            "train_index": sim_indices.tolist(),
            "simscore": sim_scores.tolist(),
            "subj": subjs,
            "obj": objs, 
            "label":labels
        })
        print("test index: ", test_index)

    return similarities

# def semeval_compute_similarity(test_data, train_data, train_embeddings, test_embeddings):
#     """Compute Consine similarity between test and train embeddings for semeval dataset
#     Args:
#         test_data (list): list of sentences
#         train_data (list): list of sentences
#         train_embeddings (list): list of sentence embeddings
#         test_embeddings (list): list of sentence embeddings
#     Returns:
#         list: list of similarity scores along with similar sentence and train data index
#     """
    
#     similarities = []

#     for test_index, _ in enumerate(test_data):
#         test_emb = test_embeddings[test_index]
#         train_similarities = []

#         for train_index, train_line in enumerate(train_data):
#             train_emb = train_embeddings[train_index]
#             sim = np.dot(test_emb,train_emb)/(norm(test_emb)*norm(train_emb))
#             train_similarities.append({"train":train_index, "simscore":sim, "sentence":train_line})
        
#         train_similarities = sorted(train_similarities, key=lambda x: x["simscore"], reverse=True)
            
#         similarities.append({"test":test_index, "similar_sentence":train_similarities[0]['sentence'],"train_idex":train_similarities[0]['train'], "simscore":float(train_similarities[0]['simscore'])})

#         print("test index: ", test_index)

#     return similarities


def main(test_file, train_file, train_emb, test_emb, output_sim_path, task):
    """Compute similarity between test and train embeddings"""

    

    # test_data = read_json(test_file)
    # train_data = read_json(train_file)

    train_embeddings = np.load(train_emb)
    test_embeddings = np.load(test_emb)
    if task == "EE": 
        test_data = read_test(test_file, test_file.split('/')[-2])
        train_data = read_train(train_file, train_file.split('/')[-2])
        similarities = compute_similarity(test_data, train_data, train_embeddings, test_embeddings)
    elif task == "RE":
        if "tacred" in train_file:
            test_data = read_json(test_file)
            train_data = read_json(train_file)
            similarities = tacred_similarity(test_data, train_data, train_embeddings, test_embeddings)

        else:
            relation_path = "/223040263/wanlong/LLM_Retreival/RAG4RE/data/semeval/original_data/relations.json"
            train_label_path = "/223040263/wanlong/LLM_Retreival/RAG4RE/data/semeval/original_data/train_relations.json"
            test_data = read_json(test_file)
            train_data = read_json(train_file)
            similarities = semeval_similarity(test_data, train_data, train_embeddings, test_embeddings, train_label_path, relation_path)
    write_json(output_sim_path, similarities)


if __name__ == "__main__":

    config = configparser.ConfigParser()
    config.read(PREFIX_PATH+"config.ini")

    test_file = config["SIMILARITY"]["test_file"]
    train_file = config["SIMILARITY"]["train_file"]
    train_emb = config["SIMILARITY"]["train_emb"]
    test_emb = config["SIMILARITY"]["test_emb"]
    output_sim_path = config["SIMILARITY"]["output_index"]
    task = "EE"

    main(test_file, train_file, train_emb, test_emb, output_sim_path, task)