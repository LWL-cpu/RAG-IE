from retrieval.retriever import benchmark_data_augmentation_call_EE, benchmark_data_augmentation_call_RE
import os
import argparse
import configparser
def main():
    parser = argparse.ArgumentParser(description="Process the config file path.")
    parser.add_argument('--test_data_path', type=str, default="/223040263/wanlong/LLM_Retreival/RAG4RE/data/tacred/test_process.json", help='Path to the test data file.')
    parser.add_argument('--train_data_path', type=str, default="/223040263/wanlong/LLM_Retreival/RAG4RE/data/tacred/train_process.json", help='Path to the train data file.')
    parser.add_argument('--similar_sentences_path', default="/223040263/wanlong/LLM_Retreival/RAG4RE/data/tacred/test_tacred_similarities.json", type=str,  help='Path to the similar sentences file.')
    parser.add_argument('--dataset', type=str, default="tacred",  help='Dataset setting.')
    parser.add_argument('--prompt_type', type=str, default="rag", help='Prompt type setting.')
    parser.add_argument('--model_name', type=str, default ="/sds_wangby/models/Meta-Llama-3-8B-Instruct",  help='Model name setting.')
    parser.add_argument('--responses_path', default="/223040263/wanlong/LLM_Retreival/RAG4RE/outputs/rams/llama3_rams_no_rag.json", type=str, help='Output path for RAG test responses.')
    parser.add_argument('--topk', type=int, default=3, help='The number of retrieval docs')
    parser.add_argument('--task', type=str, default="RE", help='Task type. EE or RE')
    parser.add_argument('--test_label_path', type=str, default="/223040263/wanlong/LLM_Retreival/RAG4RE/data/semeval/original_data/test_relations.json", help='Test_label_path for RE')
    parser.add_argument('--relation_path', type=str, default="/223040263/wanlong/LLM_Retreival/RAG4RE/data/semeval/original_data/relations.json", help='Relation file, only for RE tasks.')
    parser.add_argument('--train_label_path', type=str, default="/223040263/wanlong/LLM_Retreival/RAG4RE/data/semeval/original_data/train_relations.json", help='Train_label_path for RE')

    
    args = parser.parse_args()
    if args.task == "EE":
        benchmark_data_augmentation_call_EE(args)
    elif args.task == "RE":
        benchmark_data_augmentation_call_RE(args)
    else:
        print("Not implemented yet!")

if __name__ == "__main__":
    main()


