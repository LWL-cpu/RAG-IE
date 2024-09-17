




python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/rams/test.jsonlines \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/rams/train.jsonlines \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/rams/test_rams_similarities.json \
                      --dataset rams \
                      --prompt_type diversity_rag \
                      --model_name /sds_wangby/models/Meta-Llama-3-8B-Instruct \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/rams/llama3_rams_diversity_rag_5_doc.json \
                      --topk 5

python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/rams/test.jsonlines \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/rams/train.jsonlines \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/rams/test_rams_similarities.json \
                      --dataset rams \
                      --prompt_type diversity_rag \
                      --model_name /sds_wangby/models/Meta-Llama-3-8B-Instruct \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/rams/llama3_rams_diversity_rag_10_doc.json \
                      --topk 10

python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/rams/test.jsonlines \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/rams/train.jsonlines \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/rams/test_rams_similarities.json \
                      --dataset rams \
                      --prompt_type diversity_rag \
                      --model_name /sds_wangby/models/Meta-Llama-3-8B-Instruct \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/rams/llama3_rams_diversity_rag_15_doc.json \
                      --topk 15


