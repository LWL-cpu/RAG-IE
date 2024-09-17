
python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/WikiEvent/test.jsonl \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/WikiEvent/train.jsonl \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/WikiEvent/test_wiki_similarities.json \
                      --dataset wiki \
                      --prompt_type diversity_rag \
                      --model_name /sds_wangby/models/Meta-Llama-3-8B-Instruct \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/wiki/llama3_wiki_diversity_rag_5_doc.json \
                      --topk 5


python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/WikiEvent/test.jsonl \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/WikiEvent/train.jsonl \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/WikiEvent/test_wiki_similarities.json \
                      --dataset wiki \
                      --prompt_type diversity_rag \
                      --model_name /sds_wangby/models/Meta-Llama-3-8B-Instruct \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/wiki/llama3_wiki_diversity_rag_10_doc.json \
                      --topk 10


python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/WikiEvent/test.jsonl \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/WikiEvent/train.jsonl \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/WikiEvent/test_wiki_similarities.json \
                      --dataset wiki \
                      --prompt_type diversity_rag \
                      --model_name /sds_wangby/models/Meta-Llama-3-8B-Instruct \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/wiki/llama3_wiki_diversity_rag_15_doc.json \
                      --topk 15


                                            



python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/ace_eeqa/test_convert.json \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/ace_eeqa/train_convert.json \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/ace_eeqa/test_ace_similarities.json \
                      --dataset ace2005 \
                      --prompt_type diversity_rag \
                      --model_name /sds_wangby/models/Meta-Llama-3-8B-Instruct \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/ace2005/llama3_ace2005_diversity_rag_1_doc.json \
                      --topk 1


python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/ace_eeqa/test_convert.json \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/ace_eeqa/train_convert.json \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/ace_eeqa/test_ace_similarities.json \
                      --dataset ace2005 \
                      --prompt_type diversity_rag \
                      --model_name /sds_wangby/models/Meta-Llama-3-8B-Instruct \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/ace2005/llama3_ace2005_diversity_rag_5_doc.json \
                      --topk 5

python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/ace_eeqa/test_convert.json \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/ace_eeqa/train_convert.json \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/ace_eeqa/test_ace_similarities.json \
                      --dataset ace2005 \
                      --prompt_type diversity_rag \
                      --model_name /sds_wangby/models/Meta-Llama-3-8B-Instruct \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/ace2005/llama3_ace2005_diversity_rag_10_doc.json \
                      --topk 10

python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/ace_eeqa/test_convert.json \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/ace_eeqa/train_convert.json \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/ace_eeqa/test_ace_similarities.json \
                      --dataset ace2005 \
                      --prompt_type diversity_rag \
                      --model_name /sds_wangby/models/Meta-Llama-3-8B-Instruct \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/ace2005/llama3_ace2005_diversity_rag_15_doc.json \
                      --topk 15
