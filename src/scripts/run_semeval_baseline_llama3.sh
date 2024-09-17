python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/semeval/original_data/test_sentences.json \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/semeval/original_data/train_sentences.json \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/semeval/similarity_results/sentence_sim_full.json \
                      --dataset semeval \
                      --prompt_type simple \
                      --model_name /sds_wangby/models/Meta-Llama-3-8B-Instruct \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/semeval/llama3_semeval_no_rag.json


python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/semeval/original_data/test_sentences.json \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/semeval/original_data/train_sentences.json \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/semeval/similarity_results/sentence_sim_full.json \
                      --dataset semeval \
                      --prompt_type rag \
                      --model_name /sds_wangby/models/Meta-Llama-3-8B-Instruct \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/semeval/llama3_semeval_rag_1_doc.json \
                      --topk 1

python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/semeval/original_data/test_sentences.json \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/semeval/original_data/train_sentences.json \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/semeval/similarity_results/sentence_sim_full.json \
                      --dataset semeval \
                      --prompt_type rag \
                      --model_name /sds_wangby/models/Meta-Llama-3-8B-Instruct \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/semeval/llama3_semeval_rag_3_doc.json \
                      --topk 3

python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/semeval/original_data/test_sentences.json \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/semeval/original_data/train_sentences.json \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/semeval/similarity_results/sentence_sim_full.json \
                      --dataset semeval \
                      --prompt_type rag \
                      --model_name /sds_wangby/models/Meta-Llama-3-8B-Instruct \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/semeval/llama3_semeval_rag_5_doc.json \
                      --topk 5

python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/semeval/original_data/test_sentences.json \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/semeval/original_data/train_sentences.json \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/semeval/similarity_results/sentence_sim_full.json \
                      --dataset semeval \
                      --prompt_type rag \
                      --model_name /sds_wangby/models/Meta-Llama-3-8B-Instruct \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/semeval/llama3_semeval_rag_10_doc.json \
                      --topk 10

python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/semeval/original_data/test_sentences.json \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/semeval/original_data/train_sentences.json \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/semeval/similarity_results/sentence_sim_full.json \
                      --dataset semeval \
                      --prompt_type rag \
                      --model_name /sds_wangby/models/Meta-Llama-3-8B-Instruct \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/semeval/llama3_semeval_rag_15_doc.json \
                      --topk 15
python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/semeval/original_data/test_sentences.json \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/semeval/original_data/train_sentences.json \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/semeval/similarity_results/sentence_sim_full.json \
                      --dataset semeval \
                      --prompt_type rag \
                      --model_name /sds_wangby/models/Meta-Llama-3-8B-Instruct \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/semeval/llama3_semeval_rag_20_doc.json \
                      --topk 20
