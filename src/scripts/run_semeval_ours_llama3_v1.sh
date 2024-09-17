python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/semeval/original_data/test_sentences.json \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/semeval/original_data/train_sentences.json \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/semeval/similarity_results/sentence_sim_full.json \
                      --dataset semeval \
                      --prompt_type simple \
                      --model_name /223040263/wanlong/LLM_Retreival/RAG4RE/src/train/train_v3/EAE_train_no_retrieval/checkpoint-2-3327/tfmr \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/semeval/llama3_semeval_no_rag.json \
                      --task RE


python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/semeval/original_data/test_sentences.json \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/semeval/original_data/train_sentences.json \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/semeval/similarity_results/sentence_sim_full.json \
                      --dataset semeval \
                      --prompt_type rag \
                      --model_name /223040263/wanlong/LLM_Retreival/RAG4RE/src/train/train_v3/EAE_train_no_retrieval/checkpoint-2-3327/tfmr \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/semeval/llama3_semeval_rag_1_doc.json \
                      --topk 1 \
                      --task RE

python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/semeval/original_data/test_sentences.json \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/semeval/original_data/train_sentences.json \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/semeval/similarity_results/sentence_sim_full.json \
                      --dataset semeval \
                      --prompt_type rag \
                      --model_name /223040263/wanlong/LLM_Retreival/RAG4RE/src/train/train_v3/EAE_train_no_retrieval/checkpoint-2-3327/tfmr \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/semeval/llama3_semeval_rag_3_doc.json \
                      --topk 3 \
                      --task RE

python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/semeval/original_data/test_sentences.json \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/semeval/original_data/train_sentences.json \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/semeval/similarity_results/sentence_sim_full.json \
                      --dataset semeval \
                      --prompt_type rag \
                      --model_name /223040263/wanlong/LLM_Retreival/RAG4RE/src/train/train_v3/EAE_train_no_retrieval/checkpoint-2-3327/tfmr \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/semeval/llama3_semeval_rag_5_doc.json \
                      --topk 5 \
                      --task RE

python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/semeval/original_data/test_sentences.json \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/semeval/original_data/train_sentences.json \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/semeval/similarity_results/sentence_sim_full.json \
                      --dataset semeval \
                      --prompt_type rag \
                      --model_name /223040263/wanlong/LLM_Retreival/RAG4RE/src/train/train_v3/EAE_train_no_retrieval/checkpoint-2-3327/tfmr \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/semeval/llama3_semeval_rag_10_doc.json \
                      --topk 10 \
                      --task RE

python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/semeval/original_data/test_sentences.json \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/semeval/original_data/train_sentences.json \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/semeval/similarity_results/sentence_sim_full.json \
                      --dataset semeval \
                      --prompt_type rag \
                      --model_name /223040263/wanlong/LLM_Retreival/RAG4RE/src/train/train_v3/EAE_train_no_retrieval/checkpoint-2-3327/tfmr \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/semeval/llama3_semeval_rag_15_doc.json \
                      --topk 15 \
                      --task RE

