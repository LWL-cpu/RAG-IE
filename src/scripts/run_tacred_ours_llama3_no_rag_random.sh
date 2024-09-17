
python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/tacred/test_process.json \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/tacred/train_process.json \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/tacred/test_tacred_similarities.json \
                      --dataset tacred \
                      --prompt_type random_rag \
                      --model_name /223040263/wanlong/LLM_Retreival/RAG4RE/src/train/train_RE_no_rag_v1/EAE_train_no_retrieval/checkpoint-2-2079/tfmr \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/tacred/llama3_ours_tacred_random_rag_1_doc.json \
                      --topk 1 \
                      --task RE


python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/tacred/test_process.json \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/tacred/train_process.json \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/tacred/test_tacred_similarities.json \
                      --dataset tacred \
                      --prompt_type random_rag \
                      --model_name /223040263/wanlong/LLM_Retreival/RAG4RE/src/train/train_RE_no_rag_v1/EAE_train_no_retrieval/checkpoint-2-2079/tfmr \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/tacred/llama3_ours_tacred_random_rag_5_doc.json \
                      --topk 5 \
                      --task RE

python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/tacred/test_process.json \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/tacred/train_process.json \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/tacred/test_tacred_similarities.json \
                      --dataset tacred \
                      --prompt_type random_rag \
                      --model_name /223040263/wanlong/LLM_Retreival/RAG4RE/src/train/train_RE_no_rag_v1/EAE_train_no_retrieval/checkpoint-2-2079/tfmr \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/tacred/llama3_ours_tacred_random_rag_10_doc.json \
                      --topk 10 \
                      --task RE

python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/tacred/test_process.json \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/tacred/train_process.json \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/tacred/test_tacred_similarities.json \
                      --dataset tacred \
                      --prompt_type random_rag \
                      --model_name /223040263/wanlong/LLM_Retreival/RAG4RE/src/train/train_RE_no_rag_v1/EAE_train_no_retrieval/checkpoint-2-2079/tfmr \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/tacred/llama3_ours_tacred_random_rag_15_doc.json \
                      --topk 15 \
                      --task RE
