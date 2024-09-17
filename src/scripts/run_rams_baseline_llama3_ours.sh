python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/rams/test.jsonlines \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/rams/train.jsonlines \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/rams/test_rams_similarities.json \
                      --dataset rams \
                      --prompt_type simple \
                      --model_name /223040263/wanlong/LLM_Retreival/RAG4RE/src/train/train_EE_rams/EAE_train_no_retrieval/checkpoint-2-1830/tfmr \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/rams/llama3_rams_no_rag.json \
                      --task EE


python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/rams/test.jsonlines \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/rams/train.jsonlines \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/rams/test_rams_similarities.json \
                      --dataset rams \
                      --prompt_type rag \
                      --model_name /223040263/wanlong/LLM_Retreival/RAG4RE/src/train/train_EE_rams/EAE_train_no_retrieval/checkpoint-2-1830/tfmr \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/rams/llama3_rams_rag_1_doc.json \
                      --topk 1 \
                      --task EE

python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/rams/test.jsonlines \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/rams/train.jsonlines \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/rams/test_rams_similarities.json \
                      --dataset rams \
                      --prompt_type rag \
                      --model_name /223040263/wanlong/LLM_Retreival/RAG4RE/src/train/train_EE_rams/EAE_train_no_retrieval/checkpoint-2-1830/tfmr \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/rams/llama3_rams_rag_3_doc.json \
                      --topk 3 \
                      --task EE

python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/rams/test.jsonlines \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/rams/train.jsonlines \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/rams/test_rams_similarities.json \
                      --dataset rams \
                      --prompt_type rag \
                      --model_name /223040263/wanlong/LLM_Retreival/RAG4RE/src/train/train_EE_rams/EAE_train_no_retrieval/checkpoint-2-1830/tfmr \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/rams/llama3_rams_rag_5_doc.json \
                      --topk 5 \
                      --task EE

python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/rams/test.jsonlines \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/rams/train.jsonlines \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/rams/test_rams_similarities.json \
                      --dataset rams \
                      --prompt_type rag \
                      --model_name /223040263/wanlong/LLM_Retreival/RAG4RE/src/train/train_EE_rams/EAE_train_no_retrieval/checkpoint-2-1830/tfmr \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/rams/llama3_rams_rag_10_doc.json \
                      --topk 10 \
                      --task EE

python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/rams/test.jsonlines \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/rams/train.jsonlines \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/rams/test_rams_similarities.json \
                      --dataset rams \
                      --prompt_type rag \
                      --model_name /223040263/wanlong/LLM_Retreival/RAG4RE/src/train/train_EE_rams/EAE_train_no_retrieval/checkpoint-2-1830/tfmr \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/rams/llama3_rams_rag_15_doc.json \
                      --topk 15 \
                      --task EE

