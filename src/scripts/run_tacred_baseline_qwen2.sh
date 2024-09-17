python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/tacred/test_process.json \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/tacred/train_process.json \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/tacred/test_tacred_similarities.json \
                      --dataset tacred \
                      --prompt_type simple \
                      --model_name /sds_wangby/models/Qwen2-7B/ \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/tacred/llama3_tacred_no_rag.json \
                      --task RE


python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/tacred/test_process.json \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/tacred/train_process.json \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/tacred/test_tacred_similarities.json \
                      --dataset tacred \
                      --prompt_type rag \
                      --model_name /sds_wangby/models/Qwen2-7B/ \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/tacred/llama3_tacred_rag_1_doc.json \
                      --topk 1 \
                      --task RE

python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/tacred/test_process.json \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/tacred/train_process.json \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/tacred/test_tacred_similarities.json \
                      --dataset tacred \
                      --prompt_type rag \
                      --model_name /sds_wangby/models/Qwen2-7B/ \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/tacred/llama3_tacred_rag_3_doc.json \
                      --topk 3 \
                      --task RE

python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/tacred/test_process.json \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/tacred/train_process.json \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/tacred/test_tacred_similarities.json \
                      --dataset tacred \
                      --prompt_type rag \
                      --model_name /sds_wangby/models/Qwen2-7B/ \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/tacred/llama3_tacred_rag_5_doc.json \
                      --topk 5 \
                      --task RE

python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/tacred/test_process.json \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/tacred/train_process.json \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/tacred/test_tacred_similarities.json \
                      --dataset tacred \
                      --prompt_type rag \
                      --model_name /sds_wangby/models/Qwen2-7B/ \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/tacred/llama3_tacred_rag_10_doc.json \
                      --topk 10 \
                      --task RE

python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/tacred/test_process.json \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/tacred/train_process.json \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/tacred/test_tacred_similarities.json \
                      --dataset tacred \
                      --prompt_type rag \
                      --model_name /sds_wangby/models/Qwen2-7B/ \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/tacred/llama3_tacred_rag_15_doc.json \
                      --topk 15 \
                      --task RE
