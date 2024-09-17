python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/rams/test.jsonlines \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/rams/train.jsonlines \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/rams/test_rams_similarities.json \
                      --dataset rams \
                      --prompt_type simple \
                      --model_name /sds_wangby/models/Llama-2-13b-hf/ \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/rams/llama2_rams_no_rag.json


python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/rams/test.jsonlines \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/rams/train.jsonlines \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/rams/test_rams_similarities.json \
                      --dataset rams \
                      --prompt_type rag \
                      --model_name /sds_wangby/models/Llama-2-13b-hf/ \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/rams/llama2_rams_rag_1_doc.json \
                      --topk 1

python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/rams/test.jsonlines \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/rams/train.jsonlines \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/rams/test_rams_similarities.json \
                      --dataset rams \
                      --prompt_type rag \
                      --model_name /sds_wangby/models/Llama-2-13b-hf/ \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/rams/llama2_rams_rag_3_doc.json \
                      --topk 3

python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/rams/test.jsonlines \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/rams/train.jsonlines \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/rams/test_rams_similarities.json \
                      --dataset rams \
                      --prompt_type rag \
                      --model_name /sds_wangby/models/Llama-2-13b-hf/ \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/rams/llama2_rams_rag_5_doc.json \
                      --topk 5

python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/rams/test.jsonlines \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/rams/train.jsonlines \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/rams/test_rams_similarities.json \
                      --dataset rams \
                      --prompt_type rag \
                      --model_name /sds_wangby/models/Llama-2-13b-hf/ \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/rams/llama2_rams_rag_10_doc.json \
                      --topk 10

python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/rams/test.jsonlines \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/rams/train.jsonlines \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/rams/test_rams_similarities.json \
                      --dataset rams \
                      --prompt_type rag \
                      --model_name /sds_wangby/models/Llama-2-13b-hf/ \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/rams/llama2_rams_rag_15_doc.json \
                      --topk 15
python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/rams/test.jsonlines \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/rams/train.jsonlines \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/rams/test_rams_similarities.json \
                      --dataset rams \
                      --prompt_type rag \
                      --model_name /sds_wangby/models/Llama-2-13b-hf/ \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/rams/llama2_rams_rag_20_doc.json \
                      --topk 20
