python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/WikiEvent/test.jsonl \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/WikiEvent/train.jsonl \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/WikiEvent/test_wiki_similarities.json \
                      --dataset wiki \
                      --prompt_type simple \
                      --model_name gpt3.5 \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/wiki/gpt3.5_wiki_no_rag.json


python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/WikiEvent/test.jsonl \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/WikiEvent/train.jsonl \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/WikiEvent/test_wiki_similarities.json \
                      --dataset wiki \
                      --prompt_type rag \
                      --model_name gpt3.5 \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/wiki/gpt3.5_wiki_rag_1_doc.json \
                      --topk 1


python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/WikiEvent/test.jsonl \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/WikiEvent/train.jsonl \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/WikiEvent/test_wiki_similarities.json \
                      --dataset wiki \
                      --prompt_type rag \
                      --model_name gpt3.5 \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/wiki/gpt3.5_wiki_rag_3_doc.json \
                      --topk 3



python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/WikiEvent/test.jsonl \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/WikiEvent/train.jsonl \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/WikiEvent/test_wiki_similarities.json \
                      --dataset wiki \
                      --prompt_type rag \
                      --model_name gpt3.5 \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/wiki/gpt3.5_wiki_rag_5_doc.json \
                      --topk 5


python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/WikiEvent/test.jsonl \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/WikiEvent/train.jsonl \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/WikiEvent/test_wiki_similarities.json \
                      --dataset wiki \
                      --prompt_type rag \
                      --model_name gpt3.5 \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/wiki/gpt3.5_wiki_rag_10_doc.json \
                      --topk 10


python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/WikiEvent/test.jsonl \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/WikiEvent/train.jsonl \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/WikiEvent/test_wiki_similarities.json \
                      --dataset wiki \
                      --prompt_type rag \
                      --model_name gpt3.5 \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/wiki/gpt3.5_wiki_rag_15_doc.json \
                      --topk 15

python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/WikiEvent/test.jsonl \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/WikiEvent/train.jsonl \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/WikiEvent/test_wiki_similarities.json \
                      --dataset wiki \
                      --prompt_type rag \
                      --model_name gpt3.5 \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/wiki/gpt3.5_wiki_rag_20_doc.json \
                      --topk 20
                                            
python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/ace_eeqa/test_convert.json \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/ace_eeqa/train_convert.json \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/ace_eeqa/test_ace_similarities.json \
                      --dataset ace2005 \
                      --prompt_type simple \
                      --model_name gpt3.5 \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/ace2005/gpt3.5_ace2005_no_rag.json


python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/ace_eeqa/test_convert.json \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/ace_eeqa/train_convert.json \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/ace_eeqa/test_ace_similarities.json \
                      --dataset ace2005 \
                      --prompt_type rag \
                      --model_name gpt3.5 \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/ace2005/gpt3.5_ace2005_rag_1_doc.json \
                      --topk 1

python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/ace_eeqa/test_convert.json \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/ace_eeqa/train_convert.json \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/ace_eeqa/test_ace_similarities.json \
                      --dataset ace2005 \
                      --prompt_type rag \
                      --model_name gpt3.5 \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/ace2005/gpt3.5_ace2005_rag_3_doc.json \
                      --topk 3

python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/ace_eeqa/test_convert.json \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/ace_eeqa/train_convert.json \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/ace_eeqa/test_ace_similarities.json \
                      --dataset ace2005 \
                      --prompt_type rag \
                      --model_name gpt3.5 \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/ace2005/gpt3.5_ace2005_rag_5_doc.json \
                      --topk 5

python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/ace_eeqa/test_convert.json \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/ace_eeqa/train_convert.json \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/ace_eeqa/test_ace_similarities.json \
                      --dataset ace2005 \
                      --prompt_type rag \
                      --model_name gpt3.5 \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/ace2005/gpt3.5_ace2005_rag_10_doc.json \
                      --topk 10

python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/ace_eeqa/test_convert.json \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/ace_eeqa/train_convert.json \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/ace_eeqa/test_ace_similarities.json \
                      --dataset ace2005 \
                      --prompt_type rag \
                      --model_name gpt3.5 \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/ace2005/gpt3.5_ace2005_rag_15_doc.json \
                      --topk 15

python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/ace_eeqa/test_convert.json \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/ace_eeqa/train_convert.json \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/ace_eeqa/test_ace_similarities.json \
                      --dataset ace2005 \
                      --prompt_type rag \
                      --model_name gpt3.5 \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/ace2005/gpt3.5_ace2005_rag_20_doc.json \
                      --topk 20