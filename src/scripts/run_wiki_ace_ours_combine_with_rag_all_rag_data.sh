python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/WikiEvent/test.jsonl \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/WikiEvent/train.jsonl \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/WikiEvent/test_wiki_similarities.json \
                      --dataset wiki \
                      --prompt_type simple \
                      --model_name /223040263/wanlong/LLM_Retreival/RAG4RE/src/train/rag_combined_no_ori/EAE_train_no_retrieval/checkpoint-2-3765/tfmr/ \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/wiki/llama3_wiki_no_rag.json \
                      --task EE


python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/WikiEvent/test.jsonl \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/WikiEvent/train.jsonl \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/WikiEvent/test_wiki_similarities.json \
                      --dataset wiki \
                      --prompt_type rag \
                      --model_name /223040263/wanlong/LLM_Retreival/RAG4RE/src/train/rag_combined_no_ori/EAE_train_no_retrieval/checkpoint-2-3765/tfmr/ \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/wiki/llama3_wiki_rag_1_doc.json \
                      --topk 1 \
                      --task EE


python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/WikiEvent/test.jsonl \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/WikiEvent/train.jsonl \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/WikiEvent/test_wiki_similarities.json \
                      --dataset wiki \
                      --prompt_type rag \
                      --model_name /223040263/wanlong/LLM_Retreival/RAG4RE/src/train/rag_combined_no_ori/EAE_train_no_retrieval/checkpoint-2-3765/tfmr/ \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/wiki/llama3_wiki_rag_3_doc.json \
                      --topk 3 \
                      --task EE



python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/WikiEvent/test.jsonl \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/WikiEvent/train.jsonl \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/WikiEvent/test_wiki_similarities.json \
                      --dataset wiki \
                      --prompt_type rag \
                      --model_name /223040263/wanlong/LLM_Retreival/RAG4RE/src/train/rag_combined_no_ori/EAE_train_no_retrieval/checkpoint-2-3765/tfmr/ \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/wiki/llama3_wiki_rag_5_doc.json \
                      --topk 5 \
                      --task EE


python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/WikiEvent/test.jsonl \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/WikiEvent/train.jsonl \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/WikiEvent/test_wiki_similarities.json \
                      --dataset wiki \
                      --prompt_type rag \
                      --model_name /223040263/wanlong/LLM_Retreival/RAG4RE/src/train/rag_combined_no_ori/EAE_train_no_retrieval/checkpoint-2-3765/tfmr/ \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/wiki/llama3_wiki_rag_10_doc.json \
                      --topk 10 \
                      --task EE


python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/WikiEvent/test.jsonl \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/WikiEvent/train.jsonl \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/WikiEvent/test_wiki_similarities.json \
                      --dataset wiki \
                      --prompt_type rag \
                      --model_name /223040263/wanlong/LLM_Retreival/RAG4RE/src/train/rag_combined_no_ori/EAE_train_no_retrieval/checkpoint-2-3765/tfmr/ \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/wiki/llama3_wiki_rag_15_doc.json \
                      --topk 15 \
                      --task EE

# python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/WikiEvent/test.jsonl \
#                       --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/WikiEvent/train.jsonl \
#                       --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/WikiEvent/test_wiki_similarities.json \
#                       --dataset wiki \
#                       --prompt_type rag \
#                       --model_name /223040263/wanlong/LLM_Retreival/RAG4RE/src/train/rag_combined_no_ori/EAE_train_no_retrieval/checkpoint-2-3765/tfmr/ \
#                       --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/wiki/llama3_wiki_rag_20_doc.json \
#                       --topk 20 \
#                       --task EE
                                            
python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/ace_eeqa/test_convert.json \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/ace_eeqa/train_convert.json \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/ace_eeqa/test_ace_similarities.json \
                      --dataset ace2005 \
                      --prompt_type simple \
                      --model_name /223040263/wanlong/LLM_Retreival/RAG4RE/src/train/rag_combined_no_ori/EAE_train_no_retrieval/checkpoint-2-3765/tfmr/ \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/ace2005/llama3_ace2005_no_rag.json \
                      --task EE


python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/ace_eeqa/test_convert.json \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/ace_eeqa/train_convert.json \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/ace_eeqa/test_ace_similarities.json \
                      --dataset ace2005 \
                      --prompt_type rag \
                      --model_name /223040263/wanlong/LLM_Retreival/RAG4RE/src/train/rag_combined_no_ori/EAE_train_no_retrieval/checkpoint-2-3765/tfmr/ \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/ace2005/llama3_ace2005_rag_1_doc.json \
                      --topk 1 \
                      --task EE

python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/ace_eeqa/test_convert.json \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/ace_eeqa/train_convert.json \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/ace_eeqa/test_ace_similarities.json \
                      --dataset ace2005 \
                      --prompt_type rag \
                      --model_name /223040263/wanlong/LLM_Retreival/RAG4RE/src/train/rag_combined_no_ori/EAE_train_no_retrieval/checkpoint-2-3765/tfmr/ \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/ace2005/llama3_ace2005_rag_3_doc.json \
                      --topk 3 \
                      --task EE

python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/ace_eeqa/test_convert.json \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/ace_eeqa/train_convert.json \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/ace_eeqa/test_ace_similarities.json \
                      --dataset ace2005 \
                      --prompt_type rag \
                      --model_name /223040263/wanlong/LLM_Retreival/RAG4RE/src/train/rag_combined_no_ori/EAE_train_no_retrieval/checkpoint-2-3765/tfmr/ \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/ace2005/llama3_ace2005_rag_5_doc.json \
                      --topk 5 \
                      --task EE

python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/ace_eeqa/test_convert.json \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/ace_eeqa/train_convert.json \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/ace_eeqa/test_ace_similarities.json \
                      --dataset ace2005 \
                      --prompt_type rag \
                      --model_name /223040263/wanlong/LLM_Retreival/RAG4RE/src/train/rag_combined_no_ori/EAE_train_no_retrieval/checkpoint-2-3765/tfmr/ \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/ace2005/llama3_ace2005_rag_10_doc.json \
                      --topk 10 \
                      --task EE

python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/ace_eeqa/test_convert.json \
                      --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/ace_eeqa/train_convert.json \
                      --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/ace_eeqa/test_ace_similarities.json \
                      --dataset ace2005 \
                      --prompt_type rag \
                      --model_name /223040263/wanlong/LLM_Retreival/RAG4RE/src/train/rag_combined_no_ori/EAE_train_no_retrieval/checkpoint-2-3765/tfmr/ \
                      --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/ace2005/llama3_ace2005_rag_15_doc.json \
                      --topk 15 \
                      --task EE

# python main.py --test_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/ace_eeqa/test_convert.json \
#                       --train_data_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/ace_eeqa/train_convert.json \
#                       --similar_sentences_path /223040263/wanlong/LLM_Retreival/RAG4RE/data/ace_eeqa/test_ace_similarities.json \
#                       --dataset ace2005 \
#                       --prompt_type rag \
#                       --model_name /223040263/wanlong/LLM_Retreival/RAG4RE/src/train/rag_combined_no_ori/EAE_train_no_retrieval/checkpoint-2-3765/tfmr/ \
#                       --responses_path /223040263/wanlong/LLM_Retreival/RAG4RE/outputs/ace2005/llama3_ace2005_rag_20_doc.json \
#                       --topk 20 \
#                       --task EE