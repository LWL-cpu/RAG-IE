python src/main.py --test_data_path data/tacred/test_process.json \
                      --train_data_path data/tacred/train_process.json \
                      --similar_sentences_path data/tacred/test_tacred_similarities.json \
                      --dataset tacred \
                      --prompt_type simple \
                      --model_name gpt4 \
                      --responses_path outputs/tacred/gpt4_tacred_no_rag.json \
                      --task RE


python src/main.py --test_data_path data/tacred/test_process.json \
                      --train_data_path data/tacred/train_process.json \
                      --similar_sentences_path data/tacred/test_tacred_similarities.json \
                      --dataset tacred \
                      --prompt_type rag \
                      --model_name gpt4 \
                      --responses_path outputs/tacred/gpt4_tacred_rag_1_doc.json \
                      --topk 1 \
                      --task RE


python src/main.py --test_data_path data/tacred/test_process.json \
                      --train_data_path data/tacred/train_process.json \
                      --similar_sentences_path data/tacred/test_tacred_similarities.json \
                      --dataset tacred \
                      --prompt_type rag \
                      --model_name gpt4 \
                      --responses_path outputs/tacred/gpt4_tacred_rag_5_doc.json \
                      --topk 5 \
                      --task RE

python src/main.py --test_data_path data/tacred/test_process.json \
                      --train_data_path data/tacred/train_process.json \
                      --similar_sentences_path data/tacred/test_tacred_similarities.json \
                      --dataset tacred \
                      --prompt_type rag \
                      --model_name gpt4 \
                      --responses_path outputs/tacred/gpt4_tacred_rag_10_doc.json \
                      --topk 10 \
                      --task RE

python src/main.py --test_data_path data/tacred/test_process.json \
                      --train_data_path data/tacred/train_process.json \
                      --similar_sentences_path data/tacred/test_tacred_similarities.json \
                      --dataset tacred \
                      --prompt_type rag \
                      --model_name gpt4 \
                      --responses_path outputs/tacred/gpt4_tacred_rag_15_doc.json \
                      --topk 15 \
                      --task RE
