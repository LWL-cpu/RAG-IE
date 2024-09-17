from src.data_augmentation.embeddings.sentence_embeddings import *
train_path = './data/rams/train.jsonlines'
dev_path = './data/rams/dev.jsonlines'
test_path = './data/rams/test.jsonlines'

train_examples = process_instruction(train_path, train_path.split('/')[2])
dev_examples = process_instruction(dev_path, dev_path.split('/')[2])
test_examples = process_instruction(test_path, test_path.split('/')[2])

train_path_out = './data/rams/train.json'
dev_path_out = './data/rams/dev.json'
test_path_out = './data/rams/test.json'
def write_json(path, data):
    """ Write json file"""

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    with open(path, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

write_json(train_path_out, train_examples)
write_json(dev_path_out, dev_examples)
write_json(test_path_out, test_examples)