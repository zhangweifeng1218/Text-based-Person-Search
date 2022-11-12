import pandas as pd
import transformers as ppb
import pickle
import numpy as np
import json
import os
import transformers as ppb

def makedir(root):
    if not os.path.exists(root):
        os.makedirs(root)


def write_json(data, root):
    with open(root, 'w') as f:
        json.dump(data, f)

def gen_train_val_test_split(raw_file='../datasets/reid_raw.json', out_dir='../datasets/processed/'):
    print('process reid_raw.json...')
    with open(raw_file,'r') as f:
        imgs = json.load(f)

    val_data = []
    train_data = []
    test_data = []
    train_ids = []
    train_pathes = []
    train_captions = []
    val_ids = []
    val_pathes = []
    val_captions = []
    test_ids = []
    test_pathes = []
    test_captions = []
    for img in imgs:
        if img['split'] == 'train':
            train_data.append(img)
            for i in range(len(img['captions'])):
                train_ids.append(img['id'])
                train_pathes.append(img['file_path'])
                train_captions.append(img['captions'][i])
        elif img['split'] =='val':
            val_data.append(img)
            for i in range(len(img['captions'])):
                val_ids.append(img['id'])
                val_pathes.append(img['file_path'])
                val_captions.append(img['captions'][i])
        else:
            test_data.append(img)
            for i in range(len(img['captions'])):
                test_ids.append(img['id'])
                test_pathes.append(img['file_path'])
                test_captions.append(img['captions'][i])
    write_json(train_data, os.path.join(out_dir, 'train_reid.json'))
    write_json(val_data, os.path.join(out_dir, 'val_reid.json'))
    write_json(test_data, os.path.join(out_dir, 'test_reid.json'))

    # also generate csv files
    train_data_frame = pd.DataFrame({'id':train_ids, 'images_path': train_pathes, 'captions': train_captions})
    train_data_frame.to_csv(os.path.join(out_dir, 'train.csv'), index=False)

    val_data_frame = pd.DataFrame({'id': val_ids, 'images_path': val_pathes, 'captions': val_captions})
    val_data_frame.to_csv(os.path.join(out_dir, 'val.csv'), index=False)

    test_data_frame = pd.DataFrame({'id': test_ids, 'images_path': test_pathes, 'captions': test_captions})
    test_data_frame.to_csv(os.path.join(out_dir, 'test.csv'), index=False)

    print('{} train samples, {} val samples, {} test samples generated!'.format(len(train_ids), len(val_ids), len(test_ids)))

    return [train_data, val_data, test_data]

def gen_BERT_token(txt_path, save_path):
    # txt_path = r'../data/test.csv'
    # save_path = r'../data/BERT_encode/BERT_id_train_64_new.npz'
    print('generate BERT token for {}...'.format(txt_path))
    csv_data = pd.read_csv(txt_path, error_bad_lines=False)
    dataset = csv_data['captions']

    model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

    # Load pretrained model/tokenizer
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    tokenized = dataset.apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

    max_len = 64
    padded = []
    for i in tokenized.values:
        if len(i) < max_len:
            i += [0] * (max_len - len(i))
        else:
            i = i[:max_len]
        padded.append(i)
    padded = np.array(padded)

    print(padded.shape)  # shape;[68108,max_len]
    attention_mask = np.where(padded != 0, 1, 0)
    print(attention_mask)
    print(padded)
    print(attention_mask.shape)
    print(padded.shape)
    print(csv_data['id'].shape)
    print(csv_data['images_path'].shape)
    dict = {'caption_id': padded, 'attention_mask': attention_mask, 'images_path': csv_data['images_path'], 'labels': csv_data['id']}
    with open(save_path, 'wb') as f:
        pickle.dump(dict, f)

def main():
    raw_file = '../datasets/reid_raw.json'
    out_dir = '../datasets/processed/'
    makedir(out_dir)
    gen_train_val_test_split(raw_file, out_dir)
    gen_BERT_token(os.path.join(out_dir, 'train.csv'), os.path.join(out_dir, 'BERT_id_train_64_new.npz'))
    gen_BERT_token(os.path.join(out_dir, 'val.csv'), os.path.join(out_dir, 'BERT_id_val_64_new.npz'))
    gen_BERT_token(os.path.join(out_dir, 'test.csv'), os.path.join(out_dir, 'BERT_id_test_64_new.npz'))

if __name__=='__main__':
    main()


