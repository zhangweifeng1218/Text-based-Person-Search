from models.model import Network
import torchvision.transforms as transforms
from train_config import parse_args
import torch
import torch.utils.data as data
from data.cuhkpedes import CUHKPEDESDataset_
from PIL import Image
from imageio import imread
import numpy as np
from function import calculate_score
import json
import os
import pandas as pd
from transformers import BertTokenizer
from matplotlib import pyplot as plt
import pickle
import h5py


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SearchEngine():
    def __init__(self):
        self.args = parse_args()
        self.args.processed = False
        self.load_model()
        self.load_dataset()
        self.tokenizer = BertTokenizer.from_pretrained(os.path.join(self.args.bert_model_path, 'bert-base-uncased'))
        self.transform = transforms.Compose([
            transforms.Resize((self.args.height, self.args.width), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def load_model(self, file='./log/best.pth.tar'):
        if torch.cuda.is_available():
            self.network = Network(self.args).cuda()
            self.network.load_state_dict(torch.load(file)['state_dict'])
        else:
            self.network = Network(self.args)
            self.network.load_state_dict(torch.load(file, map_location='cpu')['state_dict'])
        self.network.eval()

    def load_dataset(self, split='test'):
        dataset = CUHKPEDESDataset_(dir, split, self.args.max_length)
        self.data_loader = data.DataLoader(dataset, 1, shuffle=False)
        self.csv = pd.read_csv('./datasets/processed/{}.csv'.format(split))
        self.test_img_fea_h5 = h5py.File(os.path.join('./datasets/processed/', 'test_img_feature.h5'), 'r')
        self.test_img_feas = self.test_img_fea_h5['image_fea']
        self.test_img_ids = self.test_img_fea_h5['id']
        self.test_img_pathes = self.test_img_fea_h5['image_path']



    def process_image(self, img_path):
        img = imread(img_path)
        if len(img.shape) == 2:
            img = np.dstack((img, img, img))
        img = Image.fromarray(img)
        img = self.transform(img)
        return img

    def tokenize(self, x):
        x = self.tokenizer(x, return_tensors='pt',
                           add_special_tokens=True,
                           padding=True)
        if x['input_ids'].size()[-1] < self.args.max_length:
            padded = torch.zeros(size=(x['input_ids'].size()[0], self.args.max_length - x['input_ids'].size(-1)),
                                 dtype=x['input_ids'].dtype).to(x['input_ids'].device)
            inputs = torch.cat([x['input_ids'], padded], dim=-1)
            attention_mask = torch.cat([x['attention_mask'], padded], dim=-1)
        else:
            inputs = x['input_ids'][:self.args.max_length]
            attention_mask = x['attention_mask'][:self.args.max_length]
        x = inputs
        return x, attention_mask

    def process_text(self, text):
        text, mask = self.tokenize(text)
        return text, mask

    def generate_all_test_img_feature(self):
        with torch.no_grad():
            img_pathes = []
            img_ids = []
            img_feas = []
            i = 0
            for image_path, captions, labels, mask in self.data_loader:
                i += 1
                if i % 2 == 0:
                    continue
                print(image_path[0])
                images = self.process_image(os.path.join('./datasets/', image_path[0]))
                images = images.to(device).unsqueeze(0)
                img_fea = self.network.get_img_feature(images)
                img_pathes.append(image_path[0])
                img_ids.append(labels.cpu().numpy())
                img_feas.append(img_fea.cpu().numpy())
            f = h5py.File(os.path.join('./datasets/processed/', 'test_img_feature.h5'), 'w')
            f['id'] = img_ids
            f['image_path'] = img_pathes
            f['image_fea'] = img_feas
            f.close()

    def search(self, query_text, return_k=20):
        # text 输入的检测词
        # return_k: 展示的图片数
        result = []
        text, attention_mask = self.process_text(query_text)
        text = text.to(device)
        attention_mask = attention_mask.to(device)
        with torch.no_grad():
            txt_fea = self.network.get_text_feature(text, attention_mask)
            img_feas = self.test_img_feas
            img_pathes = self.test_img_pathes
            img_ids = self.test_img_ids
            for i in range(len(img_feas)):
                img_fea = torch.from_numpy(img_feas[i]).to(device)
                score = calculate_score(img_fea, txt_fea).cpu().numpy()
                result.append({'img_path': img_pathes[i], 'score': score, 'id': img_ids[i]})

        result = sorted(result, key=lambda e: e.__getitem__('score'), reverse=True)[:return_k]

        return result

    def search1(self, query_text, return_k = 20):# 太慢
        # text 输入的检测词
        # return_k: 展示的图片数
        result = []
        text, attention_mask = self.process_text(query_text)
        text = text.to(device)
        attention_mask = attention_mask.to(device)
        with torch.no_grad():
            for image_path, captions, labels, mask in self.data_loader:
                print(image_path[0])
                images = self.process_image(os.path.join('./datasets/',image_path[0]))
                images = images.to(device).unsqueeze(0)
                img_fea, txt_fea = self.network(images, text, attention_mask)

                score = calculate_score(img_fea, txt_fea).cpu().numpy()
                id = labels.cpu().numpy()
                caption = self.csv[self.csv['images_path']==image_path[0][16:]]
                caption = caption.captions.tolist()
                result.append({'img_path':image_path, 'score': score, 'id': id, 'captions': caption})

        result = sorted(result, key=lambda e: e.__getitem__('score'), reverse=True)[:return_k]

        return result

    def search_(self, query_text, return_k=20):
        text, attention_mask = self.process_text(query_text)
        text = text.to(device)
        attention_mask = attention_mask.to(device)
        with torch.no_grad():
            txt_fea = self.network.get_text_feature(text, attention_mask)
            image = self.process_image(os.path.join('./datasets/', 'CUHK-PEDES/imgs/train_query/p8848_s17661.jpg'))
            image = image.to(device).unsqueeze(0)
            img_fea = self.network.get_img_feature(image)
            score = calculate_score(img_fea, txt_fea).cpu().numpy()
            print(score)

    def show_result(self, result):
        img_num = len(result)
        plt.figure(figsize=(20,20))
        for i in range(img_num):
            plt.subplot(1, img_num, i+1)
            img = imread(os.path.join('./datasets', result[i]['img_path'].decode()))
            img = Image.fromarray(img)
            img = img.resize((self.args.width, self.args.height), Image.ANTIALIAS)
            plt.axis('off')
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img)
            print('img: {}, score:{}, id: {}'.format(result[i]['img_path'].decode(), result[i]['score'], result[i]['id']))
        plt.show()
        plt.savefig('1.pdf')




if __name__=='__main__':
    engine = SearchEngine()
    query = ['The woman is wearing a black jacket with some white trim, long denim jeans, and is holding a grey handbag. She is also wearing black shoes.']
    result = engine.search(query)
    engine.show_result(result)
    print('test ok')








