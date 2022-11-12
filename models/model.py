from torchvision import models
from models.CNN_text import ResNet_text_50
import transformers as ppb
from torch.nn import init
import torch.nn as nn
import torch
import os
import torch.nn.functional as F
from models.relation import RN

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


class ResNet_image_50(nn.Module):
    def __init__(self):
        super(ResNet_image_50, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        resnet50.layer4[0].downsample[0].stride = (1, 1)
        resnet50.layer4[0].conv2.stride = (1, 1)
        self.base1 = nn.Sequential(
            resnet50.conv1,
            resnet50.bn1,
            resnet50.relu,
            resnet50.maxpool,
            resnet50.layer1,  # 256 64 32
        )
        self.base2 = nn.Sequential(
            resnet50.layer2,  # 512 32 16
        )
        self.base3 = nn.Sequential(
            resnet50.layer3,  # 1024 16 8
        )
        self.base4 = nn.Sequential(
            resnet50.layer4  # 2048 16 8
        )

    def forward(self, x):
        x1 = self.base1(x)
        x2 = self.base2(x1)
        x3 = self.base3(x2)
        x4 = self.base4(x3)
        return x1, x2, x3, x4


class Network(nn.Module):
    def __init__(self, args):
        super(Network, self).__init__()

        self.model_img = ResNet_image_50()
        self.model_txt = ResNet_text_50(args)

        if args.embedding_type == 'BERT':
            self.text_embed = ppb.BertModel.from_pretrained(os.path.join(args.bert_model_path, 'bert-base-uncased'))
            self.text_embed.eval()
            self.BERT = True
            for p in self.text_embed.parameters():
                p.requires_grad = False

        self.vision_rn = RN(args.feature_size, 512, 64)
        self.text_rn = RN(args.feature_size, 512, 64)
        self.ID_predictor = nn.Linear(args.feature_size, args.num_classes)

        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, img, txt, mask):
        with torch.no_grad():
            txt = self.text_embed(txt, attention_mask=mask)
            txt = txt[0]
            txt = txt.unsqueeze(1)
            txt = txt.permute(0, 3, 1, 2)

        _, _, img3, img4 = self.model_img(img)  # img4: batch x 2048 x 24 x 8
        img_f3 = self.max_pool(img3).squeeze(dim=-1).squeeze(dim=-1)
        img_f41 = self.max_pool(img4[:, :, 0:4, :]).squeeze(dim=-1).squeeze(dim=-1)
        img_f42 = self.max_pool(img4[:, :, 4:8, :]).squeeze(dim=-1).squeeze(dim=-1)
        img_f43 = self.max_pool(img4[:, :, 8:12, :]).squeeze(dim=-1).squeeze(dim=-1)
        img_f44 = self.max_pool(img4[:, :, 12:16, :]).squeeze(dim=-1).squeeze(dim=-1)
        img_f45 = self.max_pool(img4[:, :, 16:20, :]).squeeze(dim=-1).squeeze(dim=-1)
        img_f46 = self.max_pool(img4[:, :, 20:, :]).squeeze(dim=-1).squeeze(dim=-1)
        img_f4 = self.max_pool(img4).squeeze(dim=-1).squeeze(dim=-1)

        txt3, txt41, txt42, txt43, txt44, txt45, txt46 = self.model_txt(txt)  # txt4: batch x 2048 x 1 x 64
        txt_f3 = self.max_pool(txt3).squeeze(dim=-1).squeeze(dim=-1)
        txt_f41 = self.max_pool(txt41)
        txt_f42 = self.max_pool(txt42)
        txt_f43 = self.max_pool(txt43)
        txt_f44 = self.max_pool(txt44)
        txt_f45 = self.max_pool(txt45)
        txt_f46 = self.max_pool(txt46)
        txt_f4 = self.max_pool(torch.cat([txt_f41, txt_f42, txt_f43, txt_f44, txt_f45, txt_f46], dim=2)).squeeze(dim=-1).squeeze(dim=-1)
        txt_f41 = txt_f41.squeeze(dim=-1).squeeze(dim=-1)
        txt_f42 = txt_f42.squeeze(dim=-1).squeeze(dim=-1)
        txt_f43 = txt_f43.squeeze(dim=-1).squeeze(dim=-1)
        txt_f44 = txt_f44.squeeze(dim=-1).squeeze(dim=-1)
        txt_f45 = txt_f45.squeeze(dim=-1).squeeze(dim=-1)
        txt_f46 = txt_f46.squeeze(dim=-1).squeeze(dim=-1)

        v_parts = torch.stack([img_f41, img_f42, img_f43, img_f44, img_f45, img_f46], dim=1)
        t_parts = torch.stack([txt_f41, txt_f42, txt_f43, txt_f44, txt_f45, txt_f46], dim=1)
        v_rn, _ = self.vision_rn(v_parts)
        t_rn, _ = self.text_rn(t_parts)
        v_id = self.ID_predictor(img_f4)
        t_id = self.ID_predictor(txt_f4)

        if self.training:
            return img_f3, img_f4, img_f41, img_f42, img_f43, img_f44, img_f45, img_f46, \
                   txt_f3, txt_f4, txt_f41, txt_f42, txt_f43, txt_f44, txt_f45, txt_f46, v_rn, t_rn, v_id, t_id
        else:
            return img_f4, txt_f4

    def get_text_feature(self, txt, mask):
        with torch.no_grad():
            txt = self.text_embed(txt, attention_mask=mask)
            txt = txt[0]
            txt = txt.unsqueeze(1)
            txt = txt.permute(0, 3, 1, 2)

        txt3, txt41, txt42, txt43, txt44, txt45, txt46 = self.model_txt(txt)  # txt4: batch x 2048 x 1 x 64
        txt_f3 = self.max_pool(txt3).squeeze(dim=-1).squeeze(dim=-1)
        txt_f41 = self.max_pool(txt41)
        txt_f42 = self.max_pool(txt42)
        txt_f43 = self.max_pool(txt43)
        txt_f44 = self.max_pool(txt44)
        txt_f45 = self.max_pool(txt45)
        txt_f46 = self.max_pool(txt46)
        txt_f4 = self.max_pool(torch.cat([txt_f41, txt_f42, txt_f43, txt_f44, txt_f45, txt_f46], dim=2)).squeeze(
            dim=-1).squeeze(dim=-1)
        return txt_f4

    def get_img_feature(self, img):
        _, _, img3, img4 = self.model_img(img)
        img_f4 = self.max_pool(img4).squeeze(dim=-1).squeeze(dim=-1)
        return img_f4



if __name__ == '__main__':
    from train_config import parse_args
    from function import count_parameters
    import time
    args = parse_args()
    net = Network(args)
    net.eval()
    print(count_parameters(net))
    img = torch.randn(size=(2, 3, args.height, args.width))
    txt = torch.randint(0, 100, size=(2, args.max_length))
    mask = torch.randint(0, 2, size=(2, args.max_length))
    start = time.time()
    ret = net(img, txt, mask)
    Epoch_time = time.time() - start
    print(Epoch_time)
    print('test Network ok!')
