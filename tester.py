from __future__ import print_function
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from warpctc_pytorch import CTCLoss
import os
import utils
import dataset
import cv2
import models.crnn as crnn
from captcha.config import cfg

parser = argparse.ArgumentParser()
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image to network')
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--crnn', default='expr/netCRNN_200.pth', help="path to crnn (to continue training)")
parser.add_argument('--alphabet', type=str, default=cfg.labels)
opt = parser.parse_args()


cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

nclass = len(opt.alphabet) + 1
nc = 3

converter = utils.strLabelConverter(opt.alphabet)
criterion = CTCLoss()

crnn = crnn.CRNN(opt.imgH, nc, nclass, opt.nh)
print('loading pretrained model from %s' % opt.crnn)
old = torch.load(opt.crnn)
new_state_dict = {}
for k, v in old.items():
    new_state_dict[k[7:]] = v

crnn.load_state_dict(new_state_dict)

for p in crnn.parameters():
    p.requires_grad = False

if opt.cuda:
    crnn.cuda()
    crnn = torch.nn.DataParallel(crnn, device_ids=range(opt.ngpu))

def bytes2cv2(b):
    nparr = np.fromstring(b, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img_np


def run(tensor):
    tensor = tensor.view(-1, *tensor.shape)
    preds = crnn(tensor)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * 1))

    _, preds = preds.max(2)

    preds = preds.view(-1)
    sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
    #raw_preds = converter.decode(preds.data, preds_size.data, raw=True)
    return sim_preds

rn = dataset.resizeNormalize((100, 32))
def init(images):
    return rn(images)


import requests, time

def web_tester(url):
    img = requests.get(url).content
    name = '{}.jpg'.format(time.time())
    with open('static/' + name, 'wb') as f:
        f.write(img)
    image = bytes2cv2(img)
    tensor = init(image)
    return name, run(tensor)
