import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.io import read_image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import PIL
import gc
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import save_image

from model import NET
from data import CustomDataset
from train import Trainer
from imshow import batchshow
from imshow import imshow

## Setting ## 
annotations_file = '../../dataset/GOPRO_Large/train/label.csv'
img_dir = '../../dataset/GOPRO_Large/train'
annotations_file_test = '../../dataset/GOPRO_Large/test/label.csv'
img_dir_test = '../../dataset/GOPRO_Large/test'
savepath1 = './weight/Deblur1'
savepath2 = './weight/Deblur2'
savepath3 = './weight/Deblur3'
batch=4
epoch=200
img_size=511
transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
resize = transforms.Resize((512, 512))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_data = CustomDataset(annotations_file, img_dir, transform = transform)
train_loader = DataLoader(train_data, batch_size=batch, shuffle=True, drop_last=True)
test_data = CustomDataset(annotations_file_test, img_dir_test, transform = transform)
test_loader = DataLoader(train_data, batch_size=1, shuffle=False, drop_last=True)
criterion = nn.MSELoss().to(device)
net1 = NET().to(device)
net1.load_state_dict(torch.load(savepath1))
net2 = NET().to(device)
net2.load_state_dict(torch.load(savepath2))
net3 = NET().to(device)
net3.load_state_dict(torch.load(savepath3))
optimizer1=optim.Adam(net1.parameters(),lr=0.0001)
optimizer2=optim.Adam(net2.parameters(),lr=0.0001)
optimizer3=optim.Adam(net3.parameters(),lr=0.0001)

## Start main ## 
print('Device:', device)
torch.cuda.empty_cache()
gc.collect()

## Output Generation ##
print("Start Converting")
for i in range(30):
    with torch.no_grad():
        i=i+1
        inputs = PIL.Image.open('TestSamples/'+str(i).zfill(2)+'.png')
        inputs = transform(inputs).to(device)
        inputs = inputs.view(1,3,img_size,img_size).to(device)  
        
        outputs = net1(inputs) + inputs
        outputs = net2(outputs) + outputs
        outputs = net2(outputs) + outputs
        
        outputs = outputs[0]
        outputs = outputs/2 + 0.5
        resize(outputs)
        save_image(outputs,'results/'+str(i).zfill(2)+'.png')

## Training ##
trainer = Trainer(
    save_dir1 = savepath1, save_dir2 = savepath2, save_dir3 = savepath3,
    model1 = net1, model2 = net2, model3 = net3 ,
    optimizer1 = optimizer1, optimizer2 = optimizer2, optimizer3 = optimizer3, criterion = criterion,
    train_loader = train_loader, test_loader = test_loader, device = device)
trainer.train(epoch)
