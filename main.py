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

annotations_file = '../../dataset/GOPRO_Large/train/label.csv'
img_dir = '../../dataset/GOPRO_Large/train'
annotations_file_test = '../../dataset/GOPRO_Large/test/label.csv'
img_dir_test = '../../dataset/GOPRO_Large/test'
savepath1 = './weight/Deblur1'
savepath2 = './weight/Deblur2'
savepath3 = './weight/Deblur3'
batch=4
epoch=300
img_size=511
transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

train_data = CustomDataset(annotations_file, img_dir, transform = transform)
train_loader = DataLoader(train_data, batch_size = batch, shuffle=True, drop_last=True)
test_data = CustomDataset(annotations_file_test, img_dir_test, transform = transform)
test_loader = DataLoader(train_data, batch_size = batch, shuffle=True, drop_last=True)

criterion = nn.MSELoss().to(device)
net1 = NET().to(device)
net1.load_state_dict(torch.load(savepath1))
net2 = NET().to(device)
net2.load_state_dict(torch.load(savepath2))
net3 = NET().to(device)
net3.load_state_dict(torch.load(savepath3))
optimizer1=optim.Adam(net1.parameters(),lr=0.001)
optimizer2=optim.Adam(net2.parameters(),lr=0.001)
optimizer3=optim.Adam(net3.parameters(),lr=0.001)

## Start main ## 
torch.cuda.empty_cache()
gc.collect()

## Training ##
trainer = Trainer(save_dir1=savepath1,save_dir2=savepath2,save_dir3=savepath3,model1=net1,model2=net2,model3=net3,optimizer1=optimizer1,optimizer2=optimizer2,optimizer3=optimizer3,criterion=criterion,train_loader=train_loader,test_loader=test_loader,device=device)
trainer.train(epoch)

## Output Generation ##
for i in range(29):
    with torch.no_grad():
        i=i+1
        inputs = PIL.Image.open('TestSamples/'+str(i).zfill(2)+'.png')
        inputs = transform(inputs).to(device)
        inputs = inputs.view(1,3,img_size,img_size).to(device)
        
        combine1 = inputs.clone()
        outputs = net1(inputs)
        combine1[0:,0] = outputs[0:,0] + inputs[0:,0]
        
        combine2 = combine1.clone()
        outputs = net2(combine1)
        combine2[0:,1] = outputs[0:,0] + combine1[0:,1]
        
        combine3 = combine2.clone()
        outputs = net3(combine2)
        combine3[0:,2] = outputs[0:,0] + combine2[0:,2]
        
        combine3=combine3.detach()
        combine3=combine3[0]
        combine3=combine3/2+0.5
        
        save_image(combine3,'results/'+str(i).zfill(2)+'.png')