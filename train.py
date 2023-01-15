import os
import torch
from imshow import imshow

class Trainer():
    def __init__(self, save_dir1, save_dir2, save_dir3, model1, model2, model3, optimizer1, optimizer2, optimizer3, criterion, train_loader, test_loader, device = 'cuda'):
        print('===> Initializing trainer')
        self.mode = 'train' # 'val', 'test'
        self.save_dir1 = save_dir1
        self.save_dir2 = save_dir2
        self.save_dir3 = save_dir3
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.optimizer1 = optimizer1
        self.optimizer2 = optimizer2
        self.optimizer3 = optimizer3
        self.criterion = criterion
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

    def save(self):
        torch.save(self.model1.state_dict(), self.save_dir1)
        torch.save(self.model2.state_dict(), self.save_dir2)
        torch.save(self.model3.state_dict(), self.save_dir3)

    def train(self, epoch):
        print('===> Start training')
        
        '''
        total1 = 0.0
        total2 = 0.0
        total3 = 0.0
        for i, data in enumerate(self.train_loader,0):
            inputs, labels = data
            loss1 = self.criterion(inputs[0:,0],labels[0:,0])
            loss2 = self.criterion(inputs[0:,1],labels[0:,1])
            loss3 = self.criterion(inputs[0:,2],labels[0:,2])
            total1 += loss1.item()
            total2 += loss2.item()
            total3 += loss3.item()
          
        print('Basic loss1 = ',total1/len(self.train_loader))
        print('Basic loss2 = ',total2/len(self.train_loader))
        print('Basic loss3 = ',total3/len(self.train_loader))
        '''
        
        for epoch in range(epoch):
            loss1 = 0.0
            loss2 = 0.0
            loss3 = 0.0
            
            for i, data in enumerate(self.train_loader,0):
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer1.zero_grad()
                self.optimizer2.zero_grad()
                self.optimizer3.zero_grad()
                
                combine1 = inputs.clone()
                outputs = self.model1(inputs)
                combine1[0:,0] = outputs[0:,0] + inputs[0:,0]
                loss = self.criterion(combine1[0:,0],labels[0:,0])
                loss.to(self.device)
                loss.backward()
                self.optimizer1.step()
                combine1 = combine1.detach()
                loss1 += loss.item()
                
                combine2 = combine1.clone()
                outputs = self.model2(combine1)
                combine2[0:,1] = outputs[0:,0] + combine2[0:,1]
                loss = self.criterion(combine2[0:,1],labels[0:,1])
                loss.to(self.device)
                loss.backward()
                self.optimizer2.step()
                combine2 = combine2.detach()
                loss2 += loss.item()
                          
                combine3 = combine2.clone()
                outputs = self.model3(combine2)
                combine3[0:,2] = outputs[0:,0] + combine3[0:,2]
                loss = self.criterion(combine3[0:,2],labels[0:,2])
                loss.to(self.device)
                loss.backward()
                self.optimizer3.step()
                loss3 += loss.item()
                
                #imshow(inputs[0])
                #imshow(combine3[0])
                if (i+1)%500 == 0:
                    print(f'iter:{i+1:3d}')
                    print(f'loss1: {loss1/(i+1):.9f}')
                    print(f'loss2: {loss2/(i+1):.9f}')
                    print(f'loss3: {loss3/(i+1):.9f}')
                    
            self.save()
                    
                    
        self.model.eval()
