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
        for epoch in range(epoch):
            loss1 = loss2 = loss3 = 0.0
            for i, data in enumerate(self.train_loader,0):
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer1.zero_grad()
                self.optimizer2.zero_grad()
                self.optimizer3.zero_grad()
                
                outputs = self.model1(inputs) + inputs
                loss = self.criterion(outputs, labels)
                loss.to(self.device)
                loss.backward()
                self.optimizer1.step()
                outputs = outputs.detach()
                loss1 += loss.item()

                outputs = self.model2(outputs) + outputs
                loss = self.criterion(outputs, labels)
                loss.to(self.device)
                loss.backward()
                self.optimizer2.step()
                outputs = outputs.detach()
                loss2 += loss.item()
                          
                outputs = self.model3(outputs) + outputs
                loss = self.criterion(outputs, labels)
                loss.to(self.device)
                loss.backward()
                self.optimizer3.step()
                loss3 += loss.item()
            print(f'epoch:{epoch:3d}')
            print(f'loss1: {loss1/(len(self.train_loader)):.9f}')
            print(f'loss2: {loss2/(len(self.train_loader)):.9f}')
            print(f'loss3: {loss3/(len(self.train_loader)):.9f}')  
            self.save()
            
    def test(self):
        print('===> Start testing')
        loss1 = loss2 = loss3 = 0.0
        with torch.no_grad():
            for i, data in enumerate(self.train_loader,0):
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model1(inputs) + inputs
                loss = self.criterion(outputs, labels)
                loss1 += loss.item()

                outputs = self.model2(outputs) + outputs
                loss = self.criterion(outputs, labels)
                loss2 += loss.item()
                          
                outputs = self.model3(outputs) + outputs
                loss = self.criterion(outputs, labels)
                loss3 += loss.item()     
            print('Test loss1 = ',loss1/len(self.test_loader))
            print('Test loss2 = ',loss2/len(self.test_loader))
            print('Test loss3 = ',loss3/len(self.test_loader))