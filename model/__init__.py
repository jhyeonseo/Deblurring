import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from imshow import imshow


class NET(nn.Module):
      def __init__(self):
        super().__init__()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.encoder = nn.Sequential(
          nn.Conv2d(3,64,3,padding=1).to(self.device),
          nn.BatchNorm2d(64).to(self.device),
          nn.ReLU(),
          nn.Conv2d(64,128,3,stride=2).to(self.device),
          nn.BatchNorm2d(128).to(self.device),
          nn.ReLU(),
          nn.Conv2d(128,256,3,stride=2).to(self.device),
          nn.BatchNorm2d(256).to(self.device),
          nn.ReLU()
          )
        self.decoder = nn.Sequential(
          nn.ConvTranspose2d(256,128,3,stride=2).to(self.device),
          nn.BatchNorm2d(128).to(self.device),
          nn.ReLU(),
          nn.ConvTranspose2d(128,64,3,stride=2).to(self.device),
          nn.BatchNorm2d(64).to(self.device),
          nn.ReLU(),
          nn.Conv2d(64,3,3,padding=1).to(self.device),
          nn.Tanh()
          )
        
        self.resblock1 = nn.Sequential(
          nn.Conv2d(256,256,3,padding=1).to(self.device),
          nn.BatchNorm2d(256).to(self.device),
          nn.ReLU(),
          nn.Conv2d(256,256,3,padding=1).to(self.device),
          nn.BatchNorm2d(256).to(self.device)
        )
        self.resblock2 = nn.Sequential(
          nn.Conv2d(256,256,3,padding=1).to(self.device),
          nn.BatchNorm2d(256).to(self.device),
          nn.ReLU(),
          nn.Conv2d(256,256,3,padding=1).to(self.device),
          nn.BatchNorm2d(256).to(self.device)
        )
        self.resblock3 = nn.Sequential(
          nn.Conv2d(256,256,3,padding=1).to(self.device),
          nn.BatchNorm2d(256).to(self.device),
          nn.ReLU(),
          nn.Conv2d(256,256,3,padding=1).to(self.device),
          nn.BatchNorm2d(256).to(self.device)
        )
        self.resblock4 = nn.Sequential(
          nn.Conv2d(256,256,3,padding=1).to(self.device),
          nn.BatchNorm2d(256).to(self.device),
          nn.ReLU(),
          nn.Conv2d(256,256,3,padding=1).to(self.device),
          nn.BatchNorm2d(256).to(self.device)
        )
        self.resblock5 = nn.Sequential(
          nn.Conv2d(256,256,3,padding=1).to(self.device),
          nn.BatchNorm2d(256).to(self.device),
          nn.ReLU(),
          nn.Conv2d(256,256,3,padding=1).to(self.device),
          nn.BatchNorm2d(256).to(self.device)
        )
        self.resblock6 = nn.Sequential(
          nn.Conv2d(256,256,3,padding=1).to(self.device),
          nn.BatchNorm2d(256).to(self.device),
          nn.ReLU(),
          nn.Conv2d(256,256,3,padding=1).to(self.device),
          nn.BatchNorm2d(256).to(self.device)
        )
        self.resblock7 = nn.Sequential(
          nn.Conv2d(256,256,3,padding=1).to(self.device),
          nn.BatchNorm2d(256).to(self.device),
          nn.ReLU(),
          nn.Conv2d(256,256,3,padding=1).to(self.device),
          nn.BatchNorm2d(256).to(self.device)
        )
        self.resblock8 = nn.Sequential(
          nn.Conv2d(256,256,3,padding=1).to(self.device),
          nn.BatchNorm2d(256).to(self.device),
          nn.ReLU(),
          nn.Conv2d(256,256,3,padding=1).to(self.device),
          nn.BatchNorm2d(256).to(self.device)
        )
        self.resblock9 = nn.Sequential(
          nn.Conv2d(256,256,3,padding=1).to(self.device),
          nn.BatchNorm2d(256).to(self.device),
          nn.ReLU(),
          nn.Conv2d(256,256,3,padding=1).to(self.device),
          nn.BatchNorm2d(256).to(self.device)
        )        
        
        self.dropout = nn.Dropout(p=0.5).to(self.device)
            
      def forward(self,x):
        x = self.encoder(x)
        t = x
        
        x = self.resblock1(x) + t
        x = self.dropout(x)
        t = x
        
        x = self.resblock2(x) + t
        x = self.dropout(x)
        t = x
        
        x = self.resblock3(x) + t
        x = self.dropout(x)
        t = x
        
        x = self.resblock4(x) + t
        x = self.dropout(x)
        t = x
        
        x = self.resblock5(x) + t
        x = self.dropout(x)
        t = x
        
        x = self.resblock6(x) + t
        x = self.dropout(x)
        t = x
        
        x = self.resblock7(x) + t
        x = self.dropout(x)
        t = x
        
        x = self.resblock8(x) + t
        x = self.dropout(x)
        t = x
        
        x = self.resblock9(x) + t
        x = self.dropout(x)
        
        x = self.decoder(x)
        return x