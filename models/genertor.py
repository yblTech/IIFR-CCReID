import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.nn import Parameter
from models.textembedding import TexualEmbeddingLayer
class Discriminator(nn.Module):
    def __init__(self, feature_dim):
        super(Discriminator, self).__init__()
        self.network1 = nn.Sequential(
            nn.Linear(feature_dim+512, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1)
        )
        self.network2 = nn.Sequential(
            nn.Linear(feature_dim+512, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1)
        )
    def forward(self, x,condition, mode=0):
        if mode ==0:
            return self.network1(torch.cat((x,condition[:,0,:]),dim=1))
        return self.network1(torch.cat((x,condition[:,0,:]),dim=1))

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(4096, 1024)
        self.ln1 = nn.LayerNorm(1024) 
        self.fc2 = nn.Linear(1024, 512)
        self.ln2 = nn.LayerNorm(512)  
        self.fc3 = nn.Linear(512, 256)
        self.ln3 = nn.LayerNorm(256)   
        self.atttn1 = TexualEmbeddingLayer(embed_dim=256)
        self.fc4 = nn.Linear(256, 512)
        self.ln4 = nn.LayerNorm(512)   
        self.atttn2 = TexualEmbeddingLayer(embed_dim=512)
        self.fc5 = nn.Linear(512, 1024)
        self.ln5 = nn.LayerNorm(1024)  
        self.atttn3 = TexualEmbeddingLayer(embed_dim=1024)
        self.fc6 = nn.Linear(1024, 4096)
        
    def forward(self, x, condition, attn, text):
        x1 = F.relu(self.ln1(self.fc1(x))) 
        x2 = F.relu(self.ln2(self.fc2(x1)))
        x3 = F.relu(self.ln3(self.fc3(x2))) + self.atttn1(condition, text, attn)
        x = F.relu(self.ln4(self.fc4(x3))) + x2 + self.atttn2(condition, text, attn)
        x = F.relu(self.ln5(self.fc5(x))) + x1 + self.atttn3(condition, text, attn)
        x = self.fc6(x)
        return x
    
class DouGen(nn.Module):
    def __init__(self):
        super(DouGen, self).__init__()
        self.gen1 = SimpleNN()
        self.gen2 = SimpleNN()
    def forward(self, inputs1, inputs2,c1,a1,t1,c2,a2,t2):
        out1_2 = self.gen2(inputs1,c1,a1,t1)
        out2_1 = self.gen1(inputs2,c2,a2,t2)
        out1_1 = self.gen1(out1_2,c2,a2,t2)
        out2_2 = self.gen2(out2_1,c1,a1,t1)
        return out1_2, out2_1, out1_1, out2_2
    
