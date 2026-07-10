import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class MultiScaleFeatureEncoder(nn.Module):
    def __init__(self, model_name='resnet34', output_dim=256, freeze=True):
        super().__init__()
        
        if model_name == 'resnet18':
            resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif model_name == 'resnet34':
            resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        else:
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1 
        self.layer2 = resnet.layer2  
        self.layer3 = resnet.layer3  
        self.layer4 = resnet.layer4  
        
        if freeze:
            for param in self.parameters():
                param.requires_grad = False
        
        self.proj1 = nn.Linear(64, output_dim)
        self.proj2 = nn.Linear(128, output_dim)
        self.proj3 = nn.Linear(256, output_dim)
        self.proj4 = nn.Linear(512, output_dim)
        
        self.norm = nn.LayerNorm(output_dim)
        
        self.in_proj = nn.Linear(32, 3)
        self.eval()
    
    def forward(self, x):
        B, C, H, W = x.shape

        # print(x.min(), x.max(), x.std())

        # x = self.in_proj(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        if x.min() < 0:
            x = (x + 1) / 2
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x) # [B, 64, H // 2, W // 2]
        
        # x = self.maxpool(x)  # [B, 64, H // 8, W // 8]

        f1 = self.layer1(x)   # [B, 64,  H // 8,  W // 8]
        f2 = self.layer2(f1)  # [B, 128, H // 16,  W // 16]
        f3 = self.layer3(f2)  # [B, 256, H // 32, W // 32]
        f4 = self.layer4(f3)  # [B, 512, H // 64, W // 64]
        # return f4

        def process_feature(f, proj):
            f = F.adaptive_avg_pool2d(f, (1, 1))  # [B, C, 1, 1]
            f = f.flatten(1)  # [B, C]
            f = proj(f)  # [B, output_dim]
            return self.norm(f)
        
        features = [
            process_feature(f1, self.proj1).view(B, -1),
            process_feature(f2, self.proj2).view(B, -1),
            process_feature(f3, self.proj3).view(B, -1),
            process_feature(f4, self.proj4).view(B, -1)
        ]
        
        return features #torch.cat(features, dim=1)
