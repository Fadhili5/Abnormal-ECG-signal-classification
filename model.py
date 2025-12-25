import torch 
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, stride, momentum):
        super(ResidualBlock, self).__init__()
        padding = kernel_size // 2 
        self.conv1 = nn.Conv2d(in_features, out_features, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(in_features, out_features, kernel_size, stride, padding) 
        self.conv3 = nn.Conv2d(out_features, out_features, kernel_size, stride, padding) 
        self.bn1 = nn.BatchNorm2d(out_features, momentum)
        self.bn2 = nn.BatchNorm2d(out_features, momentum)
        self.bn3 = nn.BatchNorm2d(out_features, momentum)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        fwd = self.conv1(x)
        x = self.conv2(x) 
        x = self.bn2(x)
        fwd = self.bn1(fwd)
        fwd = self.relu(fwd)
        fwd = self.conv3(fwd)
        fwd = self.bn3(fwd)
        out = fwd + x
        out = self.relu(out)
        return out

class FeatureExtractionModule(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, pool_size, stride, momentum):
        super(FeatureExtractionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_features, out_features, stride)
        self.bn1 = nn.BatchNorm2d(out_features, momentum)
        self.relu = nn.relu()
        self.mp = nn.MaxPool2d(pool_size, stride)
        self.rb = ResidualBlock(in_features, out_features, kernel_size, stride)
        self.conv2 = nn.Conv2d(out_features, out_features, kernel_size, stride)
        ### flatten
        
        
        
