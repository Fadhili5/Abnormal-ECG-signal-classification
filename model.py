import torch 
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, stride, momentum, padding): 
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_features, out_features, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(in_features, out_features, kernel_size, stride, padding) 
        self.conv3 = nn.Conv2d(out_features, out_features, kernel_size, stride, padding) 
        self.bn1 = nn.BatchNorm2d(out_features, momentum=momentum)
        self.bn2 = nn.BatchNorm2d(out_features, momentum=momentum)
        self.bn3 = nn.BatchNorm2d(out_features, momentum=momentum)
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
    def __init__(self, in_features, out_features, kernel_size, pool_size, conv_stride, pool_stride, dropout, momentum, num_layers, softmax_dim, padding):
        super(FeatureExtractionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_features, out_features, kernel_size, stride=conv_stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_features, momentum=momentum)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(pool_size, pool_stride)
        self.layers = nn.ModuleList([ResidualBlock(out_features, out_features, kernel_size, conv_stride, momentum, padding) for _ in range(num_layers)])
        self.conv2 = nn.Conv2d(out_features, out_features, kernel_size, stride=conv_stride, padding=padding)
        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Sequential(
            nn.Linear(#trouble finding dimensions after flattening, 64),
            nn.ReLU()
        )
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Sequential(
            nn.Linear(64, 7),
            nn.Softmax(dim=softmax_dim)
        )
        

    def forward(self, x):
        fwd = self.conv1(x)
        fwd = self.bn1(fwd)
        fwd = self.relu(fwd)
        fwd = self.maxpool(fwd)
        for layer in self.layers:
            fwd = layer(fwd)
    
        fwd = self.conv2(fwd)
        fwd = self.flatten(fwd)
        fwd = self.dropout1(fwd)
        feature_vectors = self.fc1(fwd)
        fwd = self.dropout2(feature_vectors)
        out = self.fc2(fwd)
        return feature_vectors, out
        
        
        
        
        
        
        
        
        
