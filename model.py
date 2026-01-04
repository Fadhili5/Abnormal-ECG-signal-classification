import torch 
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
import cv2
import shap
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable

arr = []
ids = []
root = "." 
target_dir = "ECGDataDenoised"
for dirpath, dirnames, filenames in os.walk(root):
    if target_dir in dirnames:
        target_path = os.path.join(dirpath, target_dir)

        for file in os.listdir(target_path):
            if file.lower().endswith(".csv"):
                file_path = os.path.join(target_path, file)
                df = pd.read_csv(file_path)
                lead_ii = df.iloc[:, 1].to_numpy()
                arr.append(lead_ii)
                ids.append(os.path.splitext(file)[0])  
        break

pairs = [(rid, torch.from_numpy(sig).float()) for rid, sig in zip(ids, arr)]
pairs = [(rid, x) for rid, x in pairs if len(x) == 4999]

ids_filtered = [rid for rid, _ in pairs]
x_list_filtered = [x for _, x in pairs]

# RPM Workflow
def Z_score(input_tensor, eps=1e-8):
    out = []
    for x in input_tensor:
        x = (x - x.mean()) / (x.std(unbiased=False) + eps)
        out.append(x)
    return out


x_list_norm = Z_score(x_list_filtered)  
m = 72
y_list = [F.adaptive_avg_pool1d(x.view(1,1,-1), m).squeeze(0).squeeze(0) for x in x_list_norm]

y = y_list[0]

rpm_tensors = []

for y in y_list:
    rpm = y[:, None] - y[None, :]
    rpm_min = rpm.min()
    rpm_max = rpm.max()

    rpm_img = (rpm - rpm_min) / (rpm_max - rpm_min + 1e-8)
    rpm_tensors.append(rpm_img.unsqueeze(0).float())
    

x_rpm = torch.stack(rpm_tensors, dim=0)
print(x_rpm.shape)

# mapping the heart conditions to indices
diag = pd.read_excel(os.path.join(root, "Diagnostics.xlsx"))
id_to_labelname = dict(zip(diag.iloc[:,0].astype(str), diag.iloc[:, 1].astype(str)))

class_map = {
    "SB": 0,
    "SR": 1,
    "AFIB": 2,
    "ST": 3,
    "AF": 4,
    "SI": 5,
    "SVT": 6,
    "AT": 7,
    "AVNRT": 8,
    "AVRT": 9,
    "SAAWR": 10
}


y_list = []
keep_indices = []

for i, rid in enumerate(ids_filtered):
    if rid in id_to_labelname:
        label_name = id_to_labelname[rid].upper().strip()

        if label_name in class_map:
            y_list.append(class_map[label_name]) # maps classes to indices then appends indices as labels
            keep_indices.append(i)


x = x_rpm[keep_indices]
y = torch.tensor(y_list, dtype=torch.long)


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
            nn.LazyLinear(64),
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


class ClassificationModule(nn.Module):
    def __init__(self, in_channels, num_classes, input_length=None):
        super(ClassificationModule, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size,padding)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size, stride)
        self.conv2 = nn.Conv1d(64, 128, kernel_size, padding)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size, stride)
        self.conv3 = nn.Conv1d(128, 256, kernel_size, padding)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size, stride)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.flattened_size = None
        self.fc1 = None
        self.fc2 = None
        self.num_classes = num_classes
        self.input_length = input_length

        if input_length is not None:
            self.initialize_dense_layers(input_length)

    def initialize_dense_layers(self, input_length):
        """After we know the input size"""
        return x

    def forward(self, x):
        if self.fc1 is None:
            input_length = x.size()
            self.initialize_dense_layers(input_length)
            self.to(x.device)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.dropout1(x)
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)

        return x
            

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        #register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def generate_cam(self, input_tensor, class_idx= None):
        self.model.eval()
        
        #forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1)
            
        #backward pass
        self.model.zero_grad()
        score = output[:, class_idx] if isinstance(class_idx, int) else output.gather(1, class_idx.unsqueeze(1))
        score.backward()
        
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        weights = torch.mean(gradients, dim=(1, 2), keepdim=True) # Global average pooling of gradients
        
        cam = torch.sum(weights * activations, dim=0, keepdim=True) # Weighted combination of activation map
        cam = F.relu(cam)
        #Normalize the CAM to 0-1
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam, output
    
class SHAPExplainer:
    def __init__(self, model, background_data, device='cpu'):
        self.model = model
        self.device = device
        self.background_data = background_data.to(device)
        self.explainer = None
        
    def create_explainer(self):
        """Create the SHAP deep explainer"""
        self.model.eval()
        self.explainer = shap.DeepExplainer(self.model, self.background_data)
        return self.explainer
    
    def explain(self, input_data, n_samples=100):
        """Get SHAP values for the input data"""
        if self.explainer is None:
            self.create_explainer()
            
        input_data = input_data.to(self.device)
        shap_values = self.explainer.shap_values(input_data, nsamples = n_samples)
        return shap_values
    
    def plot_shap_values(self, shap_values, input_data, class_idx=0, max_display=10):
        shap.summary_plot(shap_values[class_idx], input_data.cpu().numpy(), max_display=max_display, show=False)
            

dataset = TensorDataset(x, y)

len_ds = len(dataset)
train = int(0.8 * len_ds)
val = len_ds - train

        
        
        
        
        
