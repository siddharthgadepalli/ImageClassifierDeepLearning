import matplotlib.pyplot as plt
import numpy as np
import torch
import json
import argparse
from torchvision import datasets, transforms, models
from PIL import Image
from matplotlib.ticker import FormatStrFormatter
import torch.nn.functional as F
from train import train_model, neural_network_pretrain_method
from torch import nn
from torchvision import models
import torchvision.models as models
from collections import OrderedDict

parser = argparse.ArgumentParser(description='Predict with the trained model with these inputs for image classification. Pass arguments in this order: path, cat_to_name, img, topk')
parser.add_argument('--load_path', type=str, help='Path where model is stored')
parser.add_argument('--cat_to_name', type=str, help='name of file that has mapping to categories of flowers')
parser.add_argument('--img', type=str, help='path to test image file')
parser.add_argument('--topk', type=int, help='top k probabilities of flowers returned')


parser.set_defaults(load_path = 'checkpoint_densenet121.pth', cat_to_name='cat_to_name.json', img='flowers/test/1/image_06743.jpg', topk=5)
args = parser.parse_args()

load_path = args.load_path
cat_to_name = args.cat_to_name
img = args.img
topk = args.topk


def load_model(load_path):
    print('Loading model from checkpoint....')
    if not torch.cuda.is_available():
        checkpoint = torch.load(load_path, map_location='cpu')
    else:
        checkpoint = torch.load(load_path)

    if 'h1' in checkpoint:
        h1 = checkpoint['h1']
    else:
        h1 = 120
    if checkpoint['transfer_model'] == 'densenet121':
        model = models.densenet121(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
            classifier = nn.Sequential(
                OrderedDict([('dropout',nn.Dropout(dropout)),
                             ('inputs', nn.Linear(1024, h1)),
                             ('relu1', nn.ReLU()),('hidden_layer1', nn.Linear(h1, 90)),
                             ('relu2',nn.ReLU()),('hidden_layer2',nn.Linear(90,80)),
                             ('relu3',nn.ReLU()),('hidden_layer3',nn.Linear(80,102)),
                             ('output', nn.LogSoftmax(dim=1))]))
            model.classifier = classifier

    else: 
        model = models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(
            OrderedDict([('dropout',nn.Dropout(dropout)),
                         ('inputs', nn.Linear(1024, h1)),
                         ('relu1', nn.ReLU()),('hidden_layer1', nn.Linear(h1, 90)),
                         ('relu2',nn.ReLU()),('hidden_layer2',nn.Linear(90,80)),
                         ('relu3',nn.ReLU()),('hidden_layer3',nn.Linear(80,102)),
                         ('output', nn.LogSoftmax(dim=1))]))
        model.fc = classifier

    model.load_state_dict(checkpoint['state_dict'])
    model.idx_to_class = checkpoint['idx_to_class']
    return model

def load_cat_to_json(cat_to_name):
    if cat_to_name == '' or cat_to_name is None:
        cat_to_name = 'cat_to_name.json'
    else: 
        cat_to_name = cat_to_name
    with open(cat_to_name, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

   
def process_image(image):
    
    image = Image.open(image)
    transformations = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor_for_image = transformations(image)
    return tensor_for_image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    image = image.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    return ax

def predict(image_path, load_path, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model = load_model(load_path) 
    device = torch.device('cuda' if torch.cuda.is_available() and gpu == 'True' else 'cpu')
    model.to(device)
    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()
    
    with torch.no_grad():
        output = model.forward(img_torch.cuda())
        
    probability = F.softmax(output.data,dim=1)
    resulting_class = model.items()
    print(resulting_class)
    return probability.topk(topk), resulting_class
    
def sanity(img, load_path, topk, gpu):
    plt.rcParams["figure.figsize"] = (10,5)
    plt.subplot(211)
    index = 2
    img_path = img
    probabilities, resulting_class = predict(img_path, load_path, gpu)
    image = process_image(img_path)
    probabilities = probabilities
    axs = imshow(image, ax = plt)
    axs.axis('off')
    axs.title(cat_to_name[str(index)])
    axs.show()
    a = np.array(probabilities[0][0])
    b = [cat_to_name[str(index + 1)] for index in np.array(probabilities[1][0])]
    #idx_to_class = {val: key for key, val in model['class_to_idx'].items()}
    #actual_classes = [idx_to_class[] for idx in np.array(probabilities[1][0])]
    categories = [cat_to_name[cls] for cls in actual_classes]
    N=float(len(b))
    fig,ax = plt.subplots(figsize=(8,3))
    width = 0.8
    tickLocations = np.arange(N)
    ax.bar(tickLocations, a, width, linewidth=4.0, align = 'center')
    ax.set_xticks(ticks = tickLocations)
    ax.set_xticklabels(b)
    ax.set_xlim(min(tickLocations)-0.6,max(tickLocations)+0.6)
    ax.set_yticks([0.2,0.4,0.6,0.8,1,1.2])
    ax.set_ylim((0,1))
    ax.yaxis.grid(True)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.show()
    
if __name__ == '__main__':
    model = load_model(load_path)
    cat_to_name = load_cat_to_json()
    tensor_for_image = process_image(img)
    ax = imshow(tensor_for_image)
    probabilities, resulting_class = predict(img, load_path, gpu)
    sanity(img, load_path, topk, gpu)
