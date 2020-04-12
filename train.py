from torchvision import datasets, transforms, models
import torch
import argparse
import torchvision.models as models
from collections import OrderedDict
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import json
from torch import optim
from matplotlib.ticker import FormatStrFormatter
import torch.nn.functional as F


parser = argparse.ArgumentParser(description='Train your deep learning model for image classification. Pass arguments in this order: data_dir, train_dir, valid_dir, test_dir, dropout, h1, learning_rate, model, epochs, print_every, steps, save_directory, category_names, gpu')
parser.add_argument('--data_dir', type=str, help='Name of the directory with the flower data')
parser.add_argument('--train_dir', type=str, help='Name of the directory with the training data')
parser.add_argument('--valid_dir', type=str, help='Name of the directory with the validation data')
parser.add_argument('--test_dir', type=str, help='Name of the directory with the testing data')
parser.add_argument('--dropout', type=float, help='Dropout value to remove that much percentage of data from the training set randomly')
parser.add_argument('--h1', type=int, help='the size of your first hidden layer')
parser.add_argument('--learning_rate', type=float, help='learning rate')
parser.add_argument('--model', type=str, help='name of model')
parser.add_argument('--epochs', type=int, help='training model epochs')
parser.add_argument('--print_every', type=int, help='print_every')
parser.add_argument('--steps', type=int, help='steps')
parser.add_argument('--save_directory', type=str, help='directory path to save your model after training')

parser.add_argument('--category_names', type=str, help='Mapping file used to map categories to real names')
parser.add_argument('--gpu', type=str, help='Use GPU for prediction')

parser.set_defaults(gpu='False', category_names = 'cat_to_name.json', data_dir='flowers', train_dir = 'flowers/train', valid_dir='flowers/valid', test_dir = 'flowers/test', dropout=0.2, learning_rate=0.003, model='densenet121', epochs=12, print_every=5, steps=0, save_directory='checkpoint_resnet18.pth')
args = parser.parse_args()

data_dir = args.data_dir
train_dir = args.train_dir
valid_dir = args.valid_dir
test_dir = args.test_dir

dropout = args.dropout
h1 = args.h1
learning_rate = args.learning_rate
model = args.model
epochs = args.epochs
print_every = args.print_every
steps = args.steps
save_directory = args.save_directory
category_names = args.category_names
gpu = args.gpu


# TODO: Define your transforms for the training, validation, and testing sets
def prepare_data():
    training_data_transformations = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    testing_data_transformations = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    validation_data_transformations = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], 
                                                                     [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    training_data = datasets.ImageFolder(train_dir, transform=training_data_transformations)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_data_transformations)
    testing_data = datasets.ImageFolder(test_dir ,transform = testing_data_transformations)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(training_data, batch_size=64, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size =32,shuffle = True)
    testing_loader = torch.utils.data.DataLoader(testing_data, batch_size = 20, shuffle = True)
    
    prepared_tensors = {}
    prepared_tensors['training_data'] = training_data
    prepared_tensors['validation_data'] = validation_data
    prepared_tensors['testing_data'] = testing_data
    prepared_tensors['train_loader'] = train_loader
    prepared_tensors['validation_loader'] = validation_loader
    prepared_tensors['testing_loader'] = testing_loader
    
    return prepared_tensors

# TODO: Build and train your network
# Densenet 121
# dropout => Dropout_Probabiliy, h1 -> First Hidden Layer, lr => Learning Rate
# To flatten the image we take the 224*224 dimensions and then divide that in half to get the total input layers

def neural_network_pretrain_method(dropout, h1, learning_rate, model_input):
    
    prepared_tensors = prepare_data()
    training_data = prepared_tensors['training_data']
    validation_data = prepared_tensors['validation_data']
    testing_data = prepared_tensors['testing_data']
    
    validation_loader = prepared_tensors['validation_loader']
    testing_loader = prepared_tensors['testing_loader']
    train_loader = prepared_tensors['train_loader']
    
    if model_input == 'resnet18':
        model = models.resnet18(pretrained=True)
        input_layer = 512
    elif model_input == 'densenet121':
        model = models.densenet121(pretrained=True)
        input_layer = 1024
    
    for p in model.parameters():
        p.requires_grad = False
        classifier = nn.Sequential(OrderedDict([('dropout',nn.Dropout(dropout)),('inputs', nn.Linear(input_layer, h1)),('relu1', nn.ReLU()),('hidden_layer1', nn.Linear(h1, 90)),('relu2',nn.ReLU()),('hidden_layer2',nn.Linear(90,80)),('relu3',nn.ReLU()),('hidden_layer3',nn.Linear(80,102)),('output', nn.LogSoftmax(dim=1))]))
        
    if model_input == 'densenet121':
        model.classifier = classifier
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), learning_rate)
    elif model_input == 'resnet18':
        model.fc = classifier
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.fc.parameters(), learning_rate)
        
    #model.to('cuda')
    gpu_check = True 
    if torch.cuda.is_available() is True and gpu == 'True':
        gpu_check = True
    else:
        gpu_check = False
    
    if gpu_check is True:
        model.to('cuda')
        print("GPU Device available.")
    else:
        print('No GPU found. Please use a GPU to train your neural network.')
    
    model_objects = {}
    model_objects['model'] = model
    model_objects['optimizer'] = optimizer
    model_objects['criterion'] = criterion
    model_objects['train_loader'] = train_loader
    model_objects['validation_loader'] = validation_loader
    model_objects['training_data'] = training_data
    
    return model_objects

def save_model(save_directory, training_data, model_input):
    # TODO: Save the checkpoint 
    print("Saving Trained Model....")
    model.class_to_idx = training_data.class_to_idx
    model.cpu
    torch.save({'h1': h1,
                'dropout': dropout,
                'transfer_model': model_input,
                'learning_rate': learning_rate,
                'state_dict':model.state_dict(),
                'class_to_idx':model.class_to_idx},
                save_directory) 

def train_model(epochs, print_every, steps, learning_rate, model_input, h1, dropout):
    
    model_objects = neural_network_pretrain_method(dropout, h1, learning_rate, model_input)
    model = model_objects['model']
    training_data = model_objects['training_data']
    optimizer = model_objects['optimizer']
    criterion = model_objects['criterion']
    train_loader = model_objects['train_loader']
    validation_loader = model_objects['validation_loader']
    loss_show=[]

    gpu_check = False 
    if torch.cuda.is_available() is True and gpu == 'True':
        gpu_check = True
    else:
        gpu_check = False
    
    if gpu_check is True:         
        model.to('cuda')
        print("GPU Device available.")
    else:
        print('No GPU found. Please use a GPU to train your neural network.')

    for epoch in range(epochs):

        running_loss = 0
        for ii, (inputs, labels) in enumerate(train_loader):
            steps += 1
            device = torch.device('cuda' if torch.cuda.is_available() and gpu == 'True' else 'cpu')   
            inputs,labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                test_loss = 0
                accuracy=0
                for ii, (inputs2,labels2) in enumerate(validation_loader):
                    optimizer.zero_grad()
                    inputs2, labels2 = inputs2.to(device) , labels2.to(device)
                    model.to(device)
                    with torch.no_grad():    
                        outputs = model.forward(inputs2)
                        valid_loss = criterion(outputs,labels2)
                        ps = torch.exp(outputs).data
                        equals = (labels2.data == ps.max(1)[1])
                        accuracy += equals.type_as(torch.FloatTensor()).mean()
                test_loss = test_loss / len(validation_loader)
                accuracy = accuracy /len(validation_loader)
                training_loss = running_loss/print_every
                print("Epoch: {}/{}... ".format(epoch+1, epochs), "Loss: {:.3f}".format(training_loss), "Validation Lost {:.2f}".format(valid_loss),"Accuracy: {:.2f}".format(accuracy))
                running_loss = 0
    return training_loss, valid_loss, accuracy, model, training_data, model_input

# TODO: Do validation on the test set
def accuracy(model):   
    prepared_tensors = prepare_data()
    testing_loader = prepared_tensors['testing_loader']
    device = torch.device('cuda' if torch.cuda.is_available() and gpu == 'True' else 'cpu') 
    num_right = 0
    count = 0
    model.to(device)
    with torch.no_grad():
        for data in testing_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            count += labels.size(0)
            num_right += (predicted == labels).sum().item()
    result = 100*num_right/count
    print(str(int(result)) + '%')
    return result
            
if __name__ == '__main__':
    args = parser.parse_args()
    training_loss, valid_loss, accuracy_result, model, training_data, model_input = train_model(epochs, print_every, steps, learning_rate, model, h1, dropout)
    save_model(save_directory, training_data, model_input)
    result = accuracy(model)
