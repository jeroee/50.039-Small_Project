import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import sys
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms, models
import pandas as pd
import sklearn as sk
import os
import time 
from datetime import datetime
import pytz

def validation(model, testloader, criterion, device):
    """
    Obtain validation accuracy and loss during evaluation stage while trianing model. 
    """
    test_loss = 0
    accuracy = 0
    for images, labels, path in testloader:
        images, labels = images.to(device), labels.to(device)    
        output = model.forward(images)
        test_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return test_loss, accuracy


# Define function to load model
def load_model(input_model, path):
    """
    loader function to load model for use.
    input_model to indicate base model used and path will indicate the path where the model will be saved
    """
    cp = torch.load(path)
    model = input_model
    model.n_in = cp['c_input']
    model.n_out = cp['c_out']
    model.labelsdict = cp['labelsdict']
    model.optimizer = cp['optimizer']
    model.loss =cp['criterion']
    model.optimizer_state_dict = cp['opti_state_dict']
    model.model_name = cp['model_name']
    model.val_loss = cp['val_loss']
    # model.classifier.class_to_idx = cp['class_to_idx']
    model.load_state_dict(cp['state_dict'])
    model.start_epoch = cp['start_epoch']
    model.epoch = cp['epoch']
    model.loss_acc = cp['loss_acc']
    return model

# Define function to save checkpoint
def save_checkpoint(model, path):
    """
    save function to save model for later use, implementation or later training.
    model attributes will be saved as a dictionary's values and then saved into the path
    """
    checkpoint = {'c_input': model.n_in,
                  'c_out': model.n_out,
                  'labelsdict': model.labelsdict,
                  'state_dict': model.state_dict(),
                  'optimizer': model.optimizer,
                  'opti_state_dict': model.optimizer_state_dict,
                  'criterion': model.loss,
                  'model_name': model.model_name,
                  'val_loss':model.val_loss,
                  'start_epoch': model.start_epoch,
                  'epoch': model.epoch,
                  'loss_acc': model.loss_acc
                  # 'class_to_idx': model.classifier.class_to_idx
                  }
    torch.save(checkpoint, path)

def plot_curves(start_epoch, epoch, loss_acc, model_name, loss_graph, accuracy_graph):
    """
    plot curves to display progress of training.
    loss graph will display training and validation losses per epoch
    accuracy graph will display training and validation accuracy per epoch
    graph images will also be saved in directories loss_graph and accuracy_graph
    start_epoch and epoch indicates the start and end period of training
    loss_acc are the data on all the loss and accuracy values which are used for plotting the graphs
    """
    e = [i+1 for i in range(start_epoch, epoch)]
    train_loss = loss_acc[0]
    val_loss = loss_acc[1]
    train_acc = loss_acc[2]
    val_acc = loss_acc[3]        
    plt.plot(e,train_loss, label='Training Loss')
    plt.plot(e,val_loss, label='Validation Loss')
    plt.xticks(np.arange(min(e), max(e)+1, 1.0))
    plt.legend()
    plt.title(f'{model_name} loss',color='black')
    plt.xlabel('epoch',color='black')
    plt.ylim(ymin=0)
    plt.ylabel('loss',color='black')
    plt.tick_params(colors='black')
    plt.savefig(loss_graph,dpi=100,bbox_inches = 'tight')
    plt.show()

    plt.plot(e,train_acc, label= 'Training Accuracy')
    plt.plot(e,val_acc, label='Validation Accuracy')
    plt.xticks(np.arange(min(e), max(e)+1, 1.0))
    plt.legend()
    plt.title(f'{model_name} accuracy',color='black')
    plt.xlabel('epoch',color='black')
    plt.xlim(1,epoch+1)
    plt.ylim(ymin=0)
    plt.ylabel('accuracy',color='black')
    plt.tick_params(colors='black')
    plt.savefig(accuracy_graph,dpi=100,bbox_inches = 'tight')
    plt.show()


def test_model(model, testloader, device='cuda'): 
    """
    Obtain the accuracy of the model applied on the test dataloader
    """
    model.to(device)
    model.eval()
    accuracy = 0

    for images, labels, path in testloader:
        images, labels = images.to(device), labels.to(device)       
        output = model.forward(images)
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    print('Testing Accuracy: {:.3f}'.format(accuracy/len(testloader)))

    
def predict_images(model, images, device = 'cuda'):
    """
    Obtain prediction label class 0 or 1 from the model applied to a single image
    """
    out = model(images)
    prediction = torch.argmax(out,dim=1)
    return prediction


