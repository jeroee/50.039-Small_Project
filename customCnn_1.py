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

from utils import load_model, plot_curves, test_model, validation, save_checkpoint


class NN_Classifier_1(nn.Module):
    def __init__(self, output_size, drop_p=0.75):
        """
        Initialising parameters used in the custom CNN model to classify healthy and infected lungs
        """
        super(NN_Classifier_1,self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)      
        #input channels, output channels, kernel size, stride, padding
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(3)
        # shape: 64 32, 75 75

        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.relu2 = nn.ReLU()

        self.fc1 = nn.Linear(64*50*50, 256)
        self.dropout = nn.Dropout(drop_p)
        self.fc2 = nn.Linear(256, output_size)
        
    def forward(self, x):
        """
        Building the custom model with the initialised parameters
        """
        
        x = self.conv1(x)
        x = self.relu1(x)  
        x = self.pool1(x)
        x = self.dropout(x)

        x= self.conv2(x)
        x = self.relu2(x)
        
        # to flatten into 1D
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x) 
        output = F.log_softmax(x, dim=1)
        return output
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
    
    
def train_model_1(model, n_epoch, labelsdict, criterion, optimizer, device, trainloader, validloader, train_data, model_name, model_path, model_path_best, loss_graph, accuracy_graph, start_epoch = 0, valid_loss = 1000):
    """
    Commence training of model
    
    model: model used
    n_epoch: number of epoch used for training
    labelsdict: dictionary containing class names which correnspond to their respective indexes
    optimizer: choice of optimizer use for training
    device: 'cuda' or 'cpu' (speed up training)
    trainloader: input training data split in batches
    validloader: input validation data split in batches
    train_data: input training data
    model_name: name of model indicated
    model_path: path where model checkpoint is saved at every epoch
    model_path_best: path where model yields best training result is saved (lowest val acc)
    loss_graph: path of graph indicating training and validation losses of model is saved
    accuracy_graph: path of graph indicating training and validation accuracies of model is saved
    start_epoch: indicate start epoch.(where start epoch != 0 when model is not trained from scratch but loaded and retrained)
    valid_acc: indicate value of best validation accuracy during point of training
    """
    print(f'Training custom CNN Model to distinguish normal and infected lungs')
    print(f'total epochs: {n_epoch}')
    if start_epoch!=0:
        print(f'Retraining model continuing from epoch {start_epoch+1}')
    n_in = next(model.fc2.modules()).in_features
    model.to(device)
    start = time.time()
    epochs = n_epoch
    steps = 0 
    running_loss = 0
    running_acc = 0
    print_every = len(trainloader)
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    val_loss_max = valid_loss
    Singapore = pytz.timezone('Asia/Singapore')
    for e in range(start_epoch,epochs):
        # Make sure training is on
        model.train()
        for images, labels, path in trainloader: # for each batch
            images, labels = images.to(device), labels.to(device)

            steps+=1

            optimizer.zero_grad()
            output = model.forward(images)
            # getting loss
            loss = criterion(output,labels)
            loss.backward()
            optimizer.step()
            # getting accuracy
            ps = torch.exp(output)
            equality = (labels == ps.max(dim=1)[1])
            
            running_acc += equality.type(torch.FloatTensor).mean()
            running_loss += loss.item()
            
            # At the end of every epoch...
            if steps % print_every == 0:   
                # Eval mode for predictions
                model.eval()
                # Turn off gradients for validation
                with torch.no_grad():
                    test_loss, accuracy = validation(model, validloader, criterion, device)
                # log results at every epoch
                print("Epoch: {}/{} - ".format(e+1, epochs),
                      "Time: {} ".format(datetime.now(Singapore)),
                      "Training Loss: {:.3f} - ".format(running_loss/len(trainloader)),
                      "Validation Loss: {:.3f} - ".format(test_loss/len(validloader)),
                      "Training Accuracy: {:.3f} - ".format(running_acc/len(trainloader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))
                
                # saving results into a list for plotting
                train_loss.append(running_loss/print_every)
                val_loss.append(test_loss/len(validloader))
                train_acc.append(running_acc/len(trainloader))
                val_acc.append(accuracy/len(validloader))
                 
                valid_loss = test_loss/len(validloader)
                # saving checkpoint
                model.n_in = n_in
                model.n_out = len(labelsdict)
                model.labelsdict = labelsdict
                model.optimizer = optimizer
                model.optimizer_state_dict = optimizer.state_dict() 
                model.model_name = model_name
                model.loss = criterion
                model.val_loss = valid_loss
                
                loss_acc=[]
                loss_acc.append(train_loss)
                loss_acc.append(val_loss)
                loss_acc.append(train_acc)
                loss_acc.append(val_acc)
                model.loss_acc = loss_acc
                model.start_epoch = start_epoch
                model.epoch = e+1
                path = model_path
                path_best = model_path_best
                # saving checkpoint model at every epoch
                save_checkpoint(model, path) 
                
                # saving best model during training, best indicated by highest validation accuracy obtained
                if valid_loss <= val_loss_max:
                    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(val_loss_max,valid_loss))
                    # update threshold
                    val_loss_max = valid_loss
                    save_checkpoint(model, path_best)
                # reset training loss and accuracy after validation, which is used again for subsequent training epoch
                running_loss = 0
                running_acc = 0 
                
    
    print('model:', model_name,'- epochs:', n_epoch)
    print(f"Run time: {(time.time() - start)/60:.3f} min")

    # plotting the graph on training and validation loss for model
    plot_curves(start_epoch, model.epoch, loss_acc, model_name, loss_graph, accuracy_graph)

    return model
