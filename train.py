import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt

import numpy as np
import json
import argparse
import os

parse = argparse.ArgumentParser(description = 'training script')

parse = argparse.ArgumentParser() # Ensure that ArgumentParser is only defined once
parse.add_argument('data_dir', help='data directory', type=str)
parse.add_argument('--save_dir', help='save directory', type=str)
parse.add_argument('--arch', help='model architecture', type=str, default='vgg16')
parse.add_argument('--learning_rate', help='learning rate', type=float, default=0.001)
parse.add_argument('--hidden_units', help='hidden units', type=int, default=512)
parse.add_argument('--epochs', help='number of epochs', type=int, default=10)
parse.add_argument('--GPU', help='use GPU', type=bool)

args = parse.parse_args(args=['/path/to/your/data', '--arch', 'vgg16', '--learning_rate', '0.01',
                                 '--hidden_units', '512', '--epochs', '10',
                                 '--GPU', 'True']) # Pass a list of strings as arguments

data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

image_datasets = {
    'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
    'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
    'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
}

train_loader = DataLoader(image_datasets['train'], batch_size=64, shuffle=True)
valid_loader = DataLoader(image_datasets['valid'], batch_size=64)
test_loader = DataLoader(image_datasets['test'], batch_size=64)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
if args.arch == 'vgg16':
    model = models.vgg16(pretrained=True)
    no_input_layer = 25088
else:
    model = models.resnet50(pretrained=True)
    no_input_layer = 2048
    
for param in model.parameters():
    param.requires_grad = False
    
if args.hidden_units != None:
  classifier = nn.Sequential(
      nn.Linear(no_input_layer, args.hidden_units),
      nn.ReLU(),
      nn.Dropout(0.2),
      nn.Linear(args.hidden_units, 2048),
      nn.ReLU(),
      nn.Dropout(0.2),
      nn.Linear(2048, 102),
      nn.LogSoftmax(dim=1))
  
else:
  classifier = nn.Sequential(
      nn.Linear(no_input_layer, 2048),
      nn.ReLU(),
      nn.Dropout(0.2),
      nn.Linear(2048, 102),
      nn.LogSoftmax(dim=1))

model.classifier = classifier

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

if args.GPU == 'True':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
    
def validation(model, valid_loader, criterion):
    model.to(device)

    valid_loss = 0
    accuracy = 0
    for inputs, labels in valid_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        output = model.forward(inputs)
        valid_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return valid_loss, accuracy


# training the model
model.to(device)
epochs = args.epochs
print_every = 10
steps = 0

print(f"\nINITIALIZING TRAINING PHASE WITH {args.epochs} EPOCHS")
print(f"Training {args.arch} network architecture.")
print(f"Learning rate is set to {args.learning_r}.")

for e in range(epochs):
    running_loss = 0
    for ii, (inputs, labels) in enumerate(train_loader):
        steps += 1

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if steps % print_every == 0:
            model.eval()

            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                valid_loss, accuracy = validation(model, valid_loader, criterion)

            print('Epoch: {}/{}.. '.format(e + 1, epochs),
                  'Training Loss: {:.2f}.. '.format(
                      running_loss / print_every),
                  'Valid Loss: {:.2f}.. '.format(
                      valid_loss / len(valid_loader)),
                  'Valid Accuracy: {:.2f}%'.format(accuracy / len(valid_loader) * 100))

            running_loss = 0

            model.train()

model.class_to_idx = image_datasets['train'].class_to_idx
checkpoint = {'classifier': model.classifier, 'state_dict': model.state_dict (), 'class_to_idx': model.class_to_idx,'arch': args.arch} 

if args.save_dir:
    torch.save (checkpoint, args.save_dir + '/checkpoint.pth')
else:
    torch.save (checkpoint, 'checkpoint.pth')
    