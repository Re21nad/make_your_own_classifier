import torch
from torchvision import transforms, datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json

import argparse

parse = argparse.ArgumentParser (description = 'prediction script')

parse.add_argument ('image_path', help = 'image path', type = str)
parse.add_argument ('checkpoint', help = 'checkpoint', type = str)
parse.add_argument ('top_k', help = 'top k', type = int)
parse.add_argument ('category_names', help = 'category names', type = str)
parse.add_argument ('GPU', help = 'use GPU', type = bool)

args = parse.parse_args()

if args.GPU == 'True':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
    
def load_model(file_path):
    
    checkpoint = torch.load(file_path)  # loading checkpoint from a file
    if checkpoint ['arch'] == 'vgg16':
        model = models.vgg16(pretrained = True)
    else: 
        model = models.resnet50(pretrained = True)
 
    model.classifier = checkpoint['classifier']
    model.load_state_dict (checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    for param in model.parameters():
        param.requires_grad = False  # turning off tuning of the model

    return model

def process_image(image):
    
    transformers = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    tensor_img = transformers(image)

    return tensor_img

def predict(image_path, model, topk):
    model.to(device)

    img = Image.open(image_path)
    img_torch = process_image(img)
    
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()
    img_torch = img_torch.to(device)

    with torch.no_grad():
        output = model.forward(img_torch)

    probability = F.softmax(output.data, dim=1)

    return probability.topk(topk)

if args.category_names:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
else:
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        pass

#loading model from checkpoint provided
model = load_model(args.load_dir)



number_classes = args.top_k
image_path= args.image_dir

# calculating probabilities and classes

to_parse = predict(image_path, model, number_classes)
    
probabilities = to_parse[0][0].cpu().numpy()


mapping = {val: key for key, val in
                model.class_to_idx.items()
                }


classes = to_parse[1][0].cpu().numpy()
classes = [mapping [item] for item in classes]
classes = [cat_to_name[str(index)] for index in classes]

for l in range(number_classes):
     print(". The Predicting: {} has probability: {:.2f}%.".format(l+1, classes[l], probabilities [l]*100))