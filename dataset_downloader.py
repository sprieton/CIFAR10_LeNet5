"""
    Code to download the CIFAR10 dataset needed to run the notebook
    this code only needs to run once, downloads the whole dataset and gets
    the images only of cats and dogs
"""
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os
import shutil

# Clases we want to work of the CIFAR10
classes = ('bird', 'cat')
label_map = {'bird': 2, 'cat': 3}

# Folder to save the dataset
output_dir = './data'

# Transformation to save the images
transform = transforms.ToTensor()

# Downaload the dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

os.makedirs(output_dir, exist_ok=True)

for split_name, dataset in [('train', trainset), ('test', testset)]:
    # Create the structure of folders
    for cls in classes:
        os.makedirs(f'{output_dir}/{split_name}/{cls}', exist_ok=True)
    
    # Save each image on each folder
    for i, (img, label) in enumerate(dataset):
        if label in label_map.values():
            class_name = [k for k, v in label_map.items() if v == label][0]
            img.save(f'{output_dir}/{split_name}/{class_name}/{i}.png')

print(f"✅ Dog and Cat images from CIFAR10 saved on {output_dir}")

# remove the full dataset file for simplicity
cifar_dir = './data/cifar-10-batches-py'
cifar_tar = './data/cifar-10-python.tar.gz'

if os.path.exists(cifar_dir):
    shutil.rmtree(cifar_dir)
    print(f"🗑️ Removed folder: {cifar_dir}")

if os.path.exists(cifar_tar):
    os.remove(cifar_tar)
    print(f"🗑️ Removed .tar: {cifar_tar}")