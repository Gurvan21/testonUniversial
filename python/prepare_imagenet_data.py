import os

import torch
from PIL import Image
from torchvision import transforms


def get_preprocess(img_size=(256, 256), crop_size=(224, 224)):
    # Images are kept in [0, 1] RGB space; the ImageNet mean/std normalization
    # is done inside the model (see demo.py), so perturbations live in pixel space.
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
    ])


def preprocess_image_batch(image_paths, img_size=(256, 256), crop_size=(224, 224)):
    """
    :param image_paths: list of image paths
    :return: tensor of size Nx3xHxW with values in [0, 1]
    """
    preprocess = get_preprocess(img_size, crop_size)
    return torch.stack([preprocess(Image.open(p).convert('RGB')) for p in image_paths])


def create_imagenet_data(path_train_imagenet, num_imgs_per_class=10, num_classes=1000):
    """
    Load num_imgs_per_class images from each class folder of the ImageNet training set.

    :param path_train_imagenet: path of the ImageNet training set (one sub-folder per class)
    :return: tensor of size Mx3x224x224 with values in [0, 1]
    """
    # Sort the directories in alphabetical order (same as synset_words.txt)
    dirs = sorted(d.path for d in os.scandir(path_train_imagenet) if d.is_dir())[:num_classes]

    image_paths = []
    for d in dirs:
        files = sorted(f for f in os.listdir(d) if not f.startswith('.'))[:num_imgs_per_class]
        image_paths += [os.path.join(d, f) for f in files]

    preprocess = get_preprocess()
    data = torch.empty(len(image_paths), 3, 224, 224)

    # Load images and pre-process
    for it, path_img in enumerate(image_paths):
        print('Processing image number ', it)
        data[it] = preprocess(Image.open(path_img).convert('RGB'))

    return data
