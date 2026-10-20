import os
import sys

import matplotlib.pyplot as plt
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# deepfool.py est dans le dossier parent (universal/python)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from deepfool import deepfool

RESULTS = os.path.join(HERE, 'results')
os.makedirs(RESULTS, exist_ok=True)

net = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

# Switch to evaluation mode
net.eval()

im_orig = Image.open(os.path.join(HERE, 'test_im1.jpg')).convert('RGB')

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


# Remove the mean
im = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean,
                         std=std)])(im_orig)

r, loop_i, label_orig, label_pert, pert_image = deepfool(im, net)

labels = open(os.path.join(HERE, 'synset_words.txt'), 'r').read().split('\n')

str_label_orig = labels[int(label_orig)].split(' ', 1)[1].split(',')[0]
str_label_pert = labels[int(label_pert)].split(' ', 1)[1].split(',')[0]


# Undo the normalization, then clip to the valid [0, 1] range of ToTensor
tf = transforms.Compose([transforms.Normalize(mean=[0, 0, 0], std=[1 / s for s in std]),
                         transforms.Normalize(mean=[-m for m in mean], std=[1, 1, 1]),
                         transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),
                         transforms.ToPILImage()])

report = ["Model: resnet34, image: test_im1.jpg",
          "Original label = " + str_label_orig,
          "Perturbed label = " + str_label_pert,
          "Iterations = %d" % loop_i,
          "||r|| / ||x|| = %.4f" % (torch.from_numpy(r).norm() / im.norm())]
print("\n".join(report))
with open(os.path.join(RESULTS, 'test_deepfool.txt'), 'w') as f:
    f.write("\n".join(report) + "\n")

perturbed = tf(pert_image.cpu()[0])
perturbed.save(os.path.join(RESULTS, 'perturbed_image.png'))

# Perturbation in pixel space (undo the std scaling), stretched to [0, 1] so it is visible
r_pixels = torch.from_numpy(r[0]) * torch.tensor(std)[:, None, None]
r_visible = (r_pixels - r_pixels.min()) / (r_pixels.max() - r_pixels.min())

fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
axes[0].imshow(r_visible.permute(1, 2, 0))
axes[0].set_title("Perturbation [scaled]")
axes[1].imshow(perturbed)
axes[1].set_title("Perturbed: " + str_label_pert)
for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, 'deepfool_resnet34.png'), dpi=100)
print("Results saved in", RESULTS)
plt.show()
