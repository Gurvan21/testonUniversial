import argparse
import os

import matplotlib.pyplot as plt
import torch

from demo import get_device, load_model
from prepare_imagenet_data import preprocess_image_batch


def to_visible(v):
    # Stretch the perturbation to [0, 1] so that it is visible (its real values are in [-xi, xi])
    return ((v - v.min()) / (v.max() - v.min())).permute(1, 2, 0).cpu()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Show a universal perturbation, amplified')
    parser.add_argument('-m', '--model', default='resnet50')
    parser.add_argument('-p', '--perturbation', default=None, help='default: data/universal_<model>.pt')
    parser.add_argument('-i', '--image', default='data/test_img.png')
    parser.add_argument('--out', default=None, help='file where the figure is saved (png)')
    args = parser.parse_args()

    device = get_device()
    model, categories = load_model(args.model, device)
    v = torch.load(args.perturbation or os.path.join('data', 'universal_' + args.model + '.pt')).to(device)
    image = preprocess_image_batch([args.image]).to(device)
    xi = v.abs().max().item()

    # First figure: the perturbation itself, stretched to [0, 1], and zooms on a corner and on the center
    fig = plt.figure(figsize=(15, 10))
    grid = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(grid[:, :2])
    ax.imshow(to_visible(v[0]))
    ax.add_patch(plt.Rectangle((0, 0), 80, 80, fill=False, edgecolor='white', linewidth=2))
    ax.add_patch(plt.Rectangle((72, 72), 80, 80, fill=False, edgecolor='black', linewidth=2))
    ax.set_title('Perturbation universelle de %s\n(amplifiée : valeurs réelles dans ±%.0f/255)' % (args.model, xi * 255))
    for row, (y, x, name) in enumerate([(0, 0, 'coin (cadre blanc)'), (72, 72, 'centre (cadre noir)')]):
        zoom = fig.add_subplot(grid[row, 2])
        zoom.imshow(to_visible(v[0])[y:y + 80, x:x + 80], interpolation='nearest')
        zoom.set_title('Zoom : ' + name)
    for a in fig.axes:
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout()

    # Second figure: the image with the perturbation amplified more and more
    scales = [0, 1, 3, 5, 10]
    fig2, axes = plt.subplots(1, len(scales), figsize=(3.4 * len(scales), 4))
    with torch.no_grad():
        for ax, scale in zip(axes, scales):
            img = (image + scale * v).clamp(0, 1)
            p = model(img).softmax(dim=1)[0]
            ax.imshow(img[0].permute(1, 2, 0).cpu())
            label = 'image originale' if scale == 0 else 'image + %d × v' % scale
            ax.set_title('%s\n→ %s (p = %.2f)' % (label, categories[p.argmax()], p.max()), fontsize=10,
                         color='black' if scale == 0 else 'red')
            ax.axis('off')
    fig2.suptitle("Ajout de la perturbation, de plus en plus amplifiée (× 1 = la vraie perturbation, invisible)")
    fig2.tight_layout(rect=[0, 0, 1, 0.9])

    if args.out:
        base = os.path.splitext(args.out)[0]
        fig.savefig(base + '.png', dpi=90)
        fig2.savefig(base + '_amplification.png', dpi=90)
        print('>> Figures saved in', base + '.png', 'and', base + '_amplification.png')
    plt.show()
