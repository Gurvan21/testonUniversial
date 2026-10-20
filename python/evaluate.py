import argparse
import os

import matplotlib.pyplot as plt
import torch
from PIL import Image

from demo import get_device, load_model
from prepare_imagenet_data import get_preprocess, preprocess_image_batch


class ImageFolderList(torch.utils.data.Dataset):
    # Images loaded on the fly (the whole validation set would not fit in memory)
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.preprocess = get_preprocess()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):
        return self.preprocess(Image.open(self.image_paths[i]).convert('RGB'))


def list_val_set(path_val, synset_file, max_per_class=None):
    """
    List the validation images (one sub-folder per class, named by its synset, e.g. n01440764).

    :return: list of image paths, and the true ImageNet labels (indices in 0..999)
    """
    synsets = [line.split(' ', 1)[0] for line in open(synset_file).read().splitlines() if line]
    image_paths, labels = [], []
    for d in sorted(os.listdir(path_val)):
        files = sorted(f for f in os.listdir(os.path.join(path_val, d)) if not f.startswith('.'))[:max_per_class]
        image_paths += [os.path.join(path_val, d, f) for f in files]
        labels += [synsets.index(d)] * len(files)
    return image_paths, torch.tensor(labels)


@torch.no_grad()
def predict(model, data, v=None, batch_size=50):
    # Predicted labels of a tensor of images, the perturbed images being clipped to [0, 1] (as in demo.py)
    device = next(model.parameters()).device
    preds = []
    for m in range(0, len(data), batch_size):
        batch = data[m:m + batch_size].to(device)
        if v is not None:
            batch = (batch + v).clamp(0, 1)
        preds.append(model(batch).argmax(dim=1).cpu())
    return torch.cat(preds)


@torch.no_grad()
def predict_paths(model, image_paths, perturbations, batch_size=100, num_workers=4):
    # Predicted labels of the images without perturbation and with each perturbation (each image is loaded once)
    device = next(model.parameters()).device
    loader = torch.utils.data.DataLoader(ImageFolderList(image_paths), batch_size=batch_size, num_workers=num_workers)
    preds = [[] for _ in range(len(perturbations) + 1)]
    for it, batch in enumerate(loader):
        print('   batch %d / %d' % (it + 1, len(loader)), end='\r')
        batch = batch.to(device)
        preds[0].append(model(batch).argmax(dim=1).cpu())
        for p, v in enumerate(perturbations):
            preds[p + 1].append(model((batch + v).clamp(0, 1)).argmax(dim=1).cpu())
    print()
    return [torch.cat(p) for p in preds]


def show_examples(image_paths, v, model, categories, n=6):
    # Original / perturbed images with the predicted labels, and the perturbation itself
    idx = torch.randperm(len(image_paths))[:n]
    images = preprocess_image_batch([image_paths[i] for i in idx])
    perturbed = (images + v.cpu()).clamp(0, 1)
    pred_orig, pred_pert = predict(model, images), predict(model, perturbed)

    fig, axes = plt.subplots(2, n + 1, figsize=(2.6 * (n + 1), 6))
    v_visible = (v[0].cpu() - v.min().cpu()) / (v.max() - v.min()).cpu()
    axes[0, 0].imshow(v_visible.permute(1, 2, 0))
    axes[0, 0].set_title('Perturbation\n[scaled]')
    axes[1, 0].text(0.5, 0.5, '(même v pour\ntoutes les images)', ha='center', va='center')
    for j in range(n):
        axes[0, j + 1].imshow(images[j].permute(1, 2, 0))
        axes[0, j + 1].set_title(categories[pred_orig[j]][:20], fontsize=9)
        axes[1, j + 1].imshow(perturbed[j].permute(1, 2, 0))
        color = 'red' if pred_pert[j] != pred_orig[j] else 'green'
        axes[1, j + 1].set_title(categories[pred_pert[j]][:20], fontsize=9, color=color)
    for ax in axes.flat:
        ax.axis('off')
    plt.suptitle('Haut : images originales — Bas : images + perturbation universelle (rouge = trompé)')
    plt.tight_layout()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate a universal perturbation on held-out images')
    parser.add_argument('-m', '--model', default='resnet18', help='model attacked (torchvision name)')
    parser.add_argument('-p', '--perturbation', default=None, help='default: data/universal_<model>.pt')
    parser.add_argument('-v', '--val_path', default='data/imagenette2-320/val')
    parser.add_argument('--synsets', default='data/synset_words.txt')
    parser.add_argument('--max_per_class', type=int, default=None, help='limit the number of validation images per class')
    parser.add_argument('--batch_size', type=int, default=100, help='lower it if the GPU runs out of memory')
    parser.add_argument('--out', default=None, help='folder where the results (text + figure) are saved')
    args = parser.parse_args()

    # Fixed seed: same shuffled perturbation and same examples at each run (reproducible results)
    torch.manual_seed(0)

    report = []

    def log(line=''):
        # Print, and keep the line for the results file
        print(line)
        report.append(line)

    device = get_device()
    model, categories = load_model(args.model, device)
    file_perturbation = args.perturbation or os.path.join('data', 'universal_' + args.model + '.pt')
    v = torch.load(file_perturbation).to(device)
    log('>> Model: %s, perturbation: %s' % (args.model, file_perturbation))
    log('>> Perturbation: l_inf = %.2f / 255, l_2 = %.2f' % (v.abs().max() * 255, v.norm()))

    log('>> Validation images: ' + args.val_path)
    image_paths, labels = list_val_set(args.val_path, args.synsets, args.max_per_class)
    log('>> %d images' % len(image_paths))

    # Random perturbation with exactly the same norms: the pixels of v shuffled
    v_random = v.flatten()[torch.randperm(v.numel(), device=device)].view_as(v)

    pred_clean, pred_univ, pred_rand = predict_paths(model, image_paths, [v, v_random], batch_size=args.batch_size)

    log()
    log('                       précision   fooling rate')
    log('  images propres        %6.1f %%         -' % (100 * (pred_clean == labels).float().mean()))
    log('  + v mélangé (témoin)  %6.1f %%     %6.1f %%' % (100 * (pred_rand == labels).float().mean(),
                                                         100 * (pred_rand != pred_clean).float().mean()))
    log('  + pert. universelle   %6.1f %%     %6.1f %%' % (100 * (pred_univ == labels).float().mean(),
                                                         100 * (pred_univ != pred_clean).float().mean()))

    # Labels to which the universal perturbation sends the images (often a few "dominant" labels)
    fooled = pred_univ[pred_univ != pred_clean]
    if len(fooled):
        values, counts = fooled.unique(return_counts=True)
        log('\n>> Classes les plus prédites après perturbation :')
        for c, n in sorted(zip(counts.tolist(), values.tolist()), reverse=True)[:5]:
            log('   %-30s %5.1f %% des images trompées' % (categories[n], 100 * c / len(fooled)))

    show_examples(image_paths, v, model, categories)

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, 'eval_%s.txt' % args.model), 'w') as f:
            f.write('\n'.join(report) + '\n')
        plt.savefig(os.path.join(args.out, 'examples_%s.png' % args.model), dpi=100)
        print('>> Results saved in', args.out)

    plt.show()
