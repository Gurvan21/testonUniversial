import argparse
import os

import torch

from demo import get_device, load_model
from evaluate import ImageFolderList, list_val_set


@torch.no_grad()
def probabilities(model, image_paths, v, batch_size=100, num_workers=4):
    # Softmax probabilities of the images without and with the perturbation (each image is loaded once)
    device = next(model.parameters()).device
    loader = torch.utils.data.DataLoader(ImageFolderList(image_paths), batch_size=batch_size, num_workers=num_workers)
    p_clean, p_pert = [], []
    for it, batch in enumerate(loader):
        print('   batch %d / %d' % (it + 1, len(loader)), end='\r')
        batch = batch.to(device)
        p_clean.append(model(batch).softmax(dim=1).cpu())
        p_pert.append(model((batch + v).clamp(0, 1)).softmax(dim=1).cpu())
    print()
    return torch.cat(p_clean), torch.cat(p_pert)


def rank_of(probs, c):
    # Rank of class c in the predictions of each image (1 = predicted class)
    return (probs > probs[:, c:c + 1]).sum(dim=1) + 1


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Statistics of one class (by default the dominant label) for a universal perturbation')
    parser.add_argument('-m', '--model', default='resnet50')
    parser.add_argument('-p', '--perturbation', default=None, help='default: data/universal_<model>.pt')
    parser.add_argument('-v', '--val_path', default='data/imagenette2-320/val')
    parser.add_argument('-c', '--class_name', default=None, help='e.g. "spider web" (default: the class most predicted after perturbation)')
    parser.add_argument('--synsets', default='data/synset_words.txt')
    parser.add_argument('--max_per_class', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--out', default=None, help='file where the report is saved')
    args = parser.parse_args()

    report = []

    def log(line=''):
        print(line)
        report.append(line)

    device = get_device()
    model, categories = load_model(args.model, device)
    file_perturbation = args.perturbation or os.path.join('data', 'universal_' + args.model + '.pt')
    v = torch.load(file_perturbation).to(device)
    synsets = [line.split(' ', 1)[0] for line in open(args.synsets).read().splitlines() if line]

    image_paths, labels = list_val_set(args.val_path, args.synsets, args.max_per_class)
    p_clean, p_pert = probabilities(model, image_paths, v, batch_size=args.batch_size)
    pred_clean, pred_pert = p_clean.argmax(dim=1), p_pert.argmax(dim=1)
    n = len(image_paths)

    if args.class_name:
        c = categories.index(args.class_name)
    else:
        fooled = pred_pert[pred_pert != pred_clean]
        c = fooled.mode().values.item() if len(fooled) else pred_pert.mode().values.item()

    log('>> Model: %s, perturbation: %s' % (args.model, file_perturbation))
    log('>> Images: %s (%d images)' % (args.val_path, n))
    log('>> Class: "%s" (index %d, synset %s)' % (categories[c], c, synsets[c]))

    # 1. The class in the dataset and on clean images
    is_c = labels == c
    log()
    log('1. Images propres')
    log('   images de cette classe dans le jeu :       %d sur %d (%.2f %%)' % (is_c.sum(), n, 100 * is_c.float().mean()))
    if is_c.any():
        log('   rappel (images de la classe bien reconnues) : %.1f %%' % (100 * (pred_clean[is_c] == c).float().mean()))
    pc = (pred_clean == c).sum().item()
    log('   images prédites dans cette classe :        %d (%.2f %%)' % (pc, 100 * pc / n))
    if pc:
        log('   précision (prédictions correctes) :        %.1f %%' % (100 * (labels[pred_clean == c] == c).float().mean()))
    log('   probabilité moyenne de la classe :         %.4f (uniforme : %.4f)' % (p_clean[:, c].mean(), 1 / len(categories)))
    log('   rang médian de la classe :                 %d / %d' % (rank_of(p_clean, c).median(), len(categories)))

    # 2. The class after perturbation
    pp = (pred_pert == c).sum().item()
    log()
    log('2. Images perturbées')
    log('   images prédites dans cette classe :        %d (%.1f %%)' % (pp, 100 * pp / n))
    log('   probabilité moyenne de la classe :         %.4f' % p_pert[:, c].mean())
    log('   rang médian de la classe :                 %d / %d' % (rank_of(p_pert, c).median(), len(categories)))
    if is_c.any():
        log('   images de la classe encore reconnues :     %.1f %%' % (100 * (pred_pert[is_c] == c).float().mean()))

    # 3. Which true classes resist (are not sent to the class)
    sent = torch.zeros(len(categories)).index_add_(0, labels, (pred_pert == c).float())
    count = torch.zeros(len(categories)).index_add_(0, labels, torch.ones(n))
    present = (count > 0).nonzero().flatten()
    frac = sent[present] / count[present]
    order = frac.argsort()
    log()
    log('3. Proportion des images envoyées vers "%s", selon leur vraie classe' % categories[c])
    log('   classes qui résistent le mieux :')
    for i in order[:5]:
        log('     %-30s %5.1f %%' % (categories[present[i]], 100 * frac[i]))
    log('   classes les plus envoyées :')
    for i in order.flip(0)[:5]:
        log('     %-30s %5.1f %%' % (categories[present[i]], 100 * frac[i]))

    # 4. The perturbation alone, on a uniform gray image, more and more amplified
    log()
    log('4. La perturbation seule, sur une image grise uniforme')
    with torch.no_grad():
        for scale in [0, 1, 2, 5, 10]:
            img = (0.5 + scale * v).clamp(0, 1)
            p = model(img).softmax(dim=1)[0]
            log('   gris + %2d x v : %-25s (p = %.2f)' % (scale, categories[p.argmax()], p.max()))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w') as f:
            f.write('\n'.join(report) + '\n')
        print('>> Report saved in', args.out)
