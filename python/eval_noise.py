import argparse
import os

import torch

from demo import get_device, load_model
from evaluate import ImageFolderList, list_val_set


@torch.no_grad()
def predict_noisy(model, image_paths, v, sigmas, batch_size=50, num_workers=4):
    """
    Predicted labels of the clean and perturbed images, with an independent Gaussian noise of standard deviation sigma
    added to each image before the prediction (a "defense"), for each sigma. Each image is loaded once.
    'clean2' is a second noise sample of the clean images: the predictions that the noise alone changes.
    """
    device = next(model.parameters()).device
    loader = torch.utils.data.DataLoader(ImageFolderList(image_paths), batch_size=batch_size, num_workers=num_workers)
    preds = {(s, k): [] for s in sigmas for k in ('clean', 'clean2', 'pert')}
    for it, batch in enumerate(loader):
        print('   batch %d / %d' % (it + 1, len(loader)), end='\r')
        batch = batch.to(device)
        for s in sigmas:
            for k, x in (('clean', batch), ('clean2', batch), ('pert', batch + v)):
                noisy = (x + s * torch.randn_like(x)).clamp(0, 1)
                preds[(s, k)].append(model(noisy).argmax(dim=1).cpu())
    print()
    return {key: torch.cat(p) for key, p in preds.items()}


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Gaussian noise added to the input as a defense against a universal perturbation')
    parser.add_argument('-m', '--model', default='resnet50')
    parser.add_argument('-p', '--perturbation', default=None, help='default: data/universal_<model>.pt')
    parser.add_argument('-v', '--val_path', default='data/imagenette2-320/val')
    parser.add_argument('--sigmas', default='0,2,5,10,20,40', help='standard deviations of the noise, on the [0, 255] scale')
    parser.add_argument('--synsets', default='data/synset_words.txt')
    parser.add_argument('--max_per_class', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--out', default=None, help='file where the results are saved')
    args = parser.parse_args()

    torch.manual_seed(0)
    device = get_device()
    model, categories = load_model(args.model, device)
    file_perturbation = args.perturbation or os.path.join('data', 'universal_' + args.model + '.pt')
    v = torch.load(file_perturbation).to(device)
    sigmas = [float(s) / 255 for s in args.sigmas.split(',')]

    image_paths, labels = list_val_set(args.val_path, args.synsets, args.max_per_class)
    preds = predict_noisy(model, image_paths, v, sigmas, batch_size=args.batch_size)

    # Reference prediction: clean image, no noise (as in evaluate.py)
    ref = preds[(0.0, 'clean')] if 0.0 in sigmas else None

    # The dominant label of the perturbation without defense
    pert0 = preds[(sigmas[0], 'pert')]
    dominant = pert0[pert0 != preds[(sigmas[0], 'clean')]].mode().values.item()

    lines = ['>> Model: %s, perturbation: %s' % (args.model, file_perturbation),
             '>> Images: %s (%d images)' % (args.val_path, len(image_paths)),
             '>> Dominant label of the perturbation: %s' % categories[dominant],
             '',
             'Colonnes (en % des images) :',
             '  A = photos normales + bruit, bien reconnues        (coût de la défense : doit rester haut)',
             '  B = photos attaquées + bruit, bien reconnues       (efficacité : doit remonter vers A)',
             '  C = réponse différente entre normale+bruit et attaquée+bruit  (fooling rate)',
             '  D = réponse différente entre normale sans bruit et attaquée+bruit',
             '  P = plancher de C : réponse différente entre deux tirages du bruit sur la photo normale (v = 0)',
             '  E = photos attaquées + bruit classées "%s"   (la classe dominante est-elle encore vue ?)' % categories[dominant],
             '',
             '  bruit σ (/255)      A          B          C          P          D          E']
    for s in sigmas:
        c, p = preds[(s, 'clean')], preds[(s, 'pert')]
        lines.append('  %8.0f       %6.1f %%   %6.1f %%   %6.1f %%   %6.1f %%   %s   %6.1f %%' % (
            s * 255, 100 * (c == labels).float().mean(), 100 * (p == labels).float().mean(),
            100 * (p != c).float().mean(), 100 * (preds[(s, 'clean2')] != c).float().mean(),
            ('%6.1f %%' % (100 * (p != ref).float().mean())) if ref is not None else '   -   ',
            100 * (p == dominant).float().mean()))
    print('\n'.join(lines))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print('>> Results saved in', args.out)
