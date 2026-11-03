import argparse
import os

import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from torchvision import transforms

from prepare_imagenet_data import preprocess_image_batch, create_imagenet_data
from universal_pert import universal_perturbation, universal_perturbation_sgd


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_model(name, device):
    # Pre-trained torchvision model, with the ImageNet normalization included so
    # that the model takes [0, 1] RGB images as input.
    weights = torchvision.models.get_model_weights(name).DEFAULT
    net = torchvision.models.get_model(name, weights=weights)
    model = nn.Sequential(
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        net,
    )
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    return model, weights.meta['categories']


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description='Universal adversarial perturbations (PyTorch)')
    parser.add_argument('-i', '--test_image', default='data/test_img.png')
    parser.add_argument('-t', '--training_path', default='/datasets2/ILSVRC2012/train')
    parser.add_argument('-m', '--model', default='resnet50', help='any torchvision classification model (resnet50, vgg16, googlenet, ...)')
    parser.add_argument('--xi', type=float, default=10, help='l_inf magnitude of the perturbation, on the [0, 255] scale')
    parser.add_argument('--delta', type=float, default=0.2, help='desired fooling rate = 1 - delta')
    parser.add_argument('--num_imgs_per_class', type=int, default=10)
    parser.add_argument('--cache', default='data/imagenet_data.pt',
                        help='file where the pre-processed training images are cached (one per dataset!)')
    parser.add_argument('--max_iter_df', type=int, default=10,
                        help='maximum number of DeepFool iterations; the update is discarded if DeepFool does not converge')
    parser.add_argument('-o', '--perturbation', default=None,
                        help='file of the universal perturbation, computed if it does not exist (default: data/universal_<model>.pt)')
    parser.add_argument('--checkpoint', default=None,
                        help='file where the computation is saved regularly, to resume it if interrupted '
                             '(default: data/checkpoint_<model>.pt, or <perturbation>_checkpoint.pt with -o)')
    parser.add_argument('--algo', choices=['deepfool', 'sgd'], default='deepfool',
                        help='deepfool: algorithm of the paper (aggregation of DeepFool steps); '
                             'sgd: gradient ascent on the loss averaged over images (and noise samples)')
    parser.add_argument('--noise_std', type=float, default=0,
                        help='EOT against a Gaussian noise defense of this standard deviation, on the [0, 255] scale (0 = no EOT)')
    parser.add_argument('--eot_samples', type=int, default=4, help='number of noise samples per image for EOT')
    parser.add_argument('--max_passes', type=int, default=None,
                        help='deepfool: maximum number of passes (default: until the fooling rate reaches 1 - delta)')
    parser.add_argument('--epochs', type=int, default=10, help='sgd: maximum number of passes over the images')
    parser.add_argument('--sgd_batch_size', type=int, default=32,
                        help='sgd: number of images per step (x eot_samples images on the GPU: lower it on small GPUs)')
    parser.add_argument('--step', type=float, default=None, help='sgd: step size on the [0, 255] scale (default xi / 10)')
    parser.add_argument('--target', default=None,
                        help='sgd: targeted attack towards this class (index or name, e.g. 815 or "spider web")')
    parser.add_argument('--loss_clip', type=float, default=None,
                        help='sgd: maximum of the per-image loss (Shafahi et al.), e.g. 5 (default: no clipping)')
    args = parser.parse_args()

    device = get_device()
    print('>> Using device', device)

    print('>> Loading model', args.model)
    model, labels = load_model(args.model, device)

    file_perturbation = args.perturbation or os.path.join('data', 'universal_' + args.model + '.pt')
    os.makedirs(os.path.dirname(file_perturbation) or '.', exist_ok=True)

    if not os.path.isfile(file_perturbation):

        # Load/Create data
        datafile = args.cache
        os.makedirs(os.path.dirname(datafile) or '.', exist_ok=True)
        if not os.path.isfile(datafile):
            print('>> Creating pre-processed imagenet data...')
            X = create_imagenet_data(args.training_path, num_imgs_per_class=args.num_imgs_per_class)

            # Save the pre-processed images
            # Caution: This can take a lot of space. Comment this part to discard saving.
            print('>> Saving the pre-processed imagenet data')
            torch.save(X, datafile)

        else:
            print('>> Pre-processed imagenet data detected')
            X = torch.load(datafile)

        # Running universal perturbation (resumed from the checkpoint if the computation was interrupted)
        if args.checkpoint:
            file_checkpoint = args.checkpoint
        elif args.perturbation:
            file_checkpoint = os.path.splitext(args.perturbation)[0] + '_checkpoint.pt'
        else:
            file_checkpoint = os.path.join('data', 'checkpoint_' + args.model + '.pt')
        if args.noise_std > 0:
            print('>> EOT against a Gaussian noise of std %.1f/255, %d samples' % (args.noise_std, args.eot_samples))
        if args.algo == 'deepfool':
            v = universal_perturbation(X, model, delta=args.delta, xi=args.xi / 255, max_iter_df=args.max_iter_df,
                                       max_iter_uni=args.max_passes or float('inf'), checkpoint=file_checkpoint, noise_std=args.noise_std / 255,
                                       eot_samples=args.eot_samples)
        else:
            target = None
            if args.target is not None:
                target = int(args.target) if args.target.isdigit() else labels.index(args.target)
                print('>> Targeted attack towards class %d (%s)' % (target, labels[target]))
            v = universal_perturbation_sgd(X, model, delta=args.delta, xi=args.xi / 255, noise_std=args.noise_std / 255,
                                           eot_samples=args.eot_samples, epochs=args.epochs,
                                           batch_size=args.sgd_batch_size,
                                           step=None if args.step is None else args.step / 255,
                                           loss_clip=args.loss_clip, target=target, checkpoint=file_checkpoint)

        # Saving the universal perturbation; the checkpoint is no longer needed
        torch.save(v.cpu(), file_perturbation)
        if os.path.isfile(file_checkpoint):
            os.remove(file_checkpoint)

    else:
        print('>> Found a pre-computed universal perturbation! Retrieving it from', file_perturbation)
        v = torch.load(file_perturbation)

    v = v.to(device)

    print('>> Testing the universal perturbation on an image')

    # Test the perturbation on the image
    image_original = preprocess_image_batch([args.test_image]).to(device)

    # Clip the perturbed image to make sure it is a valid image
    image_perturbed = (image_original + v).clamp(0, 1)

    with torch.no_grad():
        str_label_original = labels[model(image_original).argmax().item()]
        str_label_perturbed = labels[model(image_perturbed).argmax().item()]

    # Show original and perturbed image
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image_original[0].permute(1, 2, 0).cpu().numpy())
    plt.title(str_label_original)

    plt.subplot(1, 2, 2)
    plt.imshow(image_perturbed[0].permute(1, 2, 0).cpu().numpy())
    plt.title(str_label_perturbed)

    plt.show()
