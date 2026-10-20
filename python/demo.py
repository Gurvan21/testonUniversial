import argparse
import os

import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from torchvision import transforms

from prepare_imagenet_data import preprocess_image_batch, create_imagenet_data
from universal_pert import universal_perturbation


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
    parser.add_argument('--checkpoint', default=None,
                        help='file where the computation is saved regularly, to resume it if interrupted (default: data/checkpoint_<model>.pt)')
    args = parser.parse_args()

    device = get_device()
    print('>> Using device', device)

    print('>> Loading model', args.model)
    model, labels = load_model(args.model, device)

    os.makedirs('data', exist_ok=True)
    file_perturbation = os.path.join('data', 'universal_' + args.model + '.pt')

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
        file_checkpoint = args.checkpoint or os.path.join('data', 'checkpoint_' + args.model + '.pt')
        v = universal_perturbation(X, model, delta=args.delta, xi=args.xi / 255, max_iter_df=args.max_iter_df,
                                  checkpoint=file_checkpoint)

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
