import os

import numpy as np
import torch

from deepfool import deepfool


def proj_lp(v, xi, p):

    # Project on the lp ball centered at 0 and of radius xi

    # SUPPORTS only p = 2 and p = Inf for now
    if p == 2:
        v = v * min(1, xi / (v.norm().item() + 1e-12))
    elif p == np.inf:
        v = v.clamp(-xi, xi)
    else:
        raise ValueError('Values of p different from 2 and Inf are currently not supported...')

    return v


def add_noise(x, noise_std):
    # Gaussian noise of standard deviation noise_std (on the [0, 1] scale) added to the images, then clipped to [0, 1]:
    # the random "defense" that the attacker takes into account with Expectation Over Transformation (EOT)
    if noise_std == 0:
        return x
    return (x + noise_std * torch.randn_like(x)).clamp(0, 1)


@torch.no_grad()
def predict_labels(dataset, model, v=None, batch_size=100, noise_std=0):
    # Compute the estimated labels in batches (with an independent noise for each image if noise_std > 0)
    device = next(model.parameters()).device
    labels = []
    for m in range(0, dataset.shape[0], batch_size):
        batch = dataset[m:m + batch_size].to(device)
        if v is not None:
            batch = batch + v
        labels.append(model(add_noise(batch, noise_std)).argmax(dim=1).cpu())
    return torch.cat(labels)


def universal_perturbation(dataset, model, delta=0.2, max_iter_uni=np.inf, xi=10/255, p=np.inf, num_classes=10, overshoot=0.02, max_iter_df=10, batch_size=100, checkpoint=None, checkpoint_every=200,
                           noise_std=0, eot_samples=1):
    """
    :param dataset: Images tensor of size Mx3xHxW, values in [0, 1] (M: number of images)

    :param model: PyTorch classifier in eval mode (input: images, output: logits, i.e. values BEFORE softmax).

    :param delta: controls the desired fooling rate (default = 80% fooling rate)

    :param max_iter_uni: optional other termination criterion (maximum number of iteration, default = np.inf)

    :param xi: controls the l_p magnitude of the perturbation (default = 10/255, i.e. 10 on the [0, 255] pixel scale)

    :param p: norm to be used (FOR NOW, ONLY p = 2, and p = np.inf ARE ACCEPTED!) (default = np.inf)

    :param num_classes: num_classes (limits the number of classes to test against, by default = 10)

    :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).

    :param max_iter_df: maximum number of iterations for deepfool (default = 10)

    :param batch_size: batch size used to compute the fooling rate (default = 100)

    :param checkpoint: optional file where the state of the computation (v, pass, position in the pass) is saved
        every checkpoint_every images and at the end of each pass. If it exists, the computation resumes from it.

    :param checkpoint_every: number of images between two checkpoints (default = 200)

    :param noise_std: if > 0, the perturbation is computed against a defense that adds a Gaussian noise of this standard
        deviation (on the [0, 1] scale) to each image, with Expectation Over Transformation (EOT): an image counts as
        fooled if most of eot_samples noisy versions are fooled, DeepFool is run on eot_samples noisy versions of x + v
        and the steps are averaged, and the fooling rate is measured with the noise (default = 0, no defense)

    :param eot_samples: number of noise samples used for EOT (default = 1)

    :return: the universal perturbation, tensor of size 1x3xHxW.
    """

    device = next(model.parameters()).device
    v = torch.zeros((1,) + tuple(dataset.shape[1:]), device=device)
    fooling_rate = 0.0
    num_images = dataset.shape[0]  # The images should be stacked ALONG FIRST DIMENSION

    # Resume from the checkpoint, if any: pass number, shuffled order of the pass and position in it
    itr, start, order = 0, 0, None
    if checkpoint is not None and os.path.isfile(checkpoint):
        state = torch.load(checkpoint)
        v, itr, start, order, fooling_rate = state['v'].to(device), state['itr'], state['start'], state['order'], state['fooling_rate']
        print('>> Resuming from', checkpoint, ': pass', itr, ', image', start, ', fooling rate', fooling_rate)

    def save_checkpoint():
        if checkpoint is not None:
            # Write to a temporary file first, so that a kill during the save does not corrupt the checkpoint
            torch.save({'v': v.cpu(), 'itr': itr, 'start': start, 'order': order, 'fooling_rate': fooling_rate}, checkpoint + '.tmp')
            os.replace(checkpoint + '.tmp', checkpoint)

    est_labels_orig = predict_labels(dataset, model, batch_size=batch_size)

    while fooling_rate < 1 - delta and itr < max_iter_uni:
        # Shuffle the dataset (unless resuming in the middle of a pass)
        if order is None:
            order = torch.randperm(num_images).tolist()

        print('Starting pass number ', itr)
        num_deepfool, num_not_converged = 0, 0

        # Go through the data set and compute the perturbation increments sequentially
        for k_idx in range(start, num_images):
            k = order[k_idx]
            cur_img = dataset[k:(k + 1)].to(device)

            with torch.no_grad():
                label = model(cur_img).argmax()
                if noise_std == 0:
                    not_fooled = label == model(cur_img + v).argmax()
                else:
                    # EOT: fooled if the majority of the noisy versions are fooled
                    noisy = add_noise((cur_img + v).repeat(eot_samples, 1, 1, 1), noise_std)
                    not_fooled = (model(noisy).argmax(dim=1) == label).float().mean() >= 0.5

            if not_fooled:
                print('>> k = ', k_idx, ', pass #', itr)

                # Compute adversarial perturbation (EOT: on several noisy versions, and average the converged steps)
                steps = []
                for _ in range(eot_samples if noise_std > 0 else 1):
                    dr, iter, _, _, _ = deepfool(add_noise(cur_img + v, noise_std)[0], model, num_classes=num_classes,
                                                 overshoot=overshoot, max_iter=max_iter_df)
                    num_deepfool += 1

                    # Make sure it converged...
                    if iter >= max_iter_df - 1:
                        num_not_converged += 1
                    else:
                        steps.append(torch.from_numpy(dr).to(device))

                if steps:
                    v = v + torch.stack(steps).mean(dim=0)

                    # Project on l_p ball
                    v = proj_lp(v, xi, p)

            if (k_idx + 1) % checkpoint_every == 0:
                start = k_idx + 1
                save_checkpoint()
                print('>> Checkpoint at image %d / %d: DeepFool did not converge for %d of the %d images attacked'
                      % (start, num_images, num_not_converged, num_deepfool))

        itr = itr + 1
        start, order = 0, None

        # Compute the fooling rate on the perturbed dataset
        est_labels_pert = predict_labels(dataset, model, v=v, batch_size=batch_size, noise_std=noise_std)
        fooling_rate = (est_labels_pert != est_labels_orig).float().mean().item()
        print('FOOLING RATE = ', fooling_rate, '(with noise std %.1f/255)' % (noise_std * 255) if noise_std > 0 else '')
        # (counted since the start of the pass, or since the resumption)
        print('>> DeepFool did not converge (update discarded) for %d of the %d DeepFool runs' % (num_not_converged, num_deepfool))
        save_checkpoint()

    return v


def universal_perturbation_sgd(dataset, model, delta=0.2, xi=10/255, p=np.inf, noise_std=0, eot_samples=1, epochs=10,
                               batch_size=32, step=None, loss_clip=None, target=None, eval_batch_size=100,
                               checkpoint=None):
    """
    Gradient-based universal perturbation (as the "SGD" baseline of Xu & Singh, ICML 2024, and the SGD UAP of
    Shafahi et al., AAAI 2020), with Expectation Over Transformation against a Gaussian noise defense.

    At each step, the cross-entropy with the original labels is averaged over a batch of images AND over eot_samples
    noisy versions of each image, and v moves in the direction that increases it (sign of the gradient for p = Inf),
    then is projected on the l_p ball of radius xi. Unlike the DeepFool-based algorithm, it does not stop pushing an
    image once it is fooled, which gives margin to the perturbation.

    :param dataset: Images tensor of size Mx3xHxW, values in [0, 1]
    :param model: PyTorch classifier in eval mode (outputs logits)
    :param delta: stops when the fooling rate (measured with the noise) reaches 1 - delta, or after epochs passes
    :param xi, p: radius and norm of the perturbation (p = 2 or np.inf)
    :param noise_std: standard deviation of the Gaussian noise of the defense, on the [0, 1] scale (0 = no defense)
    :param eot_samples: number of noise samples per image at each step
    :param epochs: maximum number of passes over the dataset
    :param batch_size: number of images per step (each repeated eot_samples times)
    :param step: step size (default xi / 10), divided by 2 at half and by 4 at three quarters of the epochs
    :param loss_clip: optional maximum of the per-image loss, so that already fooled images stop dominating (Shafahi et al.)
    :param target: optional class index: targeted attack, v decreases the cross-entropy with this class instead of
        increasing the one with the original labels (the fooling rate is still measured against the original labels)
    :param checkpoint: optional file saved after each pass; if it exists, the computation resumes from it
    :return: the universal perturbation, tensor of size 1x3xHxW
    """
    device = next(model.parameters()).device
    step = xi / 10 if step is None else step
    num_images = dataset.shape[0]
    v = torch.zeros((1,) + tuple(dataset.shape[1:]), device=device)
    epoch, fooling_rate = 0, 0.0

    if checkpoint is not None and os.path.isfile(checkpoint):
        state = torch.load(checkpoint)
        v, epoch, fooling_rate = state['v'].to(device), state['epoch'], state['fooling_rate']
        print('>> Resuming from', checkpoint, ': epoch', epoch, ', fooling rate', fooling_rate)

    est_labels_orig = predict_labels(dataset, model, batch_size=eval_batch_size)
    m = eot_samples if noise_std > 0 else 1

    while fooling_rate < 1 - delta and epoch < epochs:
        print('Starting epoch ', epoch)
        # Smaller steps at the end, so that v settles instead of oscillating around the optimum
        cur_step = step * (1 if epoch < epochs / 2 else 0.5 if epoch < 3 * epochs / 4 else 0.25)
        order = torch.randperm(num_images)
        for b in range(0, num_images, batch_size):
            idx = order[b:b + batch_size]
            x = dataset[idx].to(device).repeat(m, 1, 1, 1)
            y = est_labels_orig[idx].to(device).repeat(m)

            v.requires_grad_(True)
            logits = model(add_noise(x + v, noise_std))
            if target is None:
                loss = torch.nn.functional.cross_entropy(logits, y, reduction='none')
            else:
                loss = -torch.nn.functional.cross_entropy(logits, torch.full_like(y, target), reduction='none')
            if loss_clip is not None:
                loss = loss.clamp(max=loss_clip)
            grad, = torch.autograd.grad(loss.mean(), v)

            with torch.no_grad():
                if p == np.inf:
                    v = v + cur_step * grad.sign()
                else:
                    v = v + cur_step * grad / (grad.norm() + 1e-12)
                v = proj_lp(v, xi, p)

        epoch += 1
        est_labels_pert = predict_labels(dataset, model, v=v, batch_size=eval_batch_size, noise_std=noise_std)
        fooling_rate = (est_labels_pert != est_labels_orig).float().mean().item()
        print('FOOLING RATE = ', fooling_rate, '(with noise std %.1f/255)' % (noise_std * 255) if noise_std > 0 else '')
        if target is not None:
            print('>> Images classified as the target:', (est_labels_pert == target).float().mean().item())

        if checkpoint is not None:
            torch.save({'v': v.detach().cpu(), 'epoch': epoch, 'fooling_rate': fooling_rate}, checkpoint + '.tmp')
            os.replace(checkpoint + '.tmp', checkpoint)

    return v.detach()
