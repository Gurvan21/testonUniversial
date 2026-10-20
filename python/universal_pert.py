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


@torch.no_grad()
def predict_labels(dataset, model, v=None, batch_size=100):
    # Compute the estimated labels in batches
    device = next(model.parameters()).device
    labels = []
    for m in range(0, dataset.shape[0], batch_size):
        batch = dataset[m:m + batch_size].to(device)
        if v is not None:
            batch = batch + v
        labels.append(model(batch).argmax(dim=1).cpu())
    return torch.cat(labels)


def universal_perturbation(dataset, model, delta=0.2, max_iter_uni=np.inf, xi=10/255, p=np.inf, num_classes=10, overshoot=0.02, max_iter_df=10, batch_size=100, checkpoint=None, checkpoint_every=200):
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
                not_fooled = model(cur_img).argmax() == model(cur_img + v).argmax()

            if not_fooled:
                print('>> k = ', k_idx, ', pass #', itr)

                # Compute adversarial perturbation
                dr, iter, _, _, _ = deepfool((cur_img + v)[0], model, num_classes=num_classes, overshoot=overshoot, max_iter=max_iter_df)
                dr = torch.from_numpy(dr).to(device)
                num_deepfool += 1

                # Make sure it converged...
                if iter >= max_iter_df - 1:
                    num_not_converged += 1
                else:
                    v = v + dr

                    # Project on l_p ball
                    v = proj_lp(v, xi, p)

            if (k_idx + 1) % checkpoint_every == 0:
                start = k_idx + 1
                save_checkpoint()

        itr = itr + 1
        start, order = 0, None

        # Compute the fooling rate on the perturbed dataset
        est_labels_pert = predict_labels(dataset, model, v=v, batch_size=batch_size)
        fooling_rate = (est_labels_pert != est_labels_orig).float().mean().item()
        print('FOOLING RATE = ', fooling_rate)
        # (counted since the start of the pass, or since the resumption)
        print('>> DeepFool did not converge (update discarded) for %d of the %d images attacked' % (num_not_converged, num_deepfool))
        save_checkpoint()

    return v
