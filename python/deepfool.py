import torch


def deepfool(image, net, num_classes=10, overshoot=0.02, max_iter=50):

    """
       :param image: Image tensor of size 3xHxW
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, original label, new estimated_label and perturbed image
    """
    # Work on the device of the network (the caller chooses it), without moving the network
    device = next(net.parameters()).device
    image = image.to(device)

    with torch.no_grad():
        f_image = net(image[None]).flatten()
    I = f_image.argsort(descending=True)[:num_classes]
    label = I[0].item()

    pert_image = image[None].clone()
    w = torch.zeros_like(pert_image)
    r_tot = torch.zeros_like(pert_image)

    loop_i = 0

    x = pert_image.clone().requires_grad_(True)
    fs = net(x)
    k_i = label

    while k_i == label and loop_i < max_iter:

        pert = float("inf")
        # Gradients are taken w.r.t. x only, so nothing accumulates in the network's parameters
        grad_orig = torch.autograd.grad(fs[0, I[0]], x, retain_graph=True)[0]

        for k in range(1, num_classes):
            cur_grad = torch.autograd.grad(fs[0, I[k]], x, retain_graph=True)[0]

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).detach()

            pert_k = (f_k.abs() / w_k.norm()).item()

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i = (pert + 1e-4) * w / w.norm()
        r_tot = r_tot + r_i

        pert_image = image + (1 + overshoot) * r_tot

        x = pert_image.detach().requires_grad_(True)
        fs = net(x)
        k_i = fs.argmax().item()

        loop_i += 1

    r_tot = (1 + overshoot) * r_tot

    return r_tot.cpu().numpy(), loop_i, label, k_i, pert_image.detach()
