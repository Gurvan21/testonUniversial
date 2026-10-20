import os
import sys

import torch
import matplotlib.pyplot as plt

# deepfool.py est dans le dossier parent (universal/python)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from deepfool import deepfool
from logistic_model import CMAP, accuracy, build_trained_model, plot_decision_regions


def run_deepfool(model, data, labels, n_classes=10):
    """Lance DeepFool sur chaque point bien classé.

    (log_softmax ne gêne pas : DeepFool n'utilise que des différences entre classes)
    Retourne les indices attaqués, les perturbations (N, 2), les itérations et le nombre de points trompés.
    """
    with torch.no_grad():
        correct = (model(data).argmax(dim=1) == labels).nonzero().flatten()

    perturbations, iterations, fooled = [], [], 0
    for i in correct:
        r, loop_i, label_orig, label_pert, pert_point = deepfool(data[i], model, num_classes=n_classes)
        perturbations.append(torch.from_numpy(r[0]))
        iterations.append(loop_i)
        fooled += label_pert != label_orig
    return correct, torch.stack(perturbations), iterations, fooled


def exact_distances(model, x):
    """Distance exacte de chaque point x à la frontière de décision la plus proche.

    Pour un modèle affine, la région de la classe prédite l est un polyèdre convexe :
    d(x) = min_{k != l} |f_k(x) - f_l(x)| / ||w_k - w_l||.
    """
    W, b = model.linear.weight.detach(), model.linear.bias.detach()
    f = x @ W.T + b
    l = f.argmax(dim=1)
    d = (f - f.gather(1, l[:, None])).abs() / (W[None] - W[l][:, None, :]).norm(dim=2).clamp_min(1e-12)
    d[torch.arange(len(l)), l] = float('inf')
    return d.min(dim=1).values


def plot_perturbations(model, data, labels, correct, perturbations, n_shown=25):
    """Visuel : quelques points et leur perturbation minimale (flèche jusqu'à la frontière la plus proche)."""
    shown = torch.randperm(len(correct))[:n_shown]
    plt.figure(figsize=(6, 6))
    plot_decision_regions(model, data)
    for j in shown:
        x0 = data[correct[j]]
        r = perturbations[j]
        plt.scatter(x0[0], x0[1], color=CMAP(labels[correct[j]].item()), s=25, edgecolors='black', zorder=3)
        plt.arrow(x0[0], x0[1], r[0], r[1], width=0.03, head_width=0.2, length_includes_head=True,
                  color='black', zorder=2)
    plt.title("DeepFool : perturbations minimales")
    plt.axis('equal')


def plot_norms(norms):
    """Visuel : distribution des tailles de perturbation."""
    plt.figure(figsize=(6, 4))
    plt.hist(norms, bins=40)
    plt.xlabel("||r||  (distance à la frontière la plus proche)")
    plt.ylabel("nombre de points")
    plt.title("DeepFool : taille des perturbations")


if __name__ == "__main__":
    results = os.path.join(HERE, 'results')
    os.makedirs(results, exist_ok=True)

    model, data, labels, means = build_trained_model()
    correct, perturbations, iterations, fooled = run_deepfool(model, data, labels)
    norms = perturbations.norm(dim=1)

    # DeepFool renvoie (1 + overshoot) * (d + 1e-4) : on le compare à la distance exacte
    d = exact_distances(model, data[correct])
    err = (norms - 1.02 * (d + 1e-4)).abs().max()

    report = ["Précision : %.1f %%" % (100 * accuracy(model, data, labels)),
              "Points attaqués : %d, trompés : %d" % (len(correct), fooled),
              "Itérations moyennes : %.2f" % (sum(iterations) / len(iterations)),
              "||r|| moyen : %.3f (min %.3f, max %.3f)" % (norms.mean(), norms.min(), norms.max()),
              "Écart max avec la distance exacte à la frontière : %.2e" % err]
    print("\n".join(report))
    with open(os.path.join(results, 'testlogistic.txt'), 'w') as f:
        f.write("\n".join(report) + "\n")

    plot_perturbations(model, data, labels, correct, perturbations)
    plt.savefig(os.path.join(results, 'deepfool_perturbations.png'), dpi=100)
    plot_norms(norms)
    plt.savefig(os.path.join(results, 'deepfool_norms.png'), dpi=100)
    print("Résultats enregistrés dans", results)
    plt.show()
