import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

CMAP = plt.get_cmap('tab10')


class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return F.log_softmax(self.linear(x), dim=1)


def make_gaussians(n_classes=10, n_features=2, n_per_class=100, spread=3.0, seed=0):
    """Génère n_classes gaussiennes de covariance identité, avec des espérances proches.

    spread : écart entre les espérances (plus petit = plus proches)
    Retourne data (N, n_features), labels (N,) et means (n_classes, n_features).
    """
    torch.manual_seed(seed)

    # 1. Une espérance (un centre) par gaussienne
    means = spread * torch.randn(n_classes, n_features)

    # 2. n_per_class points autour de chaque centre, puis on empile
    data = torch.cat([means[k] + torch.randn(n_per_class, n_features) for k in range(n_classes)])

    # 3. Le label de chaque point = numéro de sa gaussienne
    labels = torch.arange(n_classes).repeat_interleave(n_per_class)

    # 4. Mélange des exemples (sinon ils sont triés par classe)
    perm = torch.randperm(len(data))
    return data[perm], labels[perm], means


def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for i, (x, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    model.eval()


def accuracy(model, data, labels):
    with torch.no_grad():
        return (model(data).argmax(dim=1) == labels).float().mean().item()


def build_trained_model(n_classes=10, n_features=2, num_epochs=50, lr=0.1):
    """Génère les données et entraîne le modèle. Retourne model, data, labels, means."""
    data, labels, means = make_gaussians(n_classes, n_features)
    dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data, labels),
                                             batch_size=32, shuffle=True)

    model = LogisticRegression(n_features, n_classes)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    train_model(model, dataloader, nn.NLLLoss(), optimizer, num_epochs)
    return model, data, labels, means


def plot_data(data, labels, means):
    """Visuel : les données, une couleur par gaussienne, et leurs espérances."""
    n_classes = len(means)
    plt.figure(figsize=(6, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=CMAP, vmin=0, vmax=9, s=10, alpha=0.7)
    plt.scatter(means[:, 0], means[:, 1], c=range(n_classes), cmap=CMAP, vmin=0, vmax=9,
                marker='X', s=200, edgecolors='black', label='espérances')
    plt.title("%d gaussiennes en dimension 2" % n_classes)
    plt.legend()
    plt.axis('equal')


def plot_decision_regions(model, data, n_classes=10, resolution=300):
    """Dessine en fond la classe prédite par le modèle en chaque point du plan."""
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = torch.meshgrid(torch.linspace(x_min, x_max, resolution),
                            torch.linspace(y_min, y_max, resolution), indexing='xy')
    grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    with torch.no_grad():
        regions = model(grid).argmax(dim=1).reshape(xx.shape)
    plt.contourf(xx, yy, regions, levels=torch.arange(-0.5, n_classes), cmap=CMAP, alpha=0.3)


if __name__ == "__main__":
    results = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results, exist_ok=True)

    model, data, labels, means = build_trained_model()
    acc = accuracy(model, data, labels)
    print("Précision : %.1f %%" % (100 * acc))

    plot_data(data, labels, means)
    plt.savefig(os.path.join(results, 'gaussiennes.png'), dpi=100)

    plt.figure(figsize=(6, 6))
    plot_decision_regions(model, data)
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=CMAP, vmin=0, vmax=9, s=10, edgecolors='none')
    plt.title("Régions de décision (précision %.1f %%)" % (100 * acc))
    plt.axis('equal')
    plt.savefig(os.path.join(results, 'regions_decision.png'), dpi=100)
    print("Résultats enregistrés dans", results)
    plt.show()
