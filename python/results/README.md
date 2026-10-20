# Résultats : perturbations universelles

Reproduction de *Universal adversarial perturbations* (Moosavi-Dezfooli et al., CVPR 2017) avec le code PyTorch de ce dépôt.
Réglages communs : norme ℓ∞, ξ = 10/255, δ = 0,2 (objectif : 80 % de tromperie sur X), DeepFool avec les 10 classes les plus probables et un overshoot de 0,02.
Les modèles sont les poids torchvision pré-entraînés sur ImageNet (1000 classes), non modifiés.

## Résumé

| Expérience | Modèle | X (construction de v) | Évaluation | Précision propre | Précision perturbée | **Fooling rate** | Témoin (v mélangé) |
|---|---|---|---|---|---|---|---|
| [imagenette_resnet18_500](imagenette_resnet18_500/) | resnet18 | 500 images (Imagenette, 10 classes × 50) | 3 925 images (Imagenette val) | 77,7 % | 67,4 % | **24,0 %** | 8,1 % |
| [imagenet_resnet50_10k](imagenet_resnet50_10k/) | resnet50 | 10 000 images (ImageNet, 1000 classes × 10) | 40 000 images (ImageNet val) | 80,5 % | 17,1 % | **81,0 %** | 9,5 % |

- **Fooling rate** : proportion des images dont la prédiction change quand on ajoute v (métrique de l'article). Il est mesuré sur des images qui n'ont pas servi à construire v.
- **Témoin** : les pixels de v mélangés au hasard. La perturbation garde exactement les mêmes normes, mais perd sa structure. Il montre la part de tromperie due à la seule taille de la perturbation.
- Les images perturbées sont écrêtées dans [0, 1].

Référence de l'article (ℓ∞, ξ = 10, X = 10 000 images, ImageNet val) : ~78 % pour VGG-16 et GoogLeNet, ~84 % pour ResNet-152.

## Expérience 1 : Imagenette, 500 images (resnet18)

- **X** : 50 images par classe de `imagenette2-320/train`, soit 10 classes et 500 images.
- **Calcul** : sur le PC portable (CPU).
- **Perturbation** : ℓ∞ = 10/255, ℓ2 = 5,47 (maximum possible ≈ 15,2). Seulement 1,2 % des pixels atteignent ±ξ.
- **Sur X** : 85,6 % de tromperie (sans écrêtage), 51,2 % avec écrêtage dans [0, 1].
- **Sur Imagenette val** (3 925 images) : 24,0 % de tromperie, contre 8,1 % pour le témoin. Voir [eval_resnet18.txt](imagenette_resnet18_500/eval_resnet18.txt) et [examples_resnet18.png](imagenette_resnet18_500/examples_resnet18.png).
- **Pas de classe dominante** : les images trompées se répartissent entre beaucoup de classes (church 4,9 %, shower curtain 3,8 %…).

**Analyse.** v trompe 3 fois plus que le témoin, donc sa direction compte. Mais elle se généralise mal : 85,6 % sur X contre 24,0 % sur des images nouvelles. Avec seulement 500 images de 10 classes, l'algorithme atteint l'objectif de 80 % sur X après peu de passes. Il s'arrête alors avant d'avoir exploité le budget ξ (ℓ2 = 5,47 sur 15,2 possibles) : v est « apprise par cœur » sur X. Elle est aussi fragile : le simple écrêtage dans [0, 1] fait perdre 34 points sur X, car DeepFool ne franchit les frontières que de 2 %. L'article indique de son côté environ 30 % de tromperie avec 500 images (figure 6), un ordre de grandeur cohérent.

## Expérience 2 : ImageNet, 10 000 images (resnet50), protocole de l'article

- **X** : 10 images par classe du jeu de validation ILSVRC2012 (téléchargé depuis Hugging Face, `ILSVRC/imagenet-1k`), soit 1000 classes et 10 000 images.
- **Évaluation** : les 40 images restantes de chaque classe, soit 40 000 images disjointes de X.
  Différence avec l'article, qui prend X dans le jeu d'entraînement d'ImageNet : ici, X et l'évaluation sont deux parties disjointes du jeu de validation.
- **Calcul** : sur la machine `espagne` de l'école (RTX A4000), en 4 passes.
- **Perturbation** : ℓ∞ = 10/255, ℓ2 = 10,46.
- **Sur X** : 82,3 % de tromperie.
- **Sur les 40 000 images d'évaluation** : **81,0 %** de tromperie, contre 9,5 % pour le témoin. La précision chute de 80,5 % à 17,1 %. Voir [eval_resnet50.txt](imagenet_resnet50_10k/eval_resnet50.txt) et [examples_resnet50.png](imagenet_resnet50_10k/examples_resnet50.png).
- **Classe dominante** : **97,5 %** des images trompées sont classées *spider web*.

**Analyse.** Le résultat de l'article est reproduit : 81,0 %, dans la fourchette 78–84 % rapportée pour des architectures comparables. La perturbation se généralise presque parfaitement, avec 82,3 % sur X contre 81,0 % sur des images jamais vues. Par rapport à l'expérience 1, un X plus grand et plus varié (1000 classes) oblige l'algorithme à faire plus de passes et à mieux utiliser le budget ξ : ℓ2 = 10,46 contre 5,47. Le phénomène de **classe dominante** décrit dans l'article (section 4) apparaît très nettement : la perturbation envoie presque toutes les images vers *spider web*. Cela suggère que la région de décision de cette classe est très étendue et facile à atteindre depuis la plupart des images.

## Reproduire

```bash
# Expérience 1 (PC, Imagenette)
python demo.py -m resnet18 -t data/imagenette2-320/train --num_imgs_per_class 50
python evaluate.py -m resnet18 -v data/imagenette2-320/val --out results/imagenette_resnet18_500

# Expérience 2 (espagne, ImageNet val converti par prepare_imagenet_val.py)
D=/Data/gurvan.nkouathio-kemnhou
python demo.py -m resnet50 -t $D/imagenet_val/X --cache $D/cache/imagenet_val_X.pt
python evaluate.py -m resnet50 -v $D/imagenet_val/eval --out results/imagenet_resnet50_10k
```
