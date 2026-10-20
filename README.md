# Perturbations adversariales universelles : reproduction en PyTorch

Ce dépôt reproduit l'article *Universal adversarial perturbations* (Moosavi-Dezfooli, Fawzi, Fawzi, Frossard, CVPR 2017) avec PyTorch, puis étend l'étude à des architectures récentes.

Une **perturbation universelle** est un unique vecteur v, imperceptible (‖v‖∞ ≤ 10/255), qui, ajouté à **n'importe quelle** image naturelle, change la prédiction d'un classifieur dans la grande majorité des cas. Elle est construite en agrégeant les perturbations minimales calculées par [DeepFool](http://arxiv.org/pdf/1511.04599) sur un ensemble d'images.

Il s'agit d'un fork du [code des auteurs](https://github.com/LTS4/universal). La version Python d'origine (TensorFlow 1, Inception) a été réécrite en PyTorch, et le code a été complété pour l'évaluation et les calculs longs.

## Résultat principal

resnet50 (torchvision), ξ = 10/255 en norme ℓ∞, perturbation construite sur 10 000 images ImageNet (10 par classe), évaluée sur 40 000 autres images :

| | Précision | Fooling rate |
|---|---|---|
| Images propres | 80,5 % | – |
| + bruit de même norme (témoin) | 78,8 % | 9,5 % |
| + **perturbation universelle** | **17,1 %** | **81,0 %** |

Ce résultat est cohérent avec l'article (78 à 84 % selon le réseau). 97,5 % des images trompées sont classées *spider web* : c'est le phénomène de **classe dominante** décrit dans l'article.
Détails, analyse et comparaison avec une perturbation construite sur 500 images : [python/results/README.md](python/results/README.md).

## Organisation

| Dossier | Contenu |
|---|---|
| [python/](python/) | Code PyTorch : DeepFool, algorithme de la perturbation universelle, démo, évaluation, préparation d'ImageNet ([mode d'emploi](python/README.md)) |
| [python/results/](python/results/) | Résultats des expériences : perturbations calculées, chiffres, figures |
| [python/testlogistic/](python/testlogistic/) | Tests de DeepFool : régression logistique en 2D, où la perturbation minimale se calcule exactement, et resnet34 sur une image ([détails](python/testlogistic/README.md)) |
| [matlab/](matlab/), [precomputed/](precomputed/) | Code MATLAB et perturbations précalculées des auteurs (inchangés) |

## Démarrage rapide

```bash
cd python
pip install -r requirements.txt

# Tests de DeepFool
python testlogistic/testlogistic.py

# Perturbation universelle sur un petit jeu d'images (ex. Imagenette), puis évaluation
python demo.py -m resnet18 -t <dossier d'images, un sous-dossier par classe> --num_imgs_per_class 50
python evaluate.py -m resnet18 -v <dossier de validation>
```

Pour reproduire le protocole de l'article sur ImageNet, voir [python/README.md](python/README.md).

## Référence

S. Moosavi-Dezfooli\*, A. Fawzi\*, O. Fawzi, P. Frossard, [*Universal adversarial perturbations*](http://arxiv.org/pdf/1610.08401), CVPR 2017.
S. Moosavi-Dezfooli, A. Fawzi, P. Frossard, [*DeepFool: a simple and accurate method to fool deep neural networks*](http://arxiv.org/pdf/1511.04599), CVPR 2016.
