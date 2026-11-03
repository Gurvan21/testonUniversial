# UAP robustes aux transformations (EOT), attaques universelles physiques et évaluations adaptatives des défenses contre les UAP

Notes de recherche (novembre 2026). Convention : **[vérifié]** = lu dans le texte primaire pendant cette session (PDF arXiv v2 de Xu & Singh extrait localement, texte ICML 2024, pages arXiv/CVF) ; **[non vérifié]** = connu par ailleurs (mémoire/résumé secondaire), à recontrôler avant de citer un chiffre. Environ 17 appels d'outils ont été faits, donc la couverture des articles qui citent Xu & Singh 2024 ou Moosavi-Dezfooli 2017 est **partielle**.

---

## Q1. Quels travaux calculent des UAP avec espérance sur des transformations ou du bruit aléatoires, et quelle est leur robustesse (chiffres) ?

### Takeaway
La référence directe est Xu & Singh (ICML 2024, arXiv 2206.10858). Ils formalisent la « robust UAP » comme une espérance imbriquée : une espérance sur les transformations **appliquées à la perturbation** u, puis une espérance sur les données. Leur méthode RobustUAP atteint 75–97 % de robust ASR (γ=0,6), contre 0–36 % pour la UAP standard sous rotations et compositions affines. Sous bruit uniforme U(0,3), elle atteint 96–100 % (γ=0,8). Avant eux, l'EOT existait pour les exemples par image (Athalye 2018) et pour les patchs universels (Brown 2017, Li/Schmidt/Kolter 2019, UPC 2020). Ces travaux ne relèvent pas du cadre « UAP additive de faible norme ».

### Cited Findings

**(a) Xu & Singh, « Robust Universal Adversarial Perturbations », ICML 2024 (arXiv 2206.10858, v1 juin 2022, v2 juin 2023 ; soumis d'abord à ICLR 2023, OpenReview VpYBxaPLaj-)**
- Référence et liens : Changming Xu, Gagandeep Singh (UIUC). [arXiv](https://arxiv.org/abs/2206.10858), [ICML 2024 poster](https://icml.cc/virtual/2024/poster/34115), [OpenReview](https://openreview.net/forum?id=VpYBxaPLaj-), [PDF auteur](https://ggndpsngh.github.io/files/ruap.pdf), [mlanthology](https://mlanthology.org/icml/2024/xu2024icml-robust/) [vérifié].
- Modèle de menace : les transformations s'appliquent **à la perturbation u avant son ajout à x** (stickers sur caméra, transmission over-the-air), pas à l'image entière. Ils définissent ASR_R(f, μ, T, γ, u) comme la probabilité qu'un voisin transformé τ(u) reste une UAP avec un taux de succès universel ≥ γ. Formulation (Eq. 2) : arg min_u E_{u'∈N_T(u)}[ I(‖u'‖<ε) · E_{x∼μ}[δ(f(x+u'), f(x))] ]. Les auteurs notent que c'est plus dur que l'EOT, qui ne minimise que sur les transformations — [arXiv PDF](https://arxiv.org/pdf/2206.10858) [vérifié].
- Algorithme RobustUAP : par batch et par lot de transformations échantillonnées, il fait des pas de PGD (au lieu d'un pas SGD unique). Il s'arrête via EstRobustness, une borne de Hoeffding avec n = ⌈ln(2/φ)/(2ψ²)⌉ = 738 échantillons pour ψ=φ=0,05 (±5 % avec 95 % de confiance quand le voisinage a une densité bien définie) — [arXiv PDF](https://arxiv.org/pdf/2206.10858) [vérifié].
- Transformations : rotation R(θ), translation T(x,y), scaling Sc(p), shear Sh(m), contraste/luminosité B(α,β), bruit uniforme U(ε) dans une boule L∞, fog, primitives PRIME (spectral/spatial/couleur). Généralisation testée sur les corruptions de type ImageNet-C (JPEG, frost, motion blur…) — [arXiv PDF](https://arxiv.org/pdf/2206.10858) [vérifié].
- Configuration : Inception-v3 (ILSVRC 2012), VGG16 (CIFAR-10), ℓ2 avec ε=100 (ImageNet) et ε=10 (CIFAR), valeurs reprises de Moosavi-Dezfooli 2017. Les UAP sont entraînées sur 2000 images et testées sur 1000. Plafond de 5 epochs, 40 itérations PGD internes — [arXiv PDF](https://arxiv.org/pdf/2206.10858) [vérifié].
- **Table 1 (robust ASR, γ=0,6)**, colonnes StandardUAP / SGD / StandardUAP_RP / RobustUAP. Chiffres identiques dans la version ICML 2024 — [arXiv PDF](https://arxiv.org/pdf/2206.10858), [texte ICML 2024](https://object.cloud.sdsc.edu/v1/AUTH_da4962d3368042ac8337e2dfdd3e7bf3/ml-papers-txt/ICML/2024/robust_universal_adversarial_perturbations__a3ae8ee0.txt) [vérifié] :
  - ImageNet R(20) : 0,0 / 69,9 / 2,9 / 93,2 %
  - ImageNet T(2,2) : 35,9 / 96,1 / 38,8 / 97,1 %
  - ImageNet Sc(5),R(5),B(5,0.01) : 22,3 / 85,4 / 43,7 / 96,1 %
  - ImageNet R(10),T(2,2),Sh(2),Sc(2),B(2,0.001) : 0,0 / 63,1 / 2,9 / 86,4 %
  - CIFAR R(30),B(2,0.001) : 0,0 / 64,1 / 2,9 / 75,7 %
  - CIFAR R(2),Sh(2) : 42,7 / 88,3 / 52,4 / 96,1 %
  - CIFAR composition complète : 0,0 / 58,3 / 7,8 / 79,6 %
- **Bruit uniforme (Table 4 de arXiv v2 = Table 5 ICML, γ=0,8)** — [arXiv PDF](https://arxiv.org/pdf/2206.10858) [vérifié] :
  - ImageNet U(0.1) : 81,6 / 94,2 / 91,3 / 99,0 %
  - ImageNet U(0.3) : 10,7 / 68,9 / 42,7 / 96,1 %
  - CIFAR U(0.1) : 66,0 / 98,1 / 96,1 / 100 %
  - CIFAR U(0.3) : 5,8 / 96,1 / 47,6 / 100 %
  - Exemple cité : RobustUAP a un ASR_U > 0,9 dans plus de 97 % des tirages sous U(0.1) sur CIFAR, contre au plus 30 % pour les autres méthodes.
  - Attention : ce bruit est **ajouté à u, pas à x+u** côté défenseur, et l'échelle dépend de la normalisation des modèles.
- ASR universel moyen après transformation (Table 5 arXiv), composition complète sur ImageNet : 13,6 / 64,8 / 29,0 / 75,3 %. Les UAP standards ne sont pas nulles en moyenne, mais elles sont rarement très efficaces — [arXiv PDF](https://arxiv.org/pdf/2206.10858) [vérifié].
- **Coût de la robustesse** (Table 7, ASR universel *sans* transformation, UAP robustes à la composition complète) — [arXiv PDF](https://arxiv.org/pdf/2206.10858) [vérifié] :
  - ImageNet : StandardUAP 95,5 %, SGD 85,6 %, StandardUAP_RP 82,3 %, RobustUAP 91,3 %
  - CIFAR : 96,2 / 89,3 / 84,0 / 93,7 %
- Autres résultats [vérifié] :
  - MobileNet, composition complète : 4,1 / 75,7 / 8,6 / 86,2 %.
  - ViT : 2,0 / 72,1 / 12,9 / 88,5 %.
  - Transfert de la robustesse : au moins 65 % entre Inception et MobileNet, environ 32–43 % vers ou depuis ViT.
  - PRIME-trained : plus de 58,3 % de robust ASR sur tous les jeux testés, dont 73,3 % sur JPEG ; une UAP entraînée sur Affine ne fait que 37,3 % sur JPEG.
  - Temps de calcul : StandardUAP 37 min, SGD 32, StandardUAP_RP 43, RobustUAP 118.
  - Source : [arXiv PDF](https://arxiv.org/pdf/2206.10858).
- Code : **aucun lien GitHub** dans arXiv v2 ni dans le texte ICML 2024 ; aucun dépôt trouvé par recherche — [texte ICML 2024](https://object.cloud.sdsc.edu/v1/AUTH_da4962d3368042ac8337e2dfdd3e7bf3/ml-papers-txt/ICML/2024/robust_universal_adversarial_perturbations__a3ae8ee0.txt) [vérifié]. Pas d'expériences physiques réelles : « We did not have the hardware to print high resolution transparent stickers » — [arXiv PDF](https://arxiv.org/pdf/2206.10858) [vérifié].

**(b) Athalye, Engstrom, Ilyas, Kwok, « Synthesizing Robust Adversarial Examples », ICML 2018 (arXiv 1707.07397)**
- Introduit l'EOT : maximiser E_{t∼T}[log P(y_t | t(x'))] sous contrainte de distance perceptuelle attendue. Appliqué en 2D, en 3D (texture de tortue imprimée 3D) et dans le monde physique — [PMLR PDF](http://proceedings.mlr.press/v80/athalye18b/athalye18b.pdf) [vérifié pour la méthode].
- Chiffres [non vérifiés dans cette session, de mémoire] : environ 96 % d'adversarialité moyenne en 2D, environ 84 % en 3D simulé, et la tortue physique classée « rifle » sur environ 82 % des vues. C'est une perturbation **par objet**, pas universelle sur des images. Code : implémentations publiques non officielles (tutoriel anishathalye.com) [non vérifié].

**(c) Brown, Mané, Roy, Abadi, Gilmer, « Adversarial Patch », arXiv 1712.09665 (NIPS 2017 workshop MLSec)**
- Patch universel, robuste et ciblé (« toaster »), entraîné par une variante de l'EOT (position, rotation, échelle) ; il n'est pas contraint en norme — [PDF MLSec](https://machine-learning-and-security.github.io/papers/mlsec17_paper_27.pdf) [vérifié pour la méthode].
- Chiffres [non vérifiés] : succès ciblé élevé (de l'ordre de 90 % et plus en white-box) pour un patch couvrant environ 10 % de l'image. Code : tutoriel CleverHans / Colab [non vérifié].

**(d) Li, Schmidt, Kolter, « Adversarial camera stickers: A physical camera-based attack on deep learning systems », ICML 2019 (arXiv 1904.00759)**
- Sticker translucide posé sur l'objectif : la perturbation est donc universelle sur toutes les images vues par la caméra. Modèle d'alpha-blending différentiable, optimisé itérativement avec le modèle de menace. Résultat : 49,6 % de succès **ciblé** sur ImageNet en conditions physiques — [arXiv](https://arxiv.org/abs/1904.00759), [ICML](https://icml.cc/virtual/2019/poster/4262) [vérifié pour le 49,6 %]. Xu & Singh citent ce cadre (sticker caméra) comme motivation principale — [arXiv PDF](https://arxiv.org/pdf/2206.10858) [vérifié].

**(e) Huang et al., « Universal Physical Camouflage Attacks on Object Detectors » (UPC), CVPR 2020 (arXiv 1909.04326)**
- Camouflage universel par catégorie. Il attaque conjointement le RPN, le classifieur et le régresseur de Faster R-CNN, et ajoute des transformations de déformation non rigide (vêtements) et une contrainte de naturalité. Il introduit le benchmark virtuel AttackScenes et des tests réels (masques, T-shirts). Code et démos : https://mesunhlf.github.io/index_physical.html — [CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Huang_Universal_Physical_Camouflage_Attacks_on_Object_Detectors_CVPR_2020_paper.html) [vérifié]. Domaine : détection, pas classification.

**(f) Algorithmes SGD/PGD et génératifs (contexte)**
- Shafahi, Najibi, Xu, Dickerson, Davis, Goldstein, « Universal Adversarial Training », AAAI 2020 (arXiv 1811.11304). UAP par SGD/PGD stochastique et entraînement min-max à environ 2× le coût naturel. Xu & Singh disent explicitement s'appuyer sur Shafahi pour leur baseline SGD — [mlanthology](https://mlanthology.org/aaai/2020/shafahi2020aaai-universal), [arXiv PDF Xu&Singh](https://arxiv.org/pdf/2206.10858) [vérifié]. Chiffres CIFAR rapportés par un résumé Medium (SGD 42,56 % vs ADAM 13,30 % vs PGD 13,79 %, WRN-32, ε=8) : **non vérifiés**, et la métrique n'est pas claire — [Medium](https://medium.com/ml-summaries/universal-adversarial-training-paper-summary-e9150fd7aa1c).
- Mummadi, Brox, Metzen, « Defending Against Universal Perturbations With Shared Adversarial Training » (arXiv 1812.03705, ICCV 2019). Environ 3× de robustesse aux attaques universelles pour au plus environ 5 % de perte de précision propre — [arXiv](https://arxiv.org/pdf/1812.03705) (via résumé de recherche, [non vérifié dans le texte]).
- Li et al., « Regional Homogeneity », ECCV 2020 (arXiv 1904.00979). Perturbations régionalement homogènes, quasi universelles, contre 9 défenses (dont des défenses par transformation d'entrée selon le résumé de recherche), en **transfert** et non en adaptatif EOT. Amélioration moyenne de 14,0 %. Code : https://github.com/LiYingwei/Regional-Homogeneity — [arXiv](https://arxiv.org/abs/1904.00979) [vérifié].
- Weng et al., « Comparative Evaluation of Recent Universal Adversarial Perturbations in Image Classification » (arXiv 2306.11261). Benchmark noise-based vs generator-based avec examen de défenses ; l'abstract ne mentionne pas d'attaques adaptatives — [arXiv](https://arxiv.org/abs/2306.11261) [vérifié pour l'abstract].
- Mopuri et al., « Generalizable Data-free Objective for Crafting UAPs » (GD-UAP, TPAMI, arXiv 1801.08092) — apparu dans les résultats [abstract non lu].
- Zhang et al., « Data-free UAP and Black-box Attack », ICCV 2021 — [CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_Data-Free_Universal_Adversarial_Perturbation_and_Black-Box_Attack_ICCV_2021_paper.pdf).

### Inferences
- Le cadre de Xu & Singh (transformation de **u seul**) diffère du cadre d'une défense par bruit/transformation d'entrée, qui transforme x+u. Pour un bruit additif indépendant, les deux coïncident presque, car x + (u+n) = (x+u) + n. Les résultats « U(ε) » de Xu & Singh sont donc l'analogue le plus proche d'une UAP adaptative contre une défense par bruit additif uniforme. Ce n'est pas le cas des rotations ou du scaling, qui dans une vraie défense s'appliqueraient aussi à x.
- Les UAP standards (DeepFool) sont fragiles au bruit fort : 10,7 % de robust ASR sous U(0.3) sur ImageNet. Cela laisse penser qu'une défense par bruit aléatoire « marche » contre un attaquant non adaptatif, mais quasiment pas contre un attaquant EOT (96,1 %).

### Gaps
- Chiffres exacts d'Athalye 2018 et de Brown 2017 non revérifiés dans le texte pendant cette session.
- GAP (Poursaeed et al., CVPR 2018, arXiv 1712.02328), NAG (Mopuri et al., CVPR 2018), Fast Feature Fool (Mopuri 2017) : non consultés. À ma connaissance, aucun ne s'entraîne explicitement avec l'EOT. GAP et NAG revendiquent surtout de la transférabilité et de la diversité, pas une robustesse aux transformations [non vérifié].
- Autres domaines non consultés (à mentionner brièvement, [non vérifié]) : audio, avec les UAP robustes over-the-air (Neekhara et al. 2019, « Universal adversarial perturbations for speech recognition systems »; Abdoli et al. 2019) ; sans-fil, avec Sadeghi & Larsson 2019 et Bahramali et al. 2021 (PGM, cité par Xu & Singh comme contournant l'entraînement adversarial et les défenses de soustraction de bruit grâce à un *ensemble* d'UAP — [arXiv PDF](https://arxiv.org/pdf/2206.10858) [vérifié pour la citation]).
- La liste des articles citant Xu & Singh 2024 n'a pas pu être explorée (pas d'accès à Semantic Scholar ou Google Scholar dans cette session).

---

## Q2. Comment l'algorithme de Moosavi-Dezfooli (DeepFool) combiné à l'EOT se compare-t-il aux UAP par gradient (SGD/PGD) + EOT ? (StandardUAP_RP vs SGD vs RobustUAP)

### Takeaway
Chez Xu & Singh, la combinaison naïve « DeepFool-UAP + perturbation robuste par point » (StandardUAP_RP) n'apporte presque rien face aux transformations géométriques. Elle obtient 2,9 % sur R(20) et sur la composition complète ImageNet, contre 0 % pour StandardUAP. Elle aide davantage pour le bruit (42,7 % vs 10,7 % sous U(0.3)). Le SGD avec EOT, à la Shafahi, est nettement meilleur (58–96 %). RobustUAP (PGD par batch + arrêt par borne probabiliste) est le meilleur (76–97 %). L'écart avec SGD est d'environ 1 point sur les transformations faciles et de 11 à 27 points sur les difficiles.

### Cited Findings
- Définitions des baselines — [arXiv PDF, App. E–G](https://arxiv.org/pdf/2206.10858) [vérifié] :
  - **StandardUAP** = Algorithme 3 (Moosavi-Dezfooli 2017, mise à jour par perturbation minimale de type DeepFool puis projection).
  - **StandardUAP_RP** = même boucle, mais avec à chaque point la plus petite modification Δu_r rendant u_r+Δu_r adversarial *en espérance sur le voisinage* (« robust adversarial perturbation », c'est-à-dire de l'EOT par point).
  - **SGD** = Algorithme 2 : SGD à momentum sur la perte moyennée sur batch × transformations échantillonnées, avec projection P_{p,ε}, d'après Shafahi et al.
  - Les auteurs qualifient les deux baselines de « naive combinations of the EoT and UAP algorithms ».
- Comparaison chiffrée : voir les tables de Q1. Sur la composition complète ImageNet : StandardUAP 0,0 %, StandardUAP_RP 2,9 %, SGD 63,1 %, RobustUAP 86,4 %. Sous U(0.3) ImageNet : 10,7 / 42,7 / 68,9 / 96,1 % — [arXiv PDF](https://arxiv.org/pdf/2206.10858) [vérifié].
- Sans transformation, StandardUAP_RP est le moins efficace des quatre (82,3 % d'ASR universel sur ImageNet vs 95,5 % pour StandardUAP) — [arXiv PDF, Table 7](https://arxiv.org/pdf/2206.10858) [vérifié].
- SGD plafonne : environ 65 % de robust ASR même à 1250 itérations, le temps de RobustUAP. RobustUAP limité au temps de SGD atteint 72,0 % (vs 63,1 % pour SGD) sur la composition complète — [arXiv PDF, App. R, Table 10](https://arxiv.org/pdf/2206.10858) [vérifié].
- Ablation sur l'optimiseur : SGD fait mieux qu'Adam, Adamax, Adagrad et RMSProp pour ce problème — [arXiv PDF, App. X](https://arxiv.org/pdf/2206.10858) [vérifié, chiffres de la Table 14 non relevés].
- Ciblé (App. S, γ=0,6) — [arXiv PDF](https://arxiv.org/pdf/2206.10858) [vérifié] :
  - toy poodle→maltese : 42,4 / 99,1 / 85,6 / 99,8 %
  - perfume→padlock : 0,0 / 63,8 / 5,1 / 76,4 %

### Inferences
- L'échec de StandardUAP_RP s'explique probablement par la mise à jour DeepFool, qui est *minimale* : elle pousse u juste au-delà de la frontière pour une image. Même calculée en espérance, elle laisse peu de marge, et la projection sur la boule ε écrase les contributions successives. Les méthodes par gradient maximisent au contraire une perte (marge) moyenne. C'est une interprétation, pas une analyse des auteurs.
- Pour le projet local (DeepFool-UAP sur ImageNet + bruit), ce résultat prédit que l'ajout d'EOT à la boucle de Moosavi-Dezfooli sera bien moins efficace qu'une UAP SGD/PGD-EOT contre une défense par bruit.

### Gaps
- Pas de comparaison publique trouvée entre DeepFool-UAP+EOT et SGD-UAP+EOT pour du **bruit gaussien ajouté à l'entrée** à plusieurs σ, ni sur des architectures modernes (ConvNeXt, Swin, MaxViT). Xu & Singh utilisent Inception-v3, MobileNet, ViT, VGG16 et ResNet-18.
- Le code de Xu & Singh n'étant pas public, la reproduction exacte de StandardUAP_RP (nombre d'échantillons EOT par point, solveur interne) n'est pas possible à partir de ce qui a été trouvé.

---

## Q3. Quelqu'un a-t-il étudié le compromis coût/protection du défenseur (précision propre vs fooling rate d'une UAP EOT adaptative, en fonction de σ) ?

### Takeaway
**Je n'ai trouvé aucun travail** qui trace, pour la classification d'images, la précision propre d'un classifieur randomisé (bruit gaussien ou uniforme d'écart-type σ) *et* le fooling rate d'une UAP adaptative (EOT) en fonction de σ. Les pièces existent séparément : des UAP robustes au bruit côté attaquant (Xu & Singh), des compromis précision/robustesse des défenses randomisées contre les attaques *par image* (Pinot et al. 2019, randomized smoothing) et des défenses anti-UAP à coût propre mesuré (Shared AT, Universal AT), mais rien qui croise les trois.

### Cited Findings
- Le résultat le plus proche est Xu & Singh 2024, section 5.6. Un entraînement adversarial « batch-wise RobustUAP » sur CIFAR-10/VGG16 (composition complète) donne 0,4 % de robust ASR pour 90,1 % de précision propre, contre 0,2 % et 83,5 % pour l'entraînement adversarial standard, en 12 min au lieu de 48. C'est un **seul point**, une défense par entraînement et non par randomisation, et l'étude est qualifiée de « preliminary » — [arXiv PDF](https://arxiv.org/pdf/2206.10858), [texte ICML 2024](https://object.cloud.sdsc.edu/v1/AUTH_da4962d3368042ac8337e2dfdd3e7bf3/ml-papers-txt/ICML/2024/robust_universal_adversarial_perturbations__a3ae8ee0.txt) [vérifié].
- Pinot et al., « Theoretical evidence for adversarial robustness through randomization » (arXiv 1902.01148, NeurIPS 2019). La précision chute quand le bruit injecté augmente, et le bruit doit être calibré. Les attaques sont évaluées avec de l'EOT Monte Carlo, mais **par image**, pas universelles — [arXiv](https://arxiv.org/pdf/1902.01148) (via résumé de recherche).
- Cohen, Rosenfeld, Kolter, « Certified Adversarial Robustness via Randomized Smoothing » (ICML 2019). Compromis σ / précision certifiée, avec 49 % de top-1 certifié pour ‖δ‖₂<0,5 sur ImageNet. Ce sont des garanties ℓ2 par point qui couvrent en particulier toute UAP de norme inférieure au rayon, mais aucune évaluation spécifique UAP — [arXiv](https://arxiv.org/pdf/1902.02918).
- Speaker recognition (arXiv 2101.08909). Le lissage gaussien n'abîme la performance propre et sous attaque universelle qu'à σ=0,2 élevé, et la précision sous attaque universelle s'améliore. Hors domaine image, et le caractère adaptatif n'est pas établi — [arXiv](https://arxiv.org/pdf/2101.08909) (via résumé de recherche, [non vérifié]).
- Le bruit aléatoire de faible amplitude nuit peu à la précision. Pour obtenir la même dégradation qu'une UAP, le bruit aléatoire doit être de l'ordre de la norme de l'image (attaque matérielle UAP) — [Information 14(9):516 / arXiv 2111.09488](https://doi.org/10.3390/info14090516).
- Les défenses par transformation d'entrée (JPEG, BaRT…) tombent à une robustesse quasi nulle face à des attaques adaptatives PGD/BPDA+EOT, mais par image — [DRIFT arXiv 2509.24359](https://arxiv.org/pdf/2509.24359) ; Tramèr et al. NeurIPS 2020 rappellent qu'une défense randomisée n'est correctement évaluée qu'avec une attaque qui moyenne sur l'aléa (EOT) — [NeurIPS PDF](https://proceedings.nips.cc/paper/2020/file/11f38f8ecd71867b42433548d1078e38-Paper.pdf).
- Mummadi et al. (Shared AT) : au plus environ 5 % de perte propre pour environ 3× de robustesse universelle, défense par entraînement — [arXiv 1812.03705](https://arxiv.org/pdf/1812.03705) [non vérifié dans le texte]. Akhtar et al., « Defense against Universal Adversarial Perturbations » (PRN, CVPR 2018) — [arXiv 1711.05929](https://arxiv.org/pdf/1711.05929) [non lu en détail ; à ma connaissance, pas d'évaluation EOT adaptative].
- Termes de recherche utilisés sans résultat direct :
  - « universal adversarial perturbation randomized defense adaptive attack noise injection clean accuracy trade-off »
  - « universal adversarial perturbations against randomized smoothing certified defense »
  - « "universal adversarial perturbation" input transformation defense adaptive attacker EOT evaluation »
  - « universal adversarial perturbation Gaussian noise augmentation defense randomized smoothing universal attack robustness »
  - « "universal" adversarial perturbation "random noise" defense fooling rate versus noise standard deviation clean accuracy adaptive »
  - Moteur web général (mode étendu) uniquement : ni Semantic Scholar ni Google Scholar.

### Inferences
- La courbe « précision propre(σ) vs fooling rate EOT-UAP(σ) » paraît être une **contribution non couverte** par la littérature trouvée. Les chiffres de Xu & Singh suggèrent qu'à amplitude modérée de bruit, une UAP EOT garde un succès quasi intact (96–100 % sous U(0.3) sur u). Le défenseur devrait donc monter σ jusqu'à une perte propre significative avant d'obtenir une protection, ce qui est cohérent avec la logique du randomized smoothing. C'est à confirmer expérimentalement.
- Un défenseur qui randomise *x+u* (et non u seul) dégrade aussi le signal propre. Le compromis doit donc mesurer la précision propre sous le même bruit, ce que Xu & Singh ne font pas.

### Gaps
- Aucun article trouvé avec cette courbe. Pour confirmer l'absence, il faudrait interroger Semantic Scholar sur les citations de Xu & Singh 2024, de Moosavi-Dezfooli 2017 avec les mots « randomized »/« smoothing »/« EOT », et de Cohen 2019 avec « universal ».
- Les certificats « universels » spécifiques (certifier qu'aucune perturbation *partagée* ne trompe plus de k % d'un lot) existent probablement dans la littérature de certification collective. Ce point n'a pas été vérifié dans cette session.

---

## Q4. Quelqu'un a-t-il étudié comment le label dominant ou le motif visuel d'une UAP change quand on la rend robuste (EOT) ?

### Takeaway
Je n'ai trouvé aucune étude du label dominant d'une UAP robuste/EOT. Xu & Singh visualisent seulement les UAP robustes (Fig. 4, App. Y/AD) et affirment une imperceptibilité « similaire » aux UAP standards. Ils ne disent rien des classes dominantes ni de la structure fréquentielle. Le phénomène de label dominant est documenté pour les UAP standards (Moosavi-Dezfooli 2017 ; Zhang et al.).

### Cited Findings
- Xu & Singh : « Our robust UAPs have a similar level of imperceptibility to standard UAPs ». Leur Fig. 4 montre des labels Inception-v3 sur des images perturbées par des UAP robustes, des UAP robustes PRIME et des UAP standards transformées. Aucune analyse de label dominant — [arXiv PDF](https://arxiv.org/pdf/2206.10858), [texte ICML 2024](https://object.cloud.sdsc.edu/v1/AUTH_da4962d3368042ac8337e2dfdd3e7bf3/ml-papers-txt/ICML/2024/robust_universal_adversarial_perturbations__a3ae8ee0.txt) [vérifié].
- Moosavi-Dezfooli et al. 2017 : les UAP poussent les prédictions vers un petit nombre de classes dominantes, et le graphe de transitions de labels est dominé par quelques classes — [CVPR 2017 PDF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Moosavi-Dezfooli_Universal_Adversarial_Perturbations_CVPR_2017_paper.pdf) (via résumé de recherche).
- Zhang et al. (CVPR 2020 « UAP: a perspective of features », et ICCV 2021 data-free UAP) : la UAP se comporte comme une feature dominante. Le label dominant serait dû aux features propres de la perturbation plutôt qu'aux images — [ICCV 2021 PDF](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_Data-Free_Universal_Adversarial_Perturbation_and_Black-Box_Attack_ICCV_2021_paper.pdf) (via résumé de recherche, [non vérifié dans le texte]).
- « Learning UAP by Adversarial Example » (AAAI 2022) : les UAP reposeraient sur des features « robustes », plus dominantes que les perturbations par image — [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/20023/19782) (via résumé de recherche).
- Xu & Singh notent qu'en ciblé, il est plus facile de transformer un chien en un autre chien qu'un parfum en cadenas (99,8 % vs 76,4 %). C'est un indice indirect que la robustesse dépend de la proximité sémantique des classes — [arXiv PDF, App. S](https://arxiv.org/pdf/2206.10858) [vérifié].

### Inferences
- Hypothèse testable, non trouvée dans la littérature : comme l'EOT moyenne sur des rotations, translations et bruits, elle devrait favoriser des motifs basse fréquence et quasi invariants par translation (textures répétitives). Elle pourrait aussi concentrer encore plus le fooling sur un label dominant « texture », dans l'esprit des features dominantes de Zhang et al. C'est une spéculation à vérifier sur les UAP du projet.

### Gaps
- Aucune source sur la distribution des labels induits par une UAP EOT vs standard, ni sur son spectre de Fourier. La « Texture Re-scalable UAP » (arXiv 2406.06089) est apparue dans les résultats mais n'a pas été lue ; elle pourrait traiter du motif texturel des UAP — [arXiv](https://arxiv.org/pdf/2406.06089).
- Les figures de l'Appendice Y/AD de Xu & Singh (UAP des trois algorithmes robustes) n'ont pas pu être examinées visuellement (texte extrait seulement).
