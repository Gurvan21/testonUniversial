# Défenses dédiées contre les perturbations adversariales universelles (UAP) en classification d'images

Note de méthode : les chiffres marqués **[vérifié]** ont été lus dans le texte du PDF arXiv (extraction pdftotext) pendant cette recherche. Les chiffres marqués **[non vérifié]** viennent d'un résumé, d'un extrait de moteur de recherche ou d'un tiers, et doivent être vérifiés avant d'être cités. Métrique : « FR » = fooling rate (taux de changement de label), « acc. UAP » = top-1 sur images + UAP. ε exprimé sur l'échelle [0,255] sauf mention contraire.

## Q1 : Quelles sont les défenses les plus fortes contre les UAP sur ImageNet, et quel est leur coût en précision propre ?

### Takeaway
Sur ImageNet complet, les défenses dédiées les mieux documentées sont (a) l'entraînement adversarial universel (UAT, Shafahi et al.) : ResNet-101 passe de 7,23 % à 74,43 % d'acc. top-1 sous UAP (ε=10), sans chiffre de précision propre publié pour ce ResNet-101 ; sur AlexNet, il coûte environ 7 points de précision propre (56,4→49,5 %). (b) La régénération sélective de caractéristiques (SFR, Borkar et al.) : ResNet-152 à 0,731 sous sPGD en boîte blanche, contre 0,79 en propre. (c) Democratic Training (ICLR 2025) : ResNet-50, acc. sous UAP ciblée de 13,4 % → 61,7 %, pour −2 points de précision propre. Les défenses de type pré-traitement ou détecteur (PRN) annoncent environ 97 % de « defense rate », mais seulement contre des UAP non adaptatives. Le coût en précision propre va de ~0–2 points (SFR, Democratic Training, ensemble Co et al.) à 5–10 points (UAT/shared AT, TRADES).

### Cited Findings

**1. Fine-tuning avec UAP, papier original (Moosavi-Dezfooli, Fawzi, Fawzi, Frossard, CVPR 2017, « Universal adversarial perturbations », arXiv 1610.08401)**
- Méthode : fine-tuning de VGG-F pendant 5 époques supplémentaires sur un jeu d'entraînement où une UAP, tirée d'un pool de 10 UAP pré-calculées, est ajoutée à une fraction des images. La direction de v est fixe et l'amplitude α est adaptative (le plus petit α qui trompe le modèle) **[vérifié]** — [arXiv 1610.08401](https://arxiv.org/abs/1610.08401)
- Menace : une **nouvelle** UAP (Algorithme 1, p=∞, ξ=10) est recalculée contre le réseau fine-tuné, donc l'attaque est adaptative au sens « UAP recalculée » **[vérifié]** — [arXiv 1610.08401](https://arxiv.org/abs/1610.08401)
- Résultat : FR de 93,7 % → 76,2 % ; une seconde itération donne 80,0 %, et répéter la procédure n'améliore pas au-delà de 76,2 % **[vérifié]** — [arXiv 1610.08401](https://arxiv.org/abs/1610.08401)
- Coût : « minor increase in the error rate on the validation set » (note 4), sans chiffre précis **[vérifié]**. Les auteurs concluent que la méthode « does not fully immune against small universal perturbations » — [arXiv 1610.08401](https://arxiv.org/abs/1610.08401)

**2. Universal Adversarial Training (Shafahi, Najibi, Xu, Dickerson, Davis, Goldstein, AAAI 2020, arXiv 1811.11304)**
- Méthode : min-max `min_w E[L(w, x+δ)]` sur un seul δ partagé (‖δ‖∞ ≤ ε). L'attaque utilise un SGD stochastique sur δ avec une perte d'entropie croisée « clippée » (paramètre β). Alg. 3 : alternance d'une mise à jour FGSM-like de δ (signe du gradient) et d'une mise à jour des poids. Alg. 4 « low-cost » : mise à jour simultanée de δ et des poids dans la même rétropropagation, donc un coût « almost no extra cost » par rapport à l'entraînement naturel **[vérifié]** — [arXiv 1811.11304](https://arxiv.org/abs/1811.11304)
- Menace : UAP recalculées contre le modèle durci, en boîte blanche, avec leur attaque SGD et iDeepFool **[vérifié]** — [arXiv 1811.11304](https://arxiv.org/abs/1811.11304)
- ImageNet, AlexNet, ε=10, Table 3 (top-1/top-5) **[vérifié]** :
  - naturel : propre 56,4/79,0 %, UAP 3,9/9,4 % ;
  - UAT : propre 49,5/72,7 %, UAP 42,0/65,8 % ;
  - UAT low-cost : propre 48,4/72,4 %, UAP 28,3/48,3 %.
  — [arXiv 1811.11304](https://arxiv.org/abs/1811.11304)
- ImageNet, ResNet-101 : naturel à 7,23 % d'acc. sous UAP ; UAT à 74,43 % top-1 et 92,00 % top-5 sous UAP **[vérifié]**. La précision propre du ResNet-101 UAT n'est pas donnée dans la version arXiv lue. Hyperparamètres : 90 époques, batch 256 — [arXiv 1811.11304](https://arxiv.org/abs/1811.11304)
- CIFAR-10, WRN-32-10, ε=8, Table 1 (UAP/propre) **[vérifié]** :
  - naturel : 9,2 %/95,2 % ;
  - FGSM-AT : 51,0 %/91,42 % ;
  - PGD-AT (Madry) : 86,1 %/87,25 % ;
  - uADAM : 91,6 %/94,28 % ;
  - uFGSM : 91,8 %/93,50 % ;
  - low-cost : 86,1 % robuste et 93,5 % propre, avec un temps d'entraînement « cut by half ».
  — [arXiv 1811.11304](https://arxiv.org/abs/1811.11304)
- Coût de l'attaque UAP sur ImageNet (Table 2) : iDeepFool 9856 s contre 482 s (InceptionV1), et 6076 s contre 953 s (VGG16) **[vérifié]** — [arXiv 1811.11304](https://arxiv.org/abs/1811.11304)
- Limite : un modèle UAT reste faible face aux attaques par instance. Contre PGD en boîte blanche : 17,21 % (UAT-FGSM) et 2,57 % (UAT-ADAM) **[vérifié]** — [arXiv 1811.11304](https://arxiv.org/abs/1811.11304)

**3. Shared Adversarial Training (Mummadi, Brox, Metzen, ICCV 2019, arXiv 1812.03705)**
- Méthode : le mini-batch de d images est découpé en d/s « heaps » de taille s (« sharedness »). Une perturbation partagée par heap est calculée par PGD avec une perte clippée, puis utilisée en adversarial training. Le paramètre σ mélange le risque propre et le risque adversarial. On obtient un risque intermédiaire entre l'AT standard (s=1) et l'UAT (s=d) **[vérifié]** — [arXiv 1812.03705](https://arxiv.org/abs/1812.03705)
- Menace : robustesse mesurée par εuni(δ), le plus petit ε∞ tel qu'une UAP (S-PGD avec recherche binaire, Foolbox étendu) atteigne un taux d'erreur ≥ δ. L'UAP est recalculée contre le modèle défendu, en boîte blanche, et la mesure est une **borne supérieure** de la vraie robustesse **[vérifié]** — [arXiv 1812.03705](https://arxiv.org/abs/1812.03705)
- CIFAR-10, ResNet20 :
  - ERM : 93,25 % de précision propre et εuni(0,75) = 14,9 ;
  - s=64 sans perte de précision : εuni = 22,7 ;
  - à 90 % de précision : εuni = 44,1, soit « nearly three times » la robustesse avec moins de 3,5 points de perte.
  — [arXiv 1812.03705](https://arxiv.org/abs/1812.03705) **[vérifié]**
- Sous-ensemble d'ImageNet (200 classes, WRN) :
  - modèle de base : 77,57 % top-1 ;
  - shared AT fait passer εuni(0,75) de 8,4 à 15,0 « without any loss of accuracy » ;
  - point de compromis : 72,74 % de précision pour εuni = 25,64 (s=32, ε=10, σ=1,0), soit environ −5 points.
  — [arXiv 1812.03705](https://arxiv.org/abs/1812.03705) **[vérifié]**
- Coût : shared AT 189 s, le même temps que l'AT standard (CIFAR-10, unité donnée telle quelle dans le texte) **[vérifié]** — [arXiv 1812.03705](https://arxiv.org/abs/1812.03705)
- Aussi en segmentation (Cityscapes, FCN-8) : UAP ciblée, εuni(0,95) ≤ 19,92 sur le modèle non défendu **[vérifié]** — [arXiv 1812.03705](https://arxiv.org/abs/1812.03705)

**4. Perturbation Rectifying Network, PRN (Akhtar, Liu, Mian, CVPR 2018, arXiv 1711.05929)**
- Méthode : des couches « pré-entrée » (PRN) sont placées devant le classifieur gelé et entraînées pour que la sortie du classifieur sur l'image rectifiée égale celle sur l'image propre. S'y ajoute un détecteur : la différence image−rectifiée est passée en DCT 2D puis dans un SVM binaire. Si une perturbation est détectée, la classification se fait sur l'image rectifiée. Les données d'entraînement sont des UAP réelles plus des UAP synthétiques générées à partir de ces UAP **[vérifié]** — [arXiv 1711.05929](https://arxiv.org/abs/1711.05929)
- Réseaux : CaffeNet, VGG-F, GoogLeNet (tableaux principaux) ; VGG-16 et ResNet-152 en transfert. UAP de Moosavi-Dezfooli, ℓ2 (ξ=2000) et ℓ∞ (ξ=10) **[vérifié]** — [arXiv 1711.05929](https://arxiv.org/abs/1711.05929)
- Résultats, GoogLeNet, même type de perturbation entre entraînement et test **[vérifié]** :
  - « defense rate » de 97,4 % (ℓ2, Prot-A) et 96,4 % (ℓ∞, Prot-A) ;
  - taux de détection de 94,6 % (ℓ2) et 98,5 % (ℓ∞) ;
  - « PRN-restoration » (images propres préservées) de 97,0 % et 95,6 %.
  — [arXiv 1711.05929](https://arxiv.org/abs/1711.05929)
- CaffeNet : defense rate de 96,4 % (ℓ2) et 95,2 % (ℓ∞). VGG-F : 95,5 % et 92,2 % (Prot-A) **[vérifié]** — [arXiv 1711.05929](https://arxiv.org/abs/1711.05929)
- Métrique : le « defense rate » est un rapport à la précision propre et non une précision absolue. Les UAP de test avaient un FR ≥ 80 % **[vérifié]** — [arXiv 1711.05929](https://arxiv.org/abs/1711.05929)
- Menace : **non adaptative.** Les auteurs recommandent de garder le PRN secret. Note 3 : « PRN+targeted network are end-to-end differentiable and the joint network can be susceptible to stronger attacks if PRN is not secretive. However, stronger perturbations are also more easily detectable by our detector. » **[vérifié]** — [arXiv 1711.05929](https://arxiv.org/abs/1711.05929)

**5. Selective Feature Regeneration, SFR (Borkar, Heide, Karam, CVPR 2020, arXiv 1906.03444)**
- Méthode : les filtres convolutifs sont classés par norme ℓ1 de leurs poids, utilisée comme proxy de leur vulnérabilité. Seuls les 50 % de filtres les plus vulnérables, dans quelques couches, sont « régénérés » par des unités résiduelles entraînables. Le DNN de base est gelé. Les unités sont entraînées sur des images propres et perturbées (UAP plus UAP synthétiques) **[vérifié]** — [arXiv 1906.03444](https://arxiv.org/abs/1906.03444)
- ImageNet, UAP ℓ∞ ξ=10, top-1 sous attaque, Table 1 **[vérifié]** :
  - CaffeNet (propre 56,4 %) : 0,109 → 0,542 ;
  - VGG-F (58,4 %) : 0,150 → 0,550 ;
  - GoogLeNet (68,6 %) : 0,260 → 0,653 ;
  - VGG-16 (68,4 %) : 0,240 → 0,655 ;
  - ResNet-152 (79 %) : 0,270 → 0,761.
  — [arXiv 1906.03444](https://arxiv.org/abs/1906.03444)
- Boîte blanche contre la défense, sPGD ℓ∞ ξ=10, ResNet-152 (Table 6) **[vérifié]** :
  - baseline 0,270 ; **SFR 0,731** ;
  - Feature Denoising 0,641 ; Adv. training 0,635 ; HGD 0,689 ;
  - Shared AT 0,727 (chiffre repris des auteurs de [45] et non ré-implémenté).
  — [arXiv 1906.03444](https://arxiv.org/abs/1906.03444)
- Attaque UAP « secondaire » en boîte blanche (CaffeNet, l'attaquant connaît le DNN et la défense, FR cible 0,85) : le FR atteint au maximum **0,42** après 600 époques d'attaque, ce que les auteurs présentent comme « 48 % lower » **[vérifié]** — [arXiv 1906.03444](https://arxiv.org/abs/1906.03444)
- Coût, d'après un tiers : SFR demande de pré-calculer 25 UAP (plus 2000 UAP synthétiques), et une UAP prend plus de 40 min avec l'implémentation de référence **[vérifié dans Sun et al. 2025]** — [arXiv 2502.05542](https://arxiv.org/abs/2502.05542)

**6. UAT avec perturbations par classe (Benz, Zhang, Karjauv, Kweon, ICME 2021, arXiv 2104.03000)**
- Méthode : UAT où l'on maintient une UAP par classe (indexée par la vraie classe) au lieu d'une seule UAP **[vérifié]** — [arXiv 2104.03000](https://arxiv.org/abs/2104.03000)
- Table 3, CIFAR-100 (propre/UAP) **[vérifié]** :
  - ResNet-18 : 76,5/6,6 ; PGD-AT 56,2/54,7 ; UAT 69,7/66,8 ; UAT par classe 70,3/68,2 ;
  - WRN-28-10 : 81,9/5,4 ; PGD-AT 60,5/59,8 ; UAT 76,0/71,8 ; UAT par classe 77,1/74,0.
  — [arXiv 2104.03000](https://arxiv.org/abs/2104.03000)
- CIFAR-10, WRN-28-10 avec UAT par classe : 94,9/94,8 **[vérifié]**. ImageNet-10 (10 classes) avec WRN-50 : la robustesse sous UAP passe de 81,3 % (UAT) à 90,4 % **[vérifié]**. Aucun résultat sur ImageNet-1k — [arXiv 2104.03000](https://arxiv.org/abs/2104.03000)

**7. Robustesse universelle des modèles biaisés texture/forme (Co, Muñoz-González, Kanthan, Glocker, Lupu, ICIP 2021, arXiv 1911.10364)**
- Résultat : les modèles entraînés sur Stylized-ImageNet ne sont pas plus robustes aux UAP. Le FR dépasse 90 % à ε=10 sur tous les modèles **[vérifié]**. Un ensemble texture+forme améliore un peu la robustesse — [arXiv 1911.10364](https://arxiv.org/abs/1911.10364)
- Table 1, ResNet-50 ImageNet, précision propre et précision sous la pire UAP à ε=10 **[vérifié]** :
  - IN : 76,13/7,50 ; SIN : 60,18/7,65 ; SIN+IN : 74,59/8,33 ;
  - ensemble hard : 73,24/17,25 ; **ensemble soft : 76,02/20,37**.
  — [arXiv 1911.10364](https://arxiv.org/abs/1911.10364)
- Menace : la « pire UAP » est le minimum sur les UAP déjà générées pour les modèles individuels, et non une UAP recalculée contre l'ensemble. L'évaluation est donc **non adaptative pour l'ensemble** (inférence d'après le texte du tableau) — [arXiv 1911.10364](https://arxiv.org/abs/1911.10364)
- Suite : la thèse de Co (Imperial, 2023) affirme que Jacobian regularisation + ensembles augmentent la robustesse aux UAP « without degrading predictive performance » **[non vérifié, résumé seulement]** — [Spiral Imperial](https://spiral.imperial.ac.uk/handle/10044/1/103574)

**8. Democratic Training (Sun, Sun, Zhao, ICLR 2025, arXiv 2502.05542)**
- Méthode : on observe que les UAP provoquent une chute d'entropie des couches internes. Le modèle est fine-tuné avec un petit jeu propre (≤ 5 % du jeu d'entraînement) pour transformer les échantillons à faible entropie en échantillons à forte entropie. L'architecture n'est pas modifiée **[vérifié]** — [arXiv 2502.05542](https://arxiv.org/abs/2502.05542)
- Table 2, UAP **ciblées**, acc. sous UAP et taux de succès (SR) avant → après **[vérifié]** :
  - ResNet-50 ImageNet : 0,134/0,714 → 0,617/0,002, avec ΔCAcc −0,02 et 8 min ;
  - VGG19 : 0,067/0,701 → 0,431/0,077, −0,04 ;
  - GoogLeNet : 0,195/0,584 → 0,549/0,004, −0,03 ;
  - moyenne sur 7 modèles : SR 79,4 % → 2,8 %, acc. 12,6 % → 69,3 %, environ −2 points de précision propre.
  — [arXiv 2502.05542](https://arxiv.org/abs/2502.05542)
- Comparaisons par les mêmes auteurs (moyennes, Tables 4–5) **[vérifié]** :
  - SFR : 0,468/0,011, ΔCAcc −0,022 ;
  - CFN : 0,150/0,559 ; FNS : 0,149/0,623 ;
  - DensePure : 0,802/0,010, ΔCAcc −0,121 ;
  - PGD-AT ciblé : 0,464/0,167, −0,104 ;
  - TRADES : 0,816/0,022, −0,110.
  — [arXiv 2502.05542](https://arxiv.org/abs/2502.05542)
- Effet de ε (ResNet-50) : à 15/255, acc. 0,027 → 0,364 et SR 0,913 → 0,115. La défense se dégrade donc nettement au-delà du budget d'entraînement **[vérifié]** — [arXiv 2502.05542](https://arxiv.org/abs/2502.05542)

**9. Détection d'UAP**
- Détecteur du PRN (DCT + SVM) : taux de détection de 81–98 % selon le réseau et la norme **[vérifié]** — [arXiv 1711.05929](https://arxiv.org/abs/1711.05929)
- Goel & Moulin, « Locally optimal detection of stochastic targeted UAPs » (arXiv 2012.04692 ; version IEEE TIFS 2022 « Fast locally optimal detection… ») : test LO-GLRT qui surpasserait le détecteur du PRN avec un temps d'exécution au moins 100× plus faible **[non vérifié, extrait de recherche]** — [arXiv 2012.04692](https://arxiv.org/abs/2012.04692), [Illinois Experts](https://experts.illinois.edu/en/publications/fast-locally-optimal-detection-of-targeted-universal-adversarial-/)
- Détecteur PCA + SVM sur les scores PCA de l'image, cité comme alternative interprétable **[non vérifié]** — [arXiv 2012.04692](https://arxiv.org/abs/2012.04692)
- HyperNeuron (Co et al., 2021, arXiv 2105.07334) : détection en temps réel de masques et patchs universels via les activations internes **[non vérifié]** — [arXiv 2105.07334](https://arxiv.org/abs/2105.07334)

**10. Filtrage fréquentiel et pré-traitement**
- Zhang et al., AAAI 2021, « UAPs through the lens of deep steganography: towards a Fourier perspective » : les UAP sont concentrées en hautes fréquences. Les auteurs en tirent une attaque « high-pass UAP » moins visible. C'est un papier d'attaque et d'analyse, pas de défense **[non vérifié, extrait]** — [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/16441)
- Une thèse TU Delft trouve un filtrage FFT « hardly better than a simple low-pass filter », avec un faible coût en précision. Ce n'est pas un benchmark dédié aux UAP **[non vérifié]** — [TU Delft](https://repository.tudelft.nl/file/File_331f4033-8e54-4c5b-9b18-9092ba7fe5ba)
- APR-Net (IEEE Trans. AI, 2025) : réseau de suppression d'UAP en pré-traitement, évalué sur ImageNet, sans modifier le classifieur **[non vérifié, aucun chiffre lu]** — [IEEE CSDL](https://www.computer.org/csdl/journal/ai/2025/04/10765144/224XRepwLvO)
- Chaubey et al. (survey 2020) : le PRN, entraîné uniquement sur des UAP de Moosavi-Dezfooli, défend mal contre des UAP data-prior, et de simples transformations d'entrée le surpassent **[non vérifié, extrait relayé]** — [arXiv 2005.08087](https://arxiv.org/abs/2005.08087)

**11. Évaluations en transfert de défenses génériques (Weng et al. 2023, arXiv 2306.11261)**
- ResNet-50 défendus (AT ℓ2=0,5, AT ℓ∞=1, SIN, SIN-IN, AugMix), avec des UAP apprises sur un ResNet-50 naturel (transfert, **non adaptatif**). Table 7 **[vérifié, lecture partielle des colonnes]** :
  - FR des modèles AT : ~10–17 % ;
  - FR de SIN : 26–45 % ; SIN-IN : 38–74 % ; AugMix : 26–63 %.
  — [arXiv 2306.11261](https://arxiv.org/abs/2306.11261)

### Inferences
- La défense la plus forte **évaluée en boîte blanche sur ImageNet-1k** est probablement UAT (ResNet-101 à 74,43 % sous UAP recalculée), ou SFR/shared AT (≈0,73 sous sPGD sur ResNet-152). Les protocoles diffèrent (ε=10, architectures, attaques), donc la comparaison directe est fragile.
- Les défenses du type « module en amont ou interne, classifieur gelé » (PRN, SFR) coûtent peu en précision propre, mais leur robustesse dépend fortement de l'attaque utilisée à l'entraînement et de son secret.
- Le coût en précision propre de l'adversarial training *universel* (≈5–7 points) est nettement plus faible que celui de l'AT par instance (≈10–20 points), pour une robustesse UAP similaire ou meilleure.

### Gaps
- Je n'ai trouvé aucun benchmark unifié des défenses UAP sur ImageNet-1k avec attaque adaptative commune.
- La précision propre du ResNet-101 UAT n'apparaît pas dans la version arXiv lue.
- Les chiffres de détection récents (LO-GLRT, HyperNeuron) n'ont pas été lus dans les PDF.

## Q2 : Quelles défenses ont été cassées par des attaques UAP adaptatives, et comment ?

### Takeaway
Je n'ai trouvé **aucun papier qui casse explicitement PRN ou SFR** avec une attaque adaptative complète (BPDA/EOT de bout en bout). Les signaux de fragilité sont les suivants : (1) les auteurs du PRN admettent eux-mêmes une vulnérabilité si le PRN n'est pas secret, puisque le système est différentiable de bout en bout ; (2) le PRN généralise mal à d'autres algorithmes d'UAP (survey Chaubey) ; (3) le fine-tuning de Moosavi-Dezfooli est « cassé » dès que l'UAP est recalculée (FR 76–80 %) ; (4) l'attaque secondaire contre SFR atteint quand même un FR de 0,42 ; (5) Democratic Training retombe à SR 23,6 % en moyenne (jusqu'à 40,9 %) face à une UAP recalculée.

### Cited Findings
- PRN : « the joint network can be susceptible to stronger attacks if PRN is not secretive » **[vérifié]** — [arXiv 1711.05929](https://arxiv.org/abs/1711.05929)
- PRN : faible performance contre des UAP générées par un autre algorithme (data-prior) **[non vérifié, extrait]** — [arXiv 2005.08087](https://arxiv.org/abs/2005.08087)
- Fine-tuning UAP : UAP recalculée, FR 76,2 % puis 80,0 % au second tour **[vérifié]** — [arXiv 1610.08401](https://arxiv.org/abs/1610.08401)
- SFR : avec une UAP en boîte blanche contre DNN+défense, le FR plafonne à 0,42 (cible 0,85) sur CaffeNet après 600 époques. En sPGD boîte blanche sur ResNet-152, la précision est de 0,731 **[vérifié]** — [arXiv 1906.03444](https://arxiv.org/abs/1906.03444)
- Democratic Training, attaque secondaire (DF-UAP recalculée sur le modèle réparé) **[vérifié]** — [arXiv 2502.05542](https://arxiv.org/abs/2502.05542) :
  - SR moyen de 23,6 % (de 5,8 % à 40,9 %) et acc. de 48,6 %, contre 81,3 %/14,7 % sans défense ;
  - sPGD adaptatif sur ResNet-50 : SR 16,5 %, acc. 48,7 % ;
  - attaque « avancée » qui contrôle l'entropie (perte (1−ρ)·CE − ρ·H) : selon les auteurs, SR < 1 % pour tout ρ.
- Défenses par normalisation des caractéristiques (CFN, Yu et al. 2021 ; FNS, Yu et al. 2023) : SR > 50 % contre des UAP ciblées selon Sun et al. Elles sont inefficaces, même sans adaptation **[vérifié dans Sun et al.]** — [arXiv 2502.05542](https://arxiv.org/abs/2502.05542)

### Inferences
- La différence d'acc. entre le cas « UAP pré-calculées » (61,7 %) et le cas « UAP recalculées » (48,0 % pour ResNet-50, Table 7) de Democratic Training montre qu'environ 14 points de robustesse disparaissent sous attaque adaptative.
- L'attaque « avancée » sur l'entropie est conçue par les auteurs de la défense eux-mêmes. Le fait qu'elle *affaiblisse* l'attaque suggère qu'elle n'est pas l'attaque adaptative optimale, donc la robustesse est probablement surestimée.
- Par analogie avec Athalye et al. 2018 (gradients obfusqués), le PRN est vraisemblablement contournable par une UAP optimisée à travers PRN+classifieur avec un terme qui évite le détecteur. Je n'ai trouvé aucune démonstration publiée ; c'est une inférence.

### Gaps
- Je n'ai trouvé aucune attaque adaptative publiée contre PRN ou SFR (recherche limitée à environ 15 requêtes).
- Le comportement d'UAT face à des UAP générées par des générateurs (GAP, NAG) sur ImageNet n'est pas documenté dans ce que j'ai lu.

## Q3 : L'entraînement adversarial universel est-il efficace contre des UAP adaptatives ? Chiffres.

### Takeaway
Oui, en partie : dans tous les papiers UAT et shared AT, l'UAP est **recalculée contre le modèle durci**. C'est par construction le scénario adaptatif « re-calcul d'UAP ». Sur ImageNet, AlexNet passe de 3,9 % à 42,0 % top-1 sous UAP pour −6,9 points de précision propre, et ResNet-101 de 7,23 % à 74,43 %. Shared AT double εuni sans perte de précision sur le sous-ensemble ImageNet-200.

### Cited Findings
- UAT, AlexNet ImageNet ε=10 : UAP 3,9 % → 42,0 % (Alg. 3) ou 28,3 % (low-cost) ; propre 56,4 → 49,5 / 48,4 % **[vérifié]** — [arXiv 1811.11304](https://arxiv.org/abs/1811.11304)
- UAT, ResNet-101 : 7,23 % → 74,43 % top-1 sous UAP **[vérifié]** — [arXiv 1811.11304](https://arxiv.org/abs/1811.11304)
- UAT CIFAR-10 : 9,2 % → 91,8 % sous UAP, propre 95,2 → 93,5 % **[vérifié]** — [arXiv 1811.11304](https://arxiv.org/abs/1811.11304)
- Shared AT, ImageNet-200 : εuni(0,75) 8,4 → 15,0 à précision égale, puis 25,64 à −5 points **[vérifié]** — [arXiv 1812.03705](https://arxiv.org/abs/1812.03705)
- Shared AT sur ResNet-152 sous sPGD boîte blanche : 0,727 (repris par Borkar et al.) **[vérifié chez Borkar, non vérifié dans Mummadi]** — [arXiv 1906.03444](https://arxiv.org/abs/1906.03444)
- UAT par classe : environ +1 à +3 points sous UAP par rapport à UAT sur CIFAR-100 ; ImageNet-10 de 81,3 % à 90,4 % **[vérifié]** — [arXiv 2104.03000](https://arxiv.org/abs/2104.03000)
- L'AT par instance protège aussi contre les UAP. CIFAR-10 PGD-AT : 86,1 % sous UAP mais 87,25 % propre **[vérifié]** — [arXiv 1811.11304](https://arxiv.org/abs/1811.11304). TRADES, d'après Sun et al. : acc. UAP 0,816 mais −11 points de précision propre **[vérifié]** — [arXiv 2502.05542](https://arxiv.org/abs/2502.05542)

### Inferences
- UAT est robuste à la *recomputation* d'UAP, mais l'attaquant est limité à une UAP unique. Rien ne garantit la robustesse contre des UAP par classe ou générées par réseau génératif. Benz et al. motivent d'ailleurs l'UAT par classe par cette faiblesse.
- La robustesse d'UAT sur AlexNet (42 % pour 49,5 % propre) laisse un écart d'environ 7,5 points entre précision propre et précision sous UAP. Sur ResNet-101, l'écart est probablement de quelques points, mais ce n'est pas vérifiable sans la précision propre.

### Gaps
- Aucun chiffre UAT sur ViT, ConvNeXt ou d'autres architectures modernes sur ImageNet-1k n'a été trouvé.
- Je n'ai pas trouvé de courbe de robustesse en fonction de ε pour UAT sur ImageNet.

## Q4 : Des papiers rapportent-ils une courbe de compromis coût/protection ?

### Takeaway
Oui. **Shared AT (Mummadi et al.)** est la référence : il donne des fronts de Pareto explicites « précision propre vs εuni » (Fig. 3 pour CIFAR-10, Fig. 4 pour ImageNet-200), en balayant la sharedness s, ε et σ. On trouve aussi des courbes partielles : nombre d'unités de régénération et nombre d'époques d'attaque (SFR), effet de ε (Democratic Training), clipping β (UAT, côté attaque), et FR en fonction de ε (Co et al.).

### Cited Findings
- Shared AT, Fig. 3 (CIFAR-10, s ∈ {1, 8, 64}) : Pareto front « clearly above all previous defenses ». Fig. 4 (ImageNet-200, s ∈ {1, 32}, ε ∈ {2,…,26}, σ variable) : « sweet spot » entre 67 % et 74 % de précision **[vérifié]** — [arXiv 1812.03705](https://arxiv.org/abs/1812.03705)
- Shared AT : le gain sature au-delà de s=64 (« diminishing returns ») **[vérifié]** — [arXiv 1812.03705](https://arxiv.org/abs/1812.03705)
- SFR : régénérer 50 % des filtres est le ratio empirique retenu. Pour VGG-16, 6 unités donnent la meilleure restoration accuracy (96,2 %) **[vérifié]** — [arXiv 1906.03444](https://arxiv.org/abs/1906.03444)
- Democratic Training, Table 6 : robustesse en fonction de ε (5, 10 et 15/255) et temps de réparation de 7 à 36 min **[vérifié]** — [arXiv 2502.05542](https://arxiv.org/abs/2502.05542)
- UAT : compromis coût de calcul/robustesse via la variante low-cost (coût d'entraînement environ divisé par deux sur CIFAR, robustesse 86,1 % au lieu de 91,8 %) **[vérifié]** — [arXiv 1811.11304](https://arxiv.org/abs/1811.11304)

### Inferences
- Aucune courbe ne porte sur le coût en *compute* d'un côté et la protection de l'autre, pour ImageNet-1k et sous attaque adaptative. C'est un manque exploitable pour le projet.

### Gaps
- Je n'ai pas vu les valeurs numériques exactes des fronts de Pareto (Table A1 du supplémentaire de Mummadi).
