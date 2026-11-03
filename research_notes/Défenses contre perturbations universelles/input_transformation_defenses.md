# Défenses par transformation d'entrée, randomisation, débruitage et certification contre les perturbations universelles (UAP)

Portée : classification d'images, priorité ImageNet. Les papiers sont séparés en (A) **évalués contre des UAP** et (B) **contexte : évalués uniquement contre des attaques par image**. Les chiffres marqués « vérifié » ont été lus dans le texte PDF du papier (extraction pdftotext). Les autres viennent des abstracts ou de résumés de recherche et sont signalés comme tels.

Notation : FR = fooling rate (taux de changement de label). « Restoration accuracy » (Akhtar / Borkar) = fraction des images (propres + perturbées) dont le label est celui de l'image propre.

---

## Q1 — Quels papiers testent le bruit gaussien ou des transformations aléatoires à l'inférence contre des UAP, et avec quels résultats ?

### Takeaway
Je n'ai trouvé **aucun papier qui évalue l'injection de bruit gaussien/uniforme additif à l'inférence comme défense contre des UAP sur ImageNet**. Les évaluations d'input-transformation contre les UAP sont surtout **déterministes** (flou gaussien, médian, bilatéral, JPEG, bit-depth, TV, quilting, ondelettes). Elles réduisent le FR en partie seulement, souvent au prix de 5 à 18 % de précision propre. La seule défense *aléatoire* évaluée contre des UAP sur ImageNet est R&P (random resize + padding, Xie et al.), et seulement dans un cadre de transfert boîte noire (Li et al., RHP).

### Cited Findings

**A. Papiers évalués contre des UAP**

1. **Mopuri, Ganeshan, Babu — « Generalizable Data-free Objective for Crafting Universal Adversarial Perturbations » (GD-UAP), IEEE TPAMI 2019 (arXiv 2018)** — [arXiv 1801.08092](https://arxiv.org/abs/1801.08092)
   - Méthode : évalue 8 transformations d'entrée (10-crop, flou gaussien, lissage médian, filtrage bilatéral, bit-depth 3 et 2 bits, JPEG 75 % et 50 %, minimisation TV, image quilting) plus PRN (Akhtar 2018). Les cibles sont des UAP data-free (GD-UAP sans prior, range prior, data prior) et l'UAP de Moosavi-Dezfooli. Modèle : GoogLeNet sur ImageNet (Table 10). — [arXiv 1801.08092](https://arxiv.org/abs/1801.08092)
   - Threat model : **non adaptatif**. Les UAP sont calculées sur le modèle nu, puis la défense est appliquée. Rien n'indique une ré-optimisation à travers la défense. — [arXiv 1801.08092](https://arxiv.org/abs/1801.08092)
   - Résultats vérifiés (Table 10, GoogLeNet ; colonnes : précision top-1 propre avec la défense / FR de l'UAP Moosavi-Dezfooli « UAP [8] » / FR GD-UAP data-prior) : sans défense 69.74 / 78.5 / 83.54 ; flou gaussien 58.12 / 32.78 / 32.66 ; médian 51.78 / 37.02 / 43.96 ; bilatéral 60.28 / 22.74 / 22.34 ; bit 3 bits 63.20 / 69.62 / 72.30 ; bit 2 bits 45.50 / 65.40 / 60.50 ; JPEG 75 % 67.50 / 42.62 / 61.38 ; JPEG 50 % 64.16 / 31.40 / 44.62 ; TV 50.50 / 26.22 / 30.9 ; quilting 51.30 / 25.84 / 34.80 ; PRN 68.90 / 21.35 / 52.50. — [arXiv 1801.08092](https://arxiv.org/abs/1801.08092)
   - Résultats vérifiés (Table 11, GD-UAP range prior, 6 réseaux, %ΔAcc = perte relative de précision propre) : JPEG 50 % : CaffeNet ΔAcc 5.23 %, FR 87.1→72.1 ; GoogLeNet ΔAcc 8 %, FR 71.4→41.8 ; VGG-16 ΔAcc 2.6 %, FR 63.1→49.2 ; VGG-19 ΔAcc 2.4 %, FR 64.7→64.7 (inchangé) ; ResNet-152 ΔAcc 4.7 %, FR 37.3→37.3 (inchangé). Bilatéral : ΔAcc 9.7 à 18.2 %, FR ramené à 21.5–35.8. — [arXiv 1801.08092](https://arxiv.org/abs/1801.08092)
   - Conclusion des auteurs : « in majority of the defenses, UAPs flip labels for more than 45% of the images ». Quand la défense baisse fortement le FR, « a huge price is paid in terms of % drop in Top-1 Accuracy », d'où « the poor fit of input transformations as a viable defense ». Ils notent aussi que PRN généralise mal aux UAP qu'il n'a pas vues à l'entraînement. — [arXiv 1801.08092](https://arxiv.org/abs/1801.08092)
   - Limites : pas d'attaque adaptative, réseaux anciens (Caffe), un seul niveau par transformation, pas de bruit additif aléatoire.

2. **Akhtar, Liu, Mian — « Defense against Universal Adversarial Perturbations » (PRN), CVPR 2018** — [arXiv 1711.05929](https://arxiv.org/abs/1711.05929) ; [CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Akhtar_Defense_Against_Universal_CVPR_2018_paper.pdf)
   - Méthode : un « Perturbation Rectifying Network » est placé avant le classifieur gelé, avec un détecteur (SVM sur des features DCT de la différence image/rectifiée). Il est entraîné sur des UAP réelles et des perturbations synthétiques. ImageNet ; CaffeNet, VGG-16, GoogLeNet, ResNet-152 ; ℓ∞ ξ=10 et ℓ2 ξ=2000. — [arXiv 1711.05929](https://arxiv.org/abs/1711.05929)
   - Résultats vérifiés : « defense rate » de 87.9 à 97.5 % et « detection rate » de 81.2 à 98.6 % selon le réseau et la norme (Tables 1 à 3). Coût sur les images propres : « unnecessary rectification of the clean images can sometimes lead to a minor (1–2%) reduction » de précision. — [arXiv 1711.05929](https://arxiv.org/abs/1711.05929)
   - Threat model : l'attaquant ne connaît pas la défense (la défense est supposée « secrète »). Ce n'est pas une transformation fixe mais un débruiteur appris.
   - Limite (vérifiée dans Borkar 2020) : sous une attaque UAP « secondaire » white-box à travers PRN, le FR atteint **0.87**. — [arXiv 1906.03444](https://arxiv.org/abs/1906.03444)

3. **Borkar, Heide, Karam — « Defending Against Universal Attacks Through Selective Feature Regeneration » (FRN), CVPR 2020** — [arXiv 1906.03444](https://arxiv.org/abs/1906.03444)
   - Méthode : des unités de régénération de features sont placées sur les 50 % de filtres les plus vulnérables (classés par la norme ℓ1 des noyaux). C'est une défense dans l'espace des features, comparée à plusieurs défenses côté entrée. — [arXiv 1906.03444](https://arxiv.org/abs/1906.03444)
   - Résultats vérifiés (Table 2, restoration accuracy, UAP ℓ∞ ξ=10, ILSVRC2012 ; ordre CaffeNet / VGG-F / GoogLeNet / VGG-16 / Res152) : baseline 0.596/0.628/0.691/0.681/0.670 ; **JPEG 0.554/0.697/0.830/0.693/0.670** (aucun gain sur Res152, perte sur CaffeNet) ; Pixel Deflection 0.873/0.813/0.884/0.818/0.845 ; Feature Distillation 0.671/0.689/0.851/0.717/0.676 ; HGD (Res152) 0.739 ; adv. training (Res152) 0.778 ; FD (Res152) 0.819 ; PRN 0.936/0.903/0.956/0.690/0.834 ; FRN 0.976/0.967/0.970/0.963/0.982. — [arXiv 1906.03444](https://arxiv.org/abs/1906.03444)
   - Résultats vérifiés (Table 4, CaffeNet, ξ plus forts que celui de l'entraînement) : à ξ=15/20/25, PRN tombe à 0.603/0.555/0.526 et PD à 0.616/0.549/0.524, quasiment au niveau baseline (0.543/0.525/0.519). FRN garde 0.952/0.896/0.854. — [arXiv 1906.03444](https://arxiv.org/abs/1906.03444)
   - Threat model : surtout non adaptatif. Voir Q2 pour l'attaque secondaire white-box.

4. **Li, Bai, Xie, Liao, Shen, Yuille — « Regional Homogeneity: Towards Learning Transferable Universal Adversarial Perturbations Against Defenses » (RHP), ECCV 2020 (arXiv 2019)** — [arXiv 1904.00979](https://arxiv.org/abs/1904.00979)
   - Défenses attaquées : TVM, HGD, **R&P (random resizing + padding, Xie et al. 2018)**, trois modèles ensemble-adv-trained, PGD-AT, ALP, Feature Denoising. Les perturbations sont transférées depuis Inception-v3 nu, en boîte noire et **sans EOT**. — [ar5iv 1904.00979](https://ar5iv.labs.arxiv.org/html/1904.00979)
   - Résultats (Table 2, hausse de l'erreur top-1 sur 5000 images ILSVRC, ε=16/32 ; lu via ar5iv et non recoupé avec le PDF) : contre R&P, l'UAP standard ne fait que **+2.42 / +6.66** points d'erreur, RHP **+23.3 / +56.1**. TVM : 4.78/12.1 contre 33.0/56.9. HGD : 1.94/11.3 contre 26.8/57.5. — [ar5iv 1904.00979](https://ar5iv.labs.arxiv.org/html/1904.00979)
   - Lecture : une UAP standard (haute fréquence) est presque neutralisée par la randomisation R&P en transfert. Une UAP « régionalement homogène » (basse fréquence, par blocs) passe largement. Les défenses par transformation reposent donc sur la *structure spectrale* de l'UAP, pas sur une propriété robuste. Incohérence interne signalée : l'abstract annonce +14.0 % de gain moyen, l'introduction +21.6 %. — [ar5iv 1904.00979](https://ar5iv.labs.arxiv.org/html/1904.00979)

5. **Ghosh, Mullick, Datta, Das, Mallipeddi, Das — « A Black-box Adversarial Attack Strategy with Adjustable Sparsity and Generalizability for Deep Image Classifiers » (DEceit), Pattern Recognition 2022 (arXiv 2020)** — [arXiv 2004.13002](https://arxiv.org/abs/2004.13002)
   - UAP boîte noire par évolution différentielle, évaluée contre 8 débruitages : ondelettes, JPEG 50/75 %, TV, NL-means, bilatéral, médian, flou gaussien. ImageNet ; VGG16 et Inception-v3. Non adaptatif. — [arXiv 2004.13002](https://arxiv.org/abs/2004.13002)
   - Résultats vérifiés (Table 2, FR %, VGG16, UAP dense ρ=50176, β=10) : sans défense 83.96 ; ondelettes 76.88 ; JPEG 50 % 71.65 ; JPEG 75 % 70.52 ; TV 72.14 ; NL-means 71.91 ; bilatéral 75.23 ; médian 77.95 ; flou gaussien 76.87. Inception-v3 : 83.03 → 70.3–77.2 selon la défense. **Baisse de FR de 6 à 13 points seulement.** La précision propre sous défense n'est pas rapportée. — [arXiv 2004.13002](https://arxiv.org/abs/2004.13002)

6. **Wu, Zhou, Liu, Liu, Zhu — « Decision-based Universal Adversarial Attack » (DUAttack), arXiv 2020** — [arXiv 2009.07024](https://arxiv.org/abs/2009.07024)
   - Évalue une UAP decision-based contre spatial smoothing (Xu et al.), JPEG (Das et al.), TVM (Guo et al.) et l'adversarial training. Le résumé indique une précision ramenée vers ~20 % à la plus grande norme. Je n'ai pas extrait les chiffres exacts de la figure, donc **non vérifié**. — [arXiv 2009.07024](https://arxiv.org/abs/2009.07024)

7. **Co, Muñoz-González, de Maupeou, Lupu — « Procedural Noise Adversarial Examples for Black-Box Attacks on Deep Convolutional Networks », ACM CCS 2019 (arXiv 2018)** — [arXiv 1810.00470](https://arxiv.org/abs/1810.00470)
   - UAP procédurales (Perlin, Gabor) sur ImageNet. Défense préliminaire par filtrage médian. Résultat vérifié : le débruitage réduit le taux d'évasion universel de seulement **7.2–10.8 points (Gabor)** et **13.2–16.9 points (Perlin)** pour ε∞ ∈ {4,8,12,16}/255. Les auteurs appellent à des défenses « input-agnostic ». — [arXiv 1810.00470](https://arxiv.org/abs/1810.00470)
   - Suite (procedural noise, Simplex/Worley) : le résumé de recherche indique que les défenses par débruitage sont faibles et que JPEG et l'ensemble-AT sont plus efficaces. **Non vérifié dans le texte.** — [arXiv 2108.04409](https://arxiv.org/abs/2108.04409)

8. **Weng, Luo, Lin, Zhong, Shen, Li — « Comparative Evaluation of Recent Universal Adversarial Perturbations in Image Classification », Computers & Security 2024 (arXiv 2023)** — [arXiv 2306.11261](https://arxiv.org/abs/2306.11261)
   - Seule défense d'entrée évaluée : **NRP** (Neural Representation Purifier, Naseer 2020), un purificateur génératif, avec un ResNet-50 ImageNet et des UAP transférées depuis le ResNet-50 nu (non adaptatif). — [arXiv 2306.11261](https://arxiv.org/abs/2306.11261)
   - Résultats vérifiés (Table 7, FR non ciblé %, UAP sur ImageNet) : NRP 17.69 à 46.47 selon la loss (FFF 46.47 ; cos 23.19 ; CW 27.85 ; NCE 17.91 ; RCE 17.69). À titre de comparaison, AT ℓ∞=1 donne 10.5–12.8, SIN-IN 28.8–74.4 et AugMix 41.2–63.5. Les FR sans défense ne sont pas répétés dans cette table. Les auteurs concluent que l'adversarial training est la défense la plus efficace. — [arXiv 2306.11261](https://arxiv.org/abs/2306.11261)

9. **Moosavi-Dezfooli, Fawzi, Fawzi, Frossard — « Universal adversarial perturbations », CVPR 2017** — [CVF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Moosavi-Dezfooli_Universal_Adversarial_Perturbations_CVPR_2017_paper.pdf) ; [arXiv 1610.08401](https://arxiv.org/abs/1610.08401)
   - Une perturbation aléatoire de même norme fait un FR d'environ 10 % seulement. Le fine-tuning sur des images perturbées ramène le FR de VGG-F de 93.7 % à 76.2 %, et le réseau reste vulnérable. Ces chiffres viennent du résumé de recherche et n'ont pas été relus dans le PDF. — [CVF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Moosavi-Dezfooli_Universal_Adversarial_Perturbations_CVPR_2017_paper.pdf)

10. **Agarwal, Singh, Vatsa, Ratha — « Image Transformation-Based Defense Against Adversarial Perturbation on Deep Learning Models », IEEE TDSC 2021** (DOI 10.1109/TDSC.2020.3027183) — [IITJ](https://research.iitj.ac.in/publication/image-transformation-based-defense-against-adversarial)
    - Filtrage par décomposition en ondelettes. Le même groupe a travaillé sur les UAP (BTAS 2018, détection d'UAP en reconnaissance faciale). **Je n'ai pas pu vérifier** que ce papier TDSC évalue des UAP spécifiquement, ni ses chiffres. À vérifier.

11. **Diffusion / purification générative** : je n'ai trouvé **aucune évaluation de DiffPure (Nie et al., ICML 2022) contre des UAP sur un classifieur ImageNet**. — [arXiv 2205.07460](https://arxiv.org/abs/2205.07460). Il existe des travaux « universal adversarial defense » par diffusion en télédétection (UAD-RS), où « universal » veut dire « une seule défense pour plusieurs attaques » et non « UAP ». Ils sont donc hors sujet au sens strict. — [arXiv 2307.16865](https://arxiv.org/abs/2307.16865)

**B. Contexte : évalués uniquement contre des attaques par image (pas d'UAP)**
- Xie et al., « Mitigating Adversarial Effects Through Randomization », ICLR 2018 (random resize + padding). — [arXiv 1711.01991](https://arxiv.org/abs/1711.01991)
- Guo et al., « Countering Adversarial Images using Input Transformations », ICLR 2018 (crop-rescale, bit-depth, JPEG, TVM, quilting). — [arXiv 1711.00117](https://arxiv.org/abs/1711.00117)
- Athalye, Carlini, Wagner, « Obfuscated Gradients Give a False Sense of Security », ICML 2018. BPDA et EOT contournent Xie 2018 et Guo 2018 (attaques par image). — [arXiv 1802.00420](https://arxiv.org/abs/1802.00420)
- Athalye et al., « Synthesizing Robust Adversarial Examples » (EOT), ICML 2018. — [PMLR](http://proceedings.mlr.press/v80/athalye18b/athalye18b.pdf)
- Gao et al., « On the Limitations of Stochastic Pre-processing Defenses », NeurIPS 2022. Analyse le compromis invariance / robustesse des défenses aléatoires (par image). — [arXiv 2206.09491](https://arxiv.org/abs/2206.09491)
- Raff et al., « Barrage of Random Transforms », CVPR 2019 (évalué avec EOT, par image). — [CVF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Raff_Barrage_of_Random_Transforms_for_Adversarially_Robust_Defense_CVPR_2019_paper.pdf)
- Das et al., SHIELD (JPEG), KDD 2018 (par image). — [arXiv 1802.06816](https://arxiv.org/abs/1802.06816)
- Liao et al., HGD, CVPR 2018 (par image, mais inclus comme baseline contre UAP par Borkar 2020 et Li 2020). — [CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Liao_Defense_Against_Adversarial_CVPR_2018_paper.pdf)
- Nie et al., DiffPure, ICML 2022 (par image, avec BPDA+EOT). — [arXiv 2205.07460](https://arxiv.org/abs/2205.07460)
- Cohen, Rosenfeld, Kolter, « Certified Adversarial Robustness via Randomized Smoothing », ICML 2019. Certificat ℓ2 par image ; 49 % de précision certifiée top-1 sur ImageNet à r=0.5. — [arXiv 1902.02918](https://arxiv.org/abs/1902.02918)

### Inferences
- Sur ImageNet, les transformations déterministes (JPEG, flou, médian, bit-depth) réduisent le FR d'une UAP non adaptative de 6 à 50 points selon l'UAP et le réseau. Les gros gains (bilatéral, TV, quilting, flou) coûtent 8 à 19 points de précision propre absolue sur GoogLeNet (d'après la Table 10 de GD-UAP). JPEG est le meilleur compromis mais donne un gain nul sur ResNet-152 et VGG-19 (Table 11 de GD-UAP ; Table 2 de Borkar).
- L'efficacité dépend du contenu fréquentiel de l'UAP. Les UAP de Moosavi-Dezfooli, haute fréquence, sont bien filtrées. Les UAP basse fréquence ou régionalement homogènes (RHP), ou procédurales, résistent beaucoup mieux (RHP ; Co et al.).
- Le bruit additif gaussien à l'inférence n'a, à ma connaissance, **pas été publié comme défense anti-UAP évaluée sur ImageNet**. C'est une lacune réelle de la littérature, pertinente pour ce projet (cf. `python/eval_noise.py`).

### Gaps
- Pas de papier trouvé sur bruit gaussien ou uniforme additif à l'inférence, ni sur randomized smoothing *empirique*, contre des UAP ImageNet (ResNet, ViT, ConvNeXt, Swin).
- Pas de chiffres vérifiés pour DUAttack, Agarwal TDSC 2021 et procedural noise 2108.04409.
- Les FR « sans défense » de Weng 2023 (Table 7) n'ont pas été recoupés dans la même table.
- Recherche non exhaustive dans les surveys (Chaubey 2020, Zhang IJCAI 2021) : les PDF ont été téléchargés, mais leur section défense n'a pas été dépouillée faute de budget.

---

## Q2 — Un papier évalue-t-il ces défenses contre un attaquant UAP adaptatif (UAP recalculée avec EOT à travers la défense) ? Que se passe-t-il ?

### Takeaway
**Aucun papier trouvé n'évalue une défense par transformation aléatoire (bruit, R&P, JPEG aléatoire) contre une UAP recalculée avec EOT ou BPDA à travers la défense.** Deux résultats proches existent. (1) Les UAP calculées avec EOT sur une distribution de transformations restent très efficaces sous ces transformations (RobustUAP, Xu & Singh), alors que les UAP standard s'effondrent. (2) Les défenses apprises (PRN) sont cassées par une UAP recalculée en white-box à travers la défense, avec un FR de 0.87.

### Cited Findings
- **Xu & Singh — « Robust Universal Adversarial Perturbations » (RobustUAP), arXiv 2022 (version ICML 2024 selon les versions ultérieures, venue non vérifiée)** — [arXiv 2206.10858](https://arxiv.org/abs/2206.10858)
  - Méthode : génère des UAP robustes à une distribution de transformations (rotation, translation, scaling, shear, luminosité/contraste, bruit uniforme, corruptions « PRIME »), avec des garanties probabilistes d'ASR. Les baselines « SGD UAP » et « Standard UAP_RP » sont explicitement décrites comme des « naive combinations of the EoT and UAP algorithms ». — [arXiv 2206.10858](https://arxiv.org/abs/2206.10858)
  - Résultats vérifiés (Table 1, robust ASR, ILSVRC 2012, Inception-v3) : sous rotation R(20°), UAP standard **0.0 %**, SGD-UAP (≈EOT) 69.9 %, UAP_RP 2.9 %, **RobustUAP 93.2 %**. Sous l'ensemble affine composé R(10), T(2,2), Sh(2), Sc(2), B : 0.0 / 63.1 / 2.9 / **86.4 %**. — [arXiv 2206.10858](https://arxiv.org/abs/2206.10858)
  - Résultats vérifiés (Table 2) : une RobustUAP entraînée sur PRIME garde un ASR de 72.1 % sous bruit gaussien, 81.3 % shot, 88.6 % impulse et 73.3 % JPEG (corruptions de type ImageNet-C). — [arXiv 2206.10858](https://arxiv.org/abs/2206.10858)
  - Section 5.1 « Robustness to Random Noise » : UAP robustes au bruit uniforme, avec résultats en Appendice M (**chiffres non extraits**). — [arXiv 2206.10858](https://arxiv.org/abs/2206.10858)
  - Défense (5.6) : un entraînement contre des RobustUAP donne un robust ASR de 0.4 % pour 90.1 % de précision, contre 0.2 % / 83.5 % pour l'AT standard (CIFAR-10 a priori, dataset non vérifié dans l'extrait). — [arXiv 2206.10858](https://arxiv.org/abs/2206.10858)
  - Ce papier n'est pas présenté comme l'évaluation d'une *défense*, mais il en est l'analogue : une défense « appliquer une transformation aléatoire T avant le classifieur » est exactement ce que RobustUAP contourne.
- **Borkar et al. CVPR 2020, attaque « secondaire » white-box** : une UAP (algorithme Moosavi-Dezfooli, CaffeNet) recalculée avec les gradients de la défense atteint un FR de **0.87 contre PRN**. Contre FRN, elle plafonne à **0.42 après 600 epochs**, mais FRN est entraîné itérativement avec des exemples adversariaux. Sur un modèle nu, la même attaque converge en moins de 2 epochs (FR 0.9). FRN n'est pas une transformation aléatoire, donc pas d'EOT. — [arXiv 1906.03444](https://arxiv.org/abs/1906.03444)
- **Li et al. (RHP)** : contre R&P (randomisé), l'UAP n'est **pas** calculée avec EOT, mais transférée. Elle passe néanmoins grâce à sa structure basse fréquence (+56.1 points d'erreur à ε=32). — [ar5iv 1904.00979](https://ar5iv.labs.arxiv.org/html/1904.00979)
- **GD-UAP, DEceit, Weng 2023** : tous non adaptatifs (perturbations calculées sur le modèle nu). — [arXiv 1801.08092](https://arxiv.org/abs/1801.08092) ; [arXiv 2004.13002](https://arxiv.org/abs/2004.13002) ; [arXiv 2306.11261](https://arxiv.org/abs/2306.11261)
- Contexte (par image) : Athalye et al. 2018 montrent que l'EOT et le BPDA ramènent à ~0 % la robustesse de Xie 2018 (R&P) et de Guo 2018. — [arXiv 1802.00420](https://arxiv.org/abs/1802.00420). Gao et al. 2022 montrent que la randomisation de pré-traitement n'apporte de robustesse que si elle réduit l'invariance, au détriment de l'utilité. — [arXiv 2206.09491](https://arxiv.org/abs/2206.09491)

### Inferences
- La table 1 de RobustUAP suggère fortement qu'une UAP standard calculée sans EOT est fragile aux transformations aléatoires : 0 % d'ASR sous R(20). Le défenseur gagne donc beaucoup contre un attaquant *naïf*. Une UAP EOT (même naïve, « SGD UAP ») retrouve 63–70 %, et une RobustUAP 86–93 %. Le gain du défenseur contre un attaquant adaptatif est donc très réduit. C'est une inférence par analogie, pas une évaluation directe d'une défense gaussienne.
- Par extension des résultats par image (Athalye, Gao), on peut s'attendre à ce qu'une défense par bruit gaussien contre UAP soit largement contournée par une UAP EOT, sauf si le niveau de bruit est assez fort pour dégrader nettement la précision propre. Ce n'est pas démontré dans la littérature UAP.

### Gaps
- Aucune évaluation publiée trouvée « UAP + EOT à travers bruit gaussien / R&P / JPEG différentiable » sur ImageNet, avec comparaison au cas non adaptatif.
- Venue exacte de RobustUAP non confirmée (arXiv 2206.10858 vérifié).

---

## Q3 — Un papier rapporte-t-il la courbe « coût en précision propre vs protection » contre des UAP (p. ex. en fonction du niveau de bruit) ?

### Takeaway
Je n'ai trouvé **aucune courbe paramétrique continue** (par exemple FR vs σ du bruit). La meilleure source est GD-UAP (TPAMI 2019) : des **points discrets** (précision propre sous défense et FR pour 10 transformations sur GoogLeNet, puis %ΔAcc et FR pour JPEG 50 % et bilatéral sur 6 réseaux). Ils montrent que les forts gains de protection coûtent cher en précision propre.

### Cited Findings
- GD-UAP Table 10 (GoogLeNet, vérifié) : précision propre 69.74 sans défense. Les défenses qui ramènent le FR de l'UAP de Moosavi-Dezfooli sous 35 % font chuter la précision propre à 50.5–60.3 (TV 50.50 → FR 26.22 ; quilting 51.30 → 25.84 ; bilatéral 60.28 → 22.74 ; flou gaussien 58.12 → 32.78). JPEG 50 % garde 64.16 de précision pour un FR de 31.40. — [arXiv 1801.08092](https://arxiv.org/abs/1801.08092)
- GD-UAP Table 11 (6 réseaux, vérifié) : JPEG 50 % coûte 0.5 à 8 % relatifs de précision pour un gain de FR de 0 à 29.6 points. Le bilatéral coûte 9.7 à 18.2 % relatifs pour un gain de 11.9 à 57 points. — [arXiv 1801.08092](https://arxiv.org/abs/1801.08092)
- GD-UAP Table 10 : bit-depth 3 → 2 bits donne un mini-« balayage » : précision 63.20 → 45.50, FR (UAP Moosavi-Dezfooli) 69.62 → 65.40. Le FR ne baisse presque pas alors que la précision s'effondre. — [arXiv 1801.08092](https://arxiv.org/abs/1801.08092)
- DEceit : JPEG 75 % vs 50 % est un mini-balayage (FR VGG16 70.52 vs 71.65), mais sans précision propre. — [arXiv 2004.13002](https://arxiv.org/abs/2004.13002)
- Akhtar 2018 : coût de PRN sur les images propres de 1–2 %. — [arXiv 1711.05929](https://arxiv.org/abs/1711.05929)
- Borkar 2020 Table 4 : courbe de *protection vs force d'attaque ξ* (et non vs niveau de défense). — [arXiv 1906.03444](https://arxiv.org/abs/1906.03444)
- RobustUAP 5.6 : un point AT-RobustUAP (ASR 0.4 %, accuracy 90.1 %) contre l'AT standard (0.2 %, 83.5 %), dans un cadre d'entraînement et non de transformation d'entrée. — [arXiv 2206.10858](https://arxiv.org/abs/2206.10858)

### Inferences
- Une courbe « σ du bruit → (précision propre, FR non adaptatif, FR adaptatif EOT) » sur ImageNet pour des modèles modernes serait, à ma connaissance, une contribution nouvelle.

### Gaps
- Pas de courbe paramétrique trouvée pour bruit gaussien, R&P ou force de JPEG contre des UAP, ni dans les papiers ni dans les surveys parcourus.

---

## Q4 — Existe-t-il des certificats contre les perturbations universelles spécifiquement (et non une certification ℓp par image) ?

### Takeaway
**Oui, mais seulement à petite échelle (MNIST, CIFAR-10, TinyImageNet) et par vérification déterministe (LiRPA/IBP + MILP), pas par randomized smoothing.** Les deux références principales sont Zeng et al. (ICLR 2023, certification) et Xu & Singh (ECCV 2024, entraînement certifié CITRUS). Je n'ai trouvé aucun certificat UAP sur ImageNet ni de randomized smoothing « universel » en vision. En texte, CR-UTP (2024) le fait pour les LLM.

### Cited Findings
- **Zeng, Shi, Jin, Kang, Lyu, Hsieh, Jia — « Towards Robustness Certification Against Universal Perturbations », ICLR 2023** — [OpenReview](https://openreview.net/forum?id=7GEvPKxjtt) ; [PDF NSF-PAR](https://par.nsf.gov/servlets/purl/10498346) ; code : github.com/ruoxi-jia-group/Universal_Pert_Cert
  - Méthode : relaxation linéaire (auto_LiRPA) par échantillon, puis un MILP qui impose que **la même perturbation δ soit partagée** par tout le batch. On obtient le « certified robust rate » (CRR) d'un batch, avec une borne probabiliste sur la population. Les auteurs l'appliquent aussi aux backdoors / triggers universels. — [PDF NSF-PAR](https://par.nsf.gov/servlets/purl/10498346)
  - Résultats vérifiés : MNIST naturel, Conv-small, ε=8/255 : CRR par échantillon 52.33 % contre **63.33 %** pour le certificat universel. Conv-4-layer, ε=8/255 : 40.67 contre **52.0**. CIFAR-10 PGD-8, ResNet-4B, ε=1/255 : 51.00 contre **54.67**. À ε=3/255, les deux méthodes certifient **0 %** pour ResNet-4B. — [PDF NSF-PAR](https://par.nsf.gov/servlets/purl/10498346)
  - Table 4 (vérifiée ; légende « ResNet-4B, CIFAR-10 », ‖δ‖∞=16/255) : CRR : modèle normal 0.33 %, FGSM-AT 27.33, PGD-AT 30.67, UAT-FGSM 24.00, UAT-PGD 28.99, **IBP 95.66**. **Attention** : l'Ori-ACC de ~99 % rapportée dans cette table est incompatible avec CIFAR-10 et ressemble plutôt à MNIST. C'est une possible erreur de légende du papier, à vérifier. — [PDF NSF-PAR](https://par.nsf.gov/servlets/purl/10498346)
  - Limites : petits réseaux (MNIST, CIFAR) ; coût du MILP qui croît avec la taille du batch ; seulement des modèles adversarialement entraînés sur CIFAR (le modèle naturel est jugé trop vulnérable) ; pas d'ImageNet ; pas de transformations d'entrée ni de smoothing. Les auteurs citent le randomized smoothing (Cohen 2019) uniquement pour les backdoors.
- **Xu & Singh — « Cross-Input Certified Training for Universal Perturbations » (CITRUS), ECCV 2024** — [arXiv 2405.09176](https://arxiv.org/abs/2405.09176)
  - Entraînement certifié (IBP) qui minimise la perte pire-cas d'une seule perturbation partagée par toutes les entrées. Abstract (vérifié) : jusqu'à +10.3 % de précision standard par rapport à l'entraînement certifié classique, et précision certifiée UAP « state of the art ». — [arXiv 2405.09176](https://arxiv.org/abs/2405.09176)
  - Chiffre trouvé uniquement dans un résumé de recherche, **non vérifié** dans le PDF : 35.62 % de précision certifiée UAP sur TinyImageNet, contre 1.51 % pour les baselines UAT. — [arXiv html 2405.09176](https://arxiv.org/html/2405.09176v1)
- Hors vision : **CR-UTP**, certification contre les « Universal Text Perturbations » pour LLM (2024), basée sur le smoothing. — [arXiv 2406.01873](https://arxiv.org/abs/2406.01873)
- Faux amis : UniCR (ECCV 2022) est un randomized smoothing « universel » au sens de n'importe quelle norme ℓp et n'importe quel bruit, avec certificat par image, pas UAP. — [arXiv 2207.02152](https://arxiv.org/abs/2207.02152). De même, « Universal Asymmetric Randomization » (2025) n'est pas un certificat UAP. — [arXiv 2510.19977](https://arxiv.org/abs/2510.19977)
- Remarque triviale (inférence logique, pas un papier) : un certificat par image (p. ex. le randomized smoothing de Cohen 2019, ℓ2) implique un certificat universel au même rayon, puisqu'une UAP de norme ≤ r est un cas particulier. La précision certifiée par image est donc une **borne inférieure** de la précision certifiée universelle (Zeng 2023 montre que cette borne est lâche). — [arXiv 1902.02918](https://arxiv.org/abs/1902.02918) ; [PDF NSF-PAR](https://par.nsf.gov/servlets/purl/10498346)

### Inferences
- Pour ImageNet, le seul certificat applicable aujourd'hui aux UAP serait le certificat par image du randomized smoothing gaussien (borne inférieure). Un certificat universel exploitant le partage de δ sous smoothing (de type « collective certification », Schuchardt et al. ICLR 2021, conçue pour les graphes et la segmentation) semble ne pas exister pour la classification d'images. Je n'ai pas vérifié l'existence d'une extension de Schuchardt aux UAP.

### Gaps
- Pas de certificat universel trouvé sur ImageNet, ni de certificat universel fondé sur le randomized smoothing en vision.
- Chiffres de CITRUS non vérifiés dans le PDF.
- Possible erreur de légende dans la Table 4 de Zeng 2023 (Ori-ACC ~99 % annoncé pour CIFAR-10).
- Collective robustness certificates (Schuchardt, Bojchevski, Klicpera, Günnemann, ICLR 2021) : non vérifié s'ils traitent un cas UAP en classification d'images.
