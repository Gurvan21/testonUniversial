# Vulnérabilité des architectures modernes (ViT/DeiT, Swin, ConvNeXt, MaxViT, hybrides) aux perturbations universelles (UAP), comparée aux CNN (ResNet, VGG)

Notes de recherche, novembre 2026. Sauf mention contraire, chaque chiffre ci-dessous a été lu dans le texte intégral du PDF (arXiv), extrait avec pdftotext. Les chiffres qui viennent seulement d'un résumé ou d'un extrait de moteur de recherche sont signalés « (non vérifié) ».

Fiches de référence (utilisées dans les quatre sections) :

| Réf. | Lien | Méthode / modèles | UAP ? |
|---|---|---|---|
| Benz, Ham, Zhang, Karjauv, Kweon, « Adversarial Robustness Comparison of Vision Transformer and MLP-Mixer to CNNs », BMVC 2021 | [arXiv:2110.02797](https://arxiv.org/abs/2110.02797) | UAP de Zhang et al. CVPR 2020 (« Understanding adversarial examples from the mutual influence of images and perturbations »), ℓ∞ = 10/255, 224 px, jeu NeurIPS-2017. ViT-B/16 et ViT-L/16 (IN-21k → IN-1k), Mixer-B/L/16, RN18/RN50 (IN-1k et SWSL) | **Oui, matrice de transfert complète** |
| Bhojanapalli et al., « Understanding Robustness of Transformers for Image Classification », ICCV 2021 | [arXiv:2103.14586](https://arxiv.org/abs/2103.14586) | PGD/FGSM, corruptions, ablation de couches ; ViT vs ResNet (BiT), pré-entraînement IN-1k / IN-21k / JFT-300M | **Non** : « universal » n'apparaît que dans la bibliographie du PDF |
| Mahmood, Mahmood, van Dijk, « On the Robustness of Vision Transformers to Adversarial Examples », ICCV 2021 | [arXiv:2104.02610](https://arxiv.org/abs/2104.02610) | 6 attaques boîte blanche et 2 boîte noire ; ViT, BiT, CNN ; CIFAR-10/100, ImageNet | Non (transfert par image) |
| Naseer et al., « Intriguing Properties of Vision Transformers », NeurIPS 2021 (spotlight) | [arXiv:2105.10497](https://arxiv.org/abs/2105.10497) | Occlusion, biais forme/texture, SIN ; **patch adversarial universel non ciblé** (Brown et al.), FGSM, PGD ; DeiT-T/S/B, T2T-24, TnT-S, ResNet50, variantes SIN | **Patch universel** (non borné), pas de UAP ℓp |
| Shao, Shi, Yi, Chen, Hsieh, « On the Adversarial Robustness of Vision Transformers », TMLR 10/2022 | [arXiv:2103.15670](https://arxiv.org/abs/2103.15670) | PGD, AutoAttack, transfert FGSM, analyse fréquentielle, lissage aléatoire débruité ; ViT, T2T, hybrides, ConvNeXt, Swin, MLP-Mixer | Non |
| Bai, Mei, Yuille, Xie, « Are Transformers More Robust Than CNNs? », NeurIPS 2021 | [arXiv:2111.05464](https://arxiv.org/abs/2111.05464) | Comparaison à recette égale DeiT-S vs ResNet-50 : PGD/AutoAttack, Texture Patch Attack (TPA), IN-C, Stylized-IN | Non (TPA = patch par image) |
| Paul & Chen, « Vision Transformers are Robust Learners », AAAI 2022 | [arXiv:2105.07581](https://arxiv.org/abs/2105.07581) | ViT vs BiT : corruptions, IN-P, Fourier, PGD, spectre des perturbations DeepFool | **Non** : aucune occurrence de « universal » dans le PDF |
| Fu et al., « Patch-Fool: Are Vision Transformers Always Robust Against Adversarial Perturbations? », ICLR 2022 | [arXiv:2203.08392](https://arxiv.org/abs/2203.08392) | Attaque par image sur un seul patch, guidée par l'attention | Non universelle |
| Gu, Tresp, Qin, « Are Vision Transformers Robust to Patch Perturbations? », ECCV 2022 | [arXiv:2111.10659](https://arxiv.org/abs/2111.10659) | Patchs naturels vs adversariaux, ViT/DeiT vs ResNet | Patchs universels testés mais jugés trop faibles (non vérifié) |
| « IAM-UAP: Inheritance Attention Matrix-Based UAPs on Vision Transformers », IEEE SPL 28:1923–1927, 2021 | [DOI 10.1109/LSP.2021.3112099](https://api.openalex.org/works/doi:10.1109%2FLSP.2021.3112099) | UAP qui perturbe l'intégration globale de l'attention | Oui (résumé seulement) |
| AS-UAP, « Attention-Shift UAP on Transformer-based Models », ACM 2025 | [ACM DL](https://dl.acm.org/doi/10.1145/3743093.3771000) | UAP qui décale l'attention ; ViT, PiT, CaiT, DeiT, DeiT-distill | Oui (le PDF renvoie HTTP 403, non vérifié) |
| Co et al., « Universal Adversarial Robustness of Texture and Shape-Biased Models », ICIP 2021 | [arXiv:1911.10364](https://arxiv.org/abs/1911.10364) | SGD-UAP sur ResNet50 IN / SIN / IN+SIN / IN+SIN→IN | Oui (CNN seulement) |
| Singh, Croce, Hein, « Revisiting Adversarial Training for ImageNet: Architectures, Training and Generalization across Threat Models », NeurIPS 2023 | [arXiv:2303.01870](https://arxiv.org/abs/2303.01870) | Entraînement adversarial ConvNeXt/ViT, ConvStem, recette | Non |

## Q1 — Taux de tromperie UAP pour ViT/DeiT/Swin/ConvNeXt vs ResNet à budget comparable (ℓ∞ 10/255 ou 16/255)

### Takeaway
À ℓ∞ = 10/255 en boîte blanche, le seul tableau vérifié (Benz et al. 2021) ne montre **pas d'écart net** : le taux de succès (ASR) est d'environ 85–94 % pour ViT-B/16 et ViT-L/16 comme pour ResNet-18/50. Mixer est à environ 99 %. Je n'ai trouvé **aucun tableau publié et vérifiable** de fooling rate UAP pour Swin, ConvNeXt ou MaxViT, ni de chiffre à 16/255 sur les Transformers.

### Cited Findings
- Benz et al. 2021, tableau 7, ASR UAP en boîte blanche (ℓ∞ = 10/255, 224 px, jeu NeurIPS-2017) : ViT-B/16 **93,9 %**, ViT-L/16 **84,8 %**, Mixer-B/16 98,6 %, Mixer-L/16 98,9 %, RN18 93,7 %, RN50 91,9 %, RN18-SWSL 91,2 %, RN50-SWSL 87,1 % — [Benz et al.](https://arxiv.org/abs/2110.02797)
- Leur conclusion, citée : « comparing ViT and CNN models, there is no obvious robustness gap with an ASR of around 90% for all models. However, in the more challenging black-box scenarios, the ViT models are significantly more robust ». ViT-L/16 est le modèle le plus robuste aux UAP — [Benz et al.](https://arxiv.org/abs/2110.02797)
- Les ViT de Benz et al. sont pré-entraînés sur IN-21k. Les auteurs écrivent que « many other factors other than the architecture itself might also have a role… it is nearly impossible to rule out all other factors » — [Benz et al.](https://arxiv.org/abs/2110.02797)
- IAM-UAP (SPL 2021) : son résumé annonce un taux d'attaque « modéré » sur les ViT et désigne la taille de patch comme facteur latent de robustesse aux UAP. Aucun chiffre lu (non vérifié) — [IAM-UAP](https://api.openalex.org/works/doi:10.1109%2FLSP.2021.3112099)
- AS-UAP (2025) : fooling rate moyen de 79,59 % sur des Transformers, contre 68,36 % pour FG-UAP (2e meilleure méthode). L'extrait cite 83,11 % sur DeiT et 83,34 % sur DeiT-distill. Le budget ε n'a pas été vu (non vérifié, HTTP 403) — [AS-UAP](https://dl.acm.org/doi/10.1145/3743093.3771000)
- Référence CNN : UAP original (Moosavi-Dezfooli et al., CVPR 2017), ℓ∞ ξ = 10 (sur [0,255]) : plus de 90 % sur CaffeNet et VGG-F, environ 80–85 % sur VGG-16/19, GoogLeNet et ResNet-152 — [Moosavi-Dezfooli et al. 2017](https://openaccess.thecvf.com/content_cvpr_2017/papers/Moosavi-Dezfooli_Universal_Adversarial_Perturbations_CVPR_2017_paper.pdf)
- Patch adversarial universel (non borné) : selon la figure 12 de Naseer et al., « ViTs even with less parameters exhibit a higher robustness than CNN » (DeiT-T/S/B, T2T-24 vs ResNet50), et les modèles IN sont plus robustes que les modèles SIN. Résultats moyennés sur 5 exécutions sur IN-val. Les valeurs sont dans une figure et n'ont pas été extraites — [Naseer et al.](https://arxiv.org/abs/2105.10497)
- Patchs adversariaux par image : les ViT sont **plus** vulnérables que les CNN, à la fois à Patch-Fool ([Fu et al.](https://arxiv.org/abs/2203.08392)) et aux patchs adversariaux de Gu et al. Gu et al. attribuent ce résultat à l'attention qui se concentre sur le patch. D'après un résumé de recherche, ils jugent aussi les patchs universels trop faibles sur un patch ViT d'environ 0,5 % de l'image (non vérifié) — [Gu et al.](https://arxiv.org/abs/2111.10659)
- Benchmarks UAP récents (DM-UAP 2503.12793, SGA 2308.06015, PSP-UAP 2502.21048) : un grep du texte intégral ne trouve aucune mention de ViT, DeiT, Swin ou ConvNeXt. Ces travaux restent centrés sur les CNN — [2503.12793](https://arxiv.org/abs/2503.12793), [2308.06015](https://arxiv.org/abs/2308.06015), [2502.21048](https://arxiv.org/abs/2502.21048)

### Inferences
- En boîte blanche, avec un algorithme UAP fort et ε = 10/255, les ViT pré-entraînés ne sont pas « immunisés » : leur ASR est de 85–94 %, comparable aux ResNet. L'avantage robuste des ViT vient surtout du transfert (Q2). Pour vos expériences maison (ConvNeXt-B, Swin-B, MaxViT-T, ViT-B/16), il faut s'attendre à des fooling rates élevés en boîte blanche. L'écart avec ResNet dépendra surtout du pré-entraînement et de la recette.
- Patch universel non borné et UAP ℓ∞ dense ne mesurent pas la même chose. Les résultats « ViT plus vulnérables aux patchs » (Patch-Fool, Gu et al.) portent sur des attaques par image et ne se transposent pas directement aux UAP denses.

### Gaps
- Aucun fooling rate UAP vérifié pour Swin, ConvNeXt, MaxViT, CoAtNet ou DeiT-III à ε = 10/255 ou 16/255. Termes cherchés : « universal adversarial perturbation Swin ConvNeXt DeiT fooling rate », « UAP vision transformers fooling rate », « transferable universal adversarial perturbation CNN to vision transformer ».
- Les chiffres précis d'IAM-UAP et d'AS-UAP (ε, modèles, CNN de référence) n'ont pas pu être vérifiés : PDF inaccessible ou seul le résumé disponible.
- Bhojanapalli et al. et Paul & Chen, souvent cités sur ce sujet, **n'évaluent pas** les UAP : vérifié par grep du PDF.

## Q2 — Les UAP transfèrent-elles entre CNN et Transformers ?

### Takeaway
**Très peu des CNN vers les ViT, un peu plus des ViT vers les CNN.** Dans Benz et al., les UAP calculées sur des ResNet ne trompent que 11–13,5 % des ViT. Les UAP calculées sur ViT-B/16 trompent 29–46 % des ResNet. Les perturbations par image montrent la même asymétrie (Shao et al., Mahmood et al.). Les modèles de type Mixer sont une exception : ils sont trompés à plus de 95 % par toute UAP source.

### Cited Findings
- Tableau 7 de Benz et al. (ASR %, ℓ∞ 10/255), transfert CNN → ViT : RN18 → ViT-B/16 13,5 et ViT-L/16 11,6 ; RN50 → 12,2 et 12,0 ; RN18-SWSL → 12,6 et 12,1 ; RN50-SWSL → 11,8 et 11,9 — [Benz et al.](https://arxiv.org/abs/2110.02797)
- Transfert ViT → CNN : ViT-B/16 → RN18 46,4, RN50 37,2, RN18-SWSL 43,3, RN50-SWSL 29,2 ; ViT-L/16 → RN18 41,6, RN50 31,7. Entre ViT : B → L 32,7 et L → B 42,1 — [Benz et al.](https://arxiv.org/abs/2110.02797)
- Transfert CNN → CNN (même famille) : RN18 → RN50 63,3, RN50 → RN18 63,4, RN18 → RN18-SWSL 66,2 — [Benz et al.](https://arxiv.org/abs/2110.02797)
- Mixer : toute source donne plus de 95 % sur Mixer-B/L (par ex. RN50 → Mixer-B 95,4). En revanche, les UAP Mixer transfèrent mal vers les ViT : Mixer-B → ViT-B 16,6 — [Benz et al.](https://arxiv.org/abs/2110.02797)
- Par image (FGSM), Shao et al. trouvent un taux d'erreur faible quand les CNN sont la source et les ViT la cible. À l'inverse, « when the ViTs are the source models, the adversarial examples they generate have higher RER when transferred ». Ils l'expliquent par le fait que les ViT s'appuient moins sur des motifs haute fréquence (« robust features ») — [Shao et al.](https://arxiv.org/abs/2103.15670)
- Mahmood et al. : « adversarial examples do not readily transfer between CNNs and transformers ». Un ensemble CNN + ViT donne une robustesse boîte noire « unprecedented » mais reste cassé en boîte blanche par leur Self-Attention Blended Gradient Attack — [Mahmood et al.](https://arxiv.org/abs/2104.02610)
- Pour des attaques transférables par image, Token Gradient Regularization (CVPR 2023) note une forte baisse du transfert ViT → CNN liée à l'écart architectural — [Zhang et al. CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Transferable_Adversarial_Attacks_on_Vision_Transformers_With_Token_Gradient_Regularization_CVPR_2023_paper.pdf). Voir aussi [Wei et al. AAAI 2022 (PNA/PatchOut)](https://cdn.aaai.org/ojs/20169/20169-13-24182-1-2-20220628.pdf) et [Ma et al. ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Ma_Transferable_Adversarial_Attack_for_Both_Vision_Transformers_and_Convolutional_Networks_ICCV_2023_paper.pdf) (par image, pas UAP)
- Générateur universel inter-architectures (JAD, « Latent Danger Zone », 2025) : générateur à diffusion latente guidé par les cartes d'attention CNN et ViT. Le transfert est annoncé fort dans les deux sens (non vérifié, résumé seulement) — [arXiv:2509.19044](https://arxiv.org/html/2509.19044)

### Inferences
- L'asymétrie (ViT → CNN plus forte que CNN → ViT) a la même direction pour les UAP et pour les attaques par image. Cela va dans le sens de l'hypothèse « les UAP CNN exploitent des motifs haute fréquence que les ViT ignorent ».
- Les ensembles ou le vote CNN + ViT sont un candidat naturel de défense contre des UAP transférées, mais pas contre un attaquant qui connaît l'ensemble.

### Gaps
- Aucune matrice de transfert UAP vérifiée qui inclut Swin, ConvNeXt ou MaxViT. Les architectures hybrides (ConvNeXt est un CNN « modernisé » à stem patchify 4×4, MaxViT mélange convolution et attention) sont précisément celles qui manquent.

## Q3 — Les ViT sont-ils moins biaisés vers la texture, et cela change-t-il les labels dominants ou l'apparence des UAP (grille 16×16) ?

### Takeaway
- **Apparence : oui.** Benz et al. observent un motif en tuiles aligné sur les patchs dans les UAP de ViT et de Mixer. Les UAP de ViT-L/16, le modèle le plus robuste, sont aussi les plus lisses localement.
- **Biais de texture : oui pour les ViT** (Naseer et al.). Mais rendre un modèle plus biaisé vers la forme (SIN) ne le protège pas contre les UAP ou les patchs universels, que ce soit un CNN (Co et al.) ou un ViT (Naseer et al.).
- **Labels dominants :** aucune étude trouvée qui compare les labels dominants des UAP entre CNN et ViT.

### Cited Findings
- Benz et al., figure 6 : « for the ViT and Mixer architectures, a tile pattern is observable, which is caused by the operation of dividing the images into patches as tokens ». Les UAP de Mixer sont « less locally smooth ». ViT-L/16 a « the most locally smooth patterns » — [Benz et al.](https://arxiv.org/abs/2110.02797)
- Toujours selon Benz et al., les ViT s'appuient davantage sur les basses fréquences, les CNN sur les hautes fréquences. Ils relient aussi la vulnérabilité des CNN à l'invariance par translation — [Benz et al.](https://arxiv.org/abs/2110.02797)
- Perturbations par image : Bhojanapalli et al. (figure 1) notent que les perturbations adversariales d'un Transformer et d'un ResNet sont « qualitatively quite different » — [Bhojanapalli et al.](https://arxiv.org/abs/2103.14586). Paul & Chen trouvent les perturbations PGD « smoother in case of ViT » (ViT L-16 vs BiT-m r101x3, annexe J). Le spectre DeepFool est plus étalé en énergie pour ViT — [Paul & Chen](https://arxiv.org/abs/2105.07581)
- Shao et al. : les caractéristiques des ViT contiennent moins de motifs haute fréquence, et les ViT sont « less sensitive to high-frequency perturbations ». Ajouter des blocs convolutionnels ou T2T, qui apprennent des caractéristiques bas niveau, réduit cette robustesse — [Shao et al.](https://arxiv.org/abs/2103.15670)
- Naseer et al. : les ViT sont beaucoup moins biaisés vers la texture que les CNN. Pour le patch universel, DeiT-S-SIN (plus biaisé forme) est **moins** robuste que DeiT-S (figure 30), et ResNet50-SIN suit la même tendance. Les patchs universels optimisés pour DeiT-T/B et T2T-24 sont visualisés en figure 31 — [Naseer et al.](https://arxiv.org/abs/2105.10497)
- Co et al. (ResNet50) : les modèles entraînés sur Stylized-ImageNet sont « still as vulnerable to these UAPs as models trained on ImageNet ». Le biais forme « shift[s] the set of features the models are vulnerable to ». Un ensemble texture + forme améliore la robustesse universelle — [Co et al.](https://arxiv.org/abs/1911.10364)

### Inferences
- Pour vos UAP maison sur ViT-B/16 et Swin-B, on peut s'attendre à des artefacts de grille alignés sur la taille de patch : 16 px pour ViT-B/16, 4 px pour le stem de Swin et de ConvNeXt. L'effet devrait être plus faible pour les architectures à stem convolutionnel (MaxViT). C'est une hypothèse à vérifier en calculant une FFT des UAP (pics à la fréquence 224/16 = 14 cycles).
- Le biais forme réduit ne protège pas contre les UAP. Il est donc plausible que le label dominant change entre architectures (autres caractéristiques exploitées), mais aucune source ne l'établit.

### Gaps
- Aucun travail trouvé sur la distribution des labels dominants des UAP pour les ViT par rapport aux CNN. Termes cherchés : « dominant label universal perturbation vision transformer », et requêtes UAP/ViT générales.
- Aucune analyse spectrale publiée qui quantifie la grille des UAP selon la taille de patch (seulement des visualisations qualitatives).

## Q4 — Défenses par bruit aléatoire contre les UAP : comparaisons CNN vs Transformers ?

### Takeaway
**Non trouvé.** Je n'ai trouvé aucune étude qui compare une défense par bruit aléatoire (bruit gaussien ou uniforme en entrée, lissage aléatoire) **contre des UAP** entre CNN et Transformers. Les travaux voisins existent : la certification par lissage aléatoire débruité ViT vs CNN (Shao et al.) et la randomisation pour ViT (CVPR 2024). Ils portent sur des attaques par image.

### Cited Findings
- Shao et al. appliquent le lissage aléatoire débruité (Salman et al. 2020, δ ~ N(0, σ²I)). Selon eux, « ViTs attain significantly better certified robustness » que les CNN (tableau 5, chiffres non extraits). Ce sont des garanties ℓ2 par image, pas des résultats UAP — [Shao et al.](https://arxiv.org/abs/2103.15670)
- Random Entangled Tokens (CVPR 2024) : les défenses par randomisation, efficaces sur CNN, transfèrent mal aux ViT à cause des différences d'architecture — [Gong et al. CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Gong_Random_Entangled_Tokens_for_Adversarially_Robust_Vision_Transformer_CVPR_2024_paper.pdf)
- Défense UAP par ajout de bruit gaussien puis débruitage par diffusion, en télédétection, sans comparaison ViT/CNN ciblée — [arXiv:2307.16865](https://arxiv.org/html/2307.16865v3)
- Rusak et al. : l'entraînement au bruit gaussien et l'entraînement adversarial universel, sur CNN, sous l'angle des corruptions — [arXiv:2001.06057](https://arxiv.org/abs/2001.06057)
- Robustesse au bruit naturel, utile pour estimer le coût en précision d'une défense par bruit : sur ImageNet-P, flip rate « Gauss » 7,36 % pour ViT L-16 contre 11,93 % pour BiT-m r101x3 — [Paul & Chen](https://arxiv.org/abs/2105.07581). L'ImageNet-C mCE de DeiT-S (54,6) est meilleur que celui de ResNet50 (76,7) — [Naseer et al.](https://arxiv.org/abs/2105.10497)

### Inferences
- Les ViT tolèrent mieux le bruit gaussien naturel (IN-P/IN-C) et ont un meilleur certificat par lissage débruité. On peut donc supposer qu'une défense « bruit aléatoire en entrée » coûte moins de précision propre sur ViT, ce qui permet un σ plus grand à précision égale. C'est une hypothèse, non un résultat publié. Votre expérience `eval_noise.py` sur ResNet18/50, ConvNeXt-B, Swin-B, MaxViT-T et ViT-B/16 semble donc combler un manque réel de la littérature.

### Gaps
- Termes cherchés sans résultat pertinent : « random noise defense universal adversarial perturbation vision transformer CNN comparison Gaussian noise input », ainsi que des variantes sur la randomisation et le lissage. Je n'ai pas cherché spécifiquement le lissage aléatoire certifié contre les UAP (Universal randomized smoothing) sur ViT, ce qui reste une piste.

## Rôle de la recette d'entraînement vs l'architecture (transversal)

### Takeaway
Une bonne partie de la « robustesse des ViT » vient du pré-entraînement et de la recette, pas seulement de l'architecture. À recette égale, ResNet-50 peut égaler DeiT-S en robustesse adversariale (Bai et al.). Pour ConvNeXt, la recette pèse plus que le stem en ℓ∞ (Singh et al.). Aucune étude ne fait cette ablation spécifiquement pour les UAP.

### Cited Findings
- Bai et al. : « CNNs can easily be as robust as Transformers on defending against adversarial attacks, if they properly adopt Transformers' training recipes ». En revanche, la meilleure généralisation hors distribution des Transformers « is largely benefited by the Transformer's self-attention-like architectures per se » — [Bai et al.](https://arxiv.org/abs/2111.05464)
- Bhojanapalli et al. : les ViT « pre-trained with a sufficient amount of data » sont au moins aussi robustes que les ResNet — [Bhojanapalli et al.](https://arxiv.org/abs/2103.14586)
- Gu et al. : l'écart de robustesse aux patchs ne peut pas être attribué à la seule architecture, car les réglages d'entraînement diffèrent (non vérifié dans le PDF) — [Gu et al.](https://arxiv.org/abs/2111.10659)
- Singh, Croce, Hein (entraînement adversarial ImageNet) : remplacer le stem patchify de ConvNeXt-T par un ConvStem améliore surtout ℓ2/ℓ1 et peu ℓ∞ (gain d'environ 0,9 point annoncé). Le pré-entraînement fort, l'augmentation lourde et l'entraînement plus long augmentent la robustesse (chiffres issus d'un extrait, à revérifier) — [Singh et al.](https://arxiv.org/abs/2303.01870)
- Benz et al. : dans le tableau 7, les ResNet SWSL (pré-entraînés sur IG-1B) sont un peu plus robustes aux UAP en boîte blanche que les ResNet IN-1k (RN50-SWSL 87,1 contre RN50 91,9) — [Benz et al.](https://arxiv.org/abs/2110.02797)

### Inferences
- Pour comparer des UAP entre architectures de façon équitable, il faut séparer le pré-entraînement (IN-1k, IN-21k, JFT) et la recette (DeiT, ConvNeXt, recette torchvision « v2 »). Les modèles torchvision `*_10k` de vos résultats n'ont pas tous la même recette : vit_b_16 et swin_b sont entraînés IN-1k avec la recette moderne, et ResNet50 a une recette V1 ou V2 selon les poids. C'est un facteur de confusion à documenter.
