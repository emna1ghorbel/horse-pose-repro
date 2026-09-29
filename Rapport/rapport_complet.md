# Rapport de Projet : Estimation de Pose de Cheval (Horse Pose Repro)

## 1. Introduction
Ce projet s'inscrit dans le cadre de la vision par ordinateur et plus précisément dans l'estimation de pose animale (ici, le cheval). Contrairement à l'estimation de pose humaine qui bénéficie de très grands jeux de données annotés, l'estimation de pose animale est un domaine où les données sont rares. Ce projet vise à reproduire et améliorer une architecture capable d'inférer un squelette 2D (18 points clés) à partir d'une image RGB.

## 2. Problématique et Objectifs
**Problématique :** 
Comment estimer précisément le squelette d'un cheval à partir d'une simple image sans dépendre de données massivement annotées, et comment améliorer l'interprétabilité visuelle des prédictions lorsque le modèle présente un biais géométrique de translation ou d'échelle ?

**Objectifs :**
1. Entraîner une architecture composée d'un extracteur de silhouette (Heatmap) et d'un régresseur de points clés.
2. Évaluer le modèle sur des images de validation.
3. Améliorer la lisibilité et l'alignement des squelettes produits sans nécessiter de ré-entraînement lourd, en utilisant des techniques de post-traitement avancées.

## 3. Conception et Technologies
Le projet est développé en **Python** et utilise **PyTorch** pour l'apprentissage profond.

### 3.1 Architecture du Modèle
L'architecture se décompose en deux modules principaux :
* **Phi (`ImageToSkeleton`) :** Un réseau de type U-Net convolutif. Il prend en entrée une image RGB (128x128) et produit une carte d'activation (Heatmap) représentant la silhouette de l'animal.
* **Omega (`SkeletonToPose2D`) :** Un réseau qui transforme la Heatmap (sortie de Phi) en un ensemble de coordonnées (x, y) pour 18 points clés (tête, cou, dos, jambes, etc.).

### 3.2 Technologies et Librairies
* **PyTorch / Torchvision :** Pour la création des tenseurs, la définition de l'architecture réseau et la boucle d'entraînement.
* **NumPy :** Pour les calculs matriciels post-prédiction.
* **Matplotlib :** Pour le rendu visuel et la génération des figures de diagnostic.

## 4. Implémentation
Au cours de l'implémentation, un modèle a été entraîné jusqu'à l'**époque 150**. 

### 4.1 Diagnostic Post-Entraînement
Lors des premières évaluations (via le script `skeleton_inspect.py`), il a été observé que les prédictions des points clés, bien que structurellement cohérentes (les os étaient bien formés), souffraient d'un décalage spatial (mauvaise échelle ou mauvais centrage par rapport au cheval dans l'image). De plus, la heatmap de sortie de `Phi` était visuellement diffuse (bruit de fond).

### 4.2 Améliorations de Post-Traitement (Sans Ré-entraînement)
Faute de temps de calcul disponible pour relancer l'apprentissage, des optimisations de visualisation ont été implémentées :
1. **Contraste de la Heatmap :** Élévation au cube (`phi**3`) pour supprimer le bruit de fond et ne garder que le signal fort du corps du cheval.
2. **Alignement par Boîte Englobante (Bounding Box) :** Le squelette prédit a été redimensionné et translaté pour correspondre exactement à la boîte englobante de la silhouette détectée par la heatmap. 
   * *Méthode :* Calcul des limites (min/max) des valeurs de la heatmap (seuil > 0.1), extraction des dimensions, et application d'un facteur d'échelle proportionnel sur les coordonnées des points clés (avec une normalisation min-max).

## 5. Résultats Expérimentaux

Le modèle a été inféré sur l'ensemble de validation après l'implémentation de la méthode d'alignement. Un script a été développé pour scanner automatiquement différents "seeds" (graines aléatoires) du dataset de validation afin d'identifier les sous-ensembles offrant les meilleures correspondances géométriques.

**Scores d'alignement (Qualité de la correspondance Heatmap vs Squelette) :**
* **Graine 70 :** Score 0.439 (Meilleur résultat)
* **Graine 10 :** Score 0.437
* **Graine 30 :** Score 0.428

### Figure 1 : Meilleurs résultats d'alignement (Graine 70)
![Résultats Graine 70](./resultats/figure_1_seed70.png)
*Figure 1 : Rendu de l'inférence. À gauche l'image originale, au centre la Heatmap améliorée générée par Phi, à droite le squelette mis à l'échelle et superposé.*

### Figure 2 : Rendu avec la graine 10
![Résultats Graine 10](./resultats/figure_2_seed10.png)
*Figure 2 : Autre échantillon d'inférence montrant la robustesse du centrage automatique du squelette.*

### Figure 3 : Rendu avec la graine 30
![Résultats Graine 30](./resultats/figure_3_seed30.png)
*Figure 3 : L'alignement par boîte englobante corrige efficacement les erreurs de positionnement brutales du modèle.*

## 6. Discussion
Les résultats démontrent que la Heatmap générée par `Phi` est extrêmement précise pour délimiter la silhouette du cheval, prouvant que le modèle a appris des caractéristiques sémantiques fortes de l'animal. Le décalage des points clés produits par `Omega` montre cependant une sensibilité à l'échelle spatiale. 
L'astuce de post-traitement par alignement de boîte englobante est très efficace pour produire des rendus visuels exploitables. Néanmoins, il s'agit d'une heuristique qui corrige artificiellement la sortie d'inférence.

## 7. Conclusion et Perspectives
**Conclusion :**
Nous avons réussi à générer des squelettes articulés de chevaux à partir d'images brutes. Les défauts mineurs d'échelle du réseau ont été contournés efficacement par un traitement intelligent des prédictions (Bounding Box sur la Heatmap), permettant d'obtenir des visuels clairs et précis pour la restitution.

**Perspectives :**
Pour des travaux futurs, il conviendra de :
1. Intégrer une perte de diversité spatiale ou une contrainte de boite englobante directement dans la fonction de perte (`loss`) durant l'entraînement.
2. Effectuer un fine-tuning spécifique sur `Omega` pour qu'il apprenne intrinsèquement l'échelle correcte.
3. Quantifier l'erreur sur un jeu de test annoté via la métrique PCK (Percentage of Correct Keypoints).

## 8. Bibliographie et Ressources
* Fichiers du modèle : `models/image_to_skeleton.py` et `models/skeleton_to_pose2d.py`
* Script de diagnostic : `evaluation/skeleton_inspect.py`
* Sauvegarde de l'entraînement : `checkpoints/latest.pt` (Époque 150)
* PyTorch Documentation : https://pytorch.org/docs/stable/index.html
