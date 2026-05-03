# Projet Python - ParlaCAP

## Description
L'objectif de ce projet est d'entraîner un modèle transformers sur des échanges parlementaires en français afin de les associer à des partis politiques.

## Installation

### Dépendances

Bibliothèques utilisées :
- **transformers** : fournit les outils pour la tokénisation, la classification et l'entrainement du modèle
- **datasets** : charge les données
- **evaluate** : évalue quantitativement le modèle
- **accelerate** : optimise le calcul
- **numpy** : fournit des fonctions mathématiques

### Notes d'installation

Pour installer les dépendances lancez la commande suivante :

```bash
pip install transformers datasets evaluate accelerate
```

## API

|Fonction|Entrée(s)|Sortie(s)|Description|
|---|---|---|---|
|compute_metrics|eval_pred(logits,labels)|dict(acc,f1)|Renvoie la précision et la F-mesure|
|preprocess_function|examples(texte brut)|dict(tokens)|Tokénise le texte et associe un tenseur à chaque token|
|group_small_parties|example(row)|example(modifié)|Regroupe les partis d'une fréquence inférience à 2%|

## Tests

Les tests ont été effectués sur les serveurs de l'UGA.

### Premier test

Le premier test a été effectué avec **3 époques**. On remarque qu'à la fin de l'apprentissage, il y a un écart entre la perte d'entrainement et la perte d'évaluation ce qui peut être le signe d'un surapprentissage.

La **précision** est de **54,06%** ce qui voudrait dire que la classe est correctement attribuée environ une fois sur deux. Cependant le **F1** de **29%**, qui fait la moyenne des performances pour chaque classe, montre qu'en réalité les classes sont moins bien reconnues qu'une fois sur deux.

Etant donné qu'une classe est majoritaire et représente environ 1/3 des discours, le modèle n'est probablement performant que pour cette classe.

### Second test

Ce test s'est également réalisé sur **3 époques**.

Paramètres modifiés par rapport à l'essai précédent : cette fois-ci on force le modèle à se baser sur le score de F1 pour déterminer la meilleure version du modèle. 

La **précision** obtenue est de **53,67%**, ce qui est équivalent au résultat du premier test. En revanche le **F1** monte à **43,52%** ce qui montre que le modèle est bien plus performant sur les plus petites classes.

L'écart entre la perte d'apprentissage et la perte d'évaluation est également très faible, ce qui montre que cette fois le modèle n'a pas fait trop de surapprentissage.

### Troisième test

-> **3 époques**

Aucun paramètre n'a été modifié par rapport au deuxième test.

**précision = 56,41%**

**F1 = 47%**

=> Ces résultats montrent que le modèle arrive mieux à reconnaître les plus petites classes.

Un rapport détaillé (voir rapport_v3.txt) montre cependant que les résultats restent très inégaux en fonction des classes. Sans surprise, les classes les plus représentées sont très bien reconnues, mais MODEM et SOC obtiennent des scores très mauvais notamment au niveau du rappel, ce qui montre que le modèle ne reconnait que difficilement les discours appartenant à ces partis.

En dehors du déséquilibre de notre corpus, cette différence pourrait s'expliquer par des proximités entre ces partis qui les rendrais difficile à différencier. Il est aussi possible qu'un certain nombre de discours soient trop "neutres" et ne contiennent donc pas d'indices explicites qui permettraient au modèle de les labéliser correctement.


### Quatrième test

-> **3 époques**

Paramètres modifiés : pour ce test on calcule des poids pour compenser la faible fréquence de certaines classes, afin de forcer le modèle à porter plus d'attention aux petits partis. 

De plus, on passe au modèle camembert-base qui est plus performant que distilcamembert. 

**précision = 51%**

**F1 = 47%**

Le changement de paramètres a fait légèrement baisser la précision puisque le modèle prend plus de risques sur les petits classes. En revanche, la perte d'évaluation (1,413) est plus haute que la perte d'entraînement (1,266) ce qui peut montrer qu'avec ces modifications de poids, le modèle a du mal à généraliser. 

### Cinquième test

-> **4 époques**

Paramètres modifiés : sur cet essai, la pondération personnalisée des classes a été enlevée.

**précision = 56%**

**F1 = 48%**

C'est avec cet essai que nous avons obtenu les meilleurs scores. L'entraînement avec 1 époque en plus ainsi que l'utilisation d'un modèle un peu plus performant a eu un impact positif sur nos résultats. De plus, les valeurs de perte d'apprentissage et d'évaluation montrent que le modèle parvient à mieux généraliser.

### Conclusion
Tous ces tests nous ont permis d'affiner les paramètres du modèle cependant, étant donné le fort déséquilibre des classes sur notre corpus, il faudrait faire un travail de nettoyage des données encore plus important pour réellement arriver à des résultats satisfaisants. 

## Bugs identifiés

Lors du troisième test, le script a planté en raison d'un problème d'encodage d'un caractère. Le problème a cependant été résolu.

## Structures des données

### Dataset avant et après tokénisation

**Données brutes avant tokénisation**

Notre dataset (https://data.crossda.hr/file.xhtml?persistentId=doi:10.23669/1ZTELP/TRB27P&version=1.0) contient des discours parlementaires enregistrés en France entre 2015 et 2022.

Le dataset est au format .tsv contient des colonnes de métadonnées (chaque discours a un identifiant, une date etc). Pour l'entraînement du modèle nous n'avons retenu que la colonne 'speaker_party' contenant les différents partis politiques (labels) ainsi que la colonne contenant le texte des discours (input fourni au modèle).
Après analyse du jeu de données par un script Python, on obtient les statistiques suivantes.

**Statistiques avant nettoyage**


|Métrique|Données brutes|
|:---:|:---:|
|Nb total de discours|714,439|
|Longueur médiane discours|13 mots|


Pour la répartition des classes :

|Parti|Proportion [%]|
|:---:|:---:|
|LAREM|29.11|
|LR|23.30|
|-|15.15|
|FI|5.96|
|SOC|5.55|
|MODEM|5.19|
|GDR|4.52|
|DEM|2.90|
|UDI-AGIR|2.85|
|LT|1.98|
|NG|1.28|
|UDI-I|0.69|
|AGIR-E|0.44|
|UDI-A-I|0.31|
|LC|0.31|
|UDI-I|0.30|
|EDS|0.13|
|RE|0.00|

=> 18 classes au total

Avec ces statistiques sur le dataset brut, on peut voir qu'il était nécessaire de faire un nettoyage :
- pour éliminer les prises de parole trop courtes
- pour éliminer les discours sans parti politique associé
- pour pallier au fort déséquilibre des classes

=> Tous les discours d'une longueur inférieure à 15 mots ont été filtrés, et les partis politiques de proportion inférieure à 2% ont été regroupés dans une catégorie AUTRE.

**Statistiques après nettoyage**

Ces statistiques ont été calculées avec le script clean_data.py. 
|Métrique|Stats|
|:---:|:---:|
|Nb total de discours|262,257|
|% de données conservées|36.7%|
|Longueur médiane discours|72 mots|



|Parti|Proportion [%]|
|:---:|:---:|
|LAREM|32.49|
|LR|23.37|
|AUTRE|16.08|
|FI|9.63|
|SOC|6.85|
|GDR|6.13|
|MODEM|5.45|

=> 7 classes après nettoyage

Il faut aussi tenir compte du fait que le modèle ne regarde entièrement que les textes d'une longueur inférieure à 512 tokens.
Après calcul, seules **5.01%** de nos données nettoyées étaient tronquées par cette limite, ce qui semble acceptable.

Le dataset nettoyé a ensuite été divisé de manière aléatoire avec 80% des données réservées pour l'entraînement et les 20% restant pour le test.

**Après tokénisation**

L'algorithme de tokénisation utilisé est celui qui est associé au modèle DistilCamemBERT, une version plus légère du modèle CamemBERT pour le français. Son vocabulaire contient 32 005 tokens.

Pour les derniers essais, nous avons aussi testé le modèle CamemBERT pour essayer d'obtenir de meilleurs résultats.

En sortie du tokéniseur, on obtient une liste de tokens convertis en ID numériques (int). Cette liste est ensuite passée dans une fonction de préparation des données, qui applique un padding pour que toutes les séquences aient la même longueur. Un attention_mask est donc ajouté afin d'indiquer au modèle les éléments de padding à ignorer.
Enfin les labels sont associés aux séquences pour permettre au modèle d'apprendre à prédire.


### Sorties du modèle

Les résultats bruts du modèle sortent sous forme de logits (float), organisés en tenseur. Ces nombres donnent pour chaque classe la force de prédiction : plus la valeur est élevée, plus il est probable que l'input appartienne à cette classe.

Le tenseur de sortie a pour taille [8,7] : 8 batchs (8 lignes) et 7 classes (7 colonnes). Une ligne contient les scores d'un batch pour chaque classe.

La fonction d'activation va ensuite traduire les logits pour les rendre interprétables, en les transformant en probabilités.

### Fonctions

Les fonctions utilisées servent à transformer les données pour les rendre utilisables par le modèle :
- group_small_parties réduit le nombre de classes afin de rendre la distribution plus équilibrée
- preprocess_function convertit l'input, sous forme de chaînes de caractères, en matrice d'entiers, format compréhensible pour la machine
- compute_metrics traite la sortie du modèle pour donner des probabilités interprétables, en transformant les 7 scores (pour chaque classe) en un seul entier correspondant à la classe prédite. A partir de cette donnée, les métriques de performance sont calculées.

### Evaluation de la complexité

Le modèle compare chaque mot d'une séquence avec tous les autres mots de la même séquence. La même opération est répétée pour toutes les autres séquences. En rappelant que le modèle a analysé environ 260,000 discours d'une longueur médiane de 72 mots on peut donc estimer la complexité à O(n²*N) avec n la longueur de la séquence .



