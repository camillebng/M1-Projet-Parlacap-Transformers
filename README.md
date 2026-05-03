# Projet Python - ParlaCAP

## Description

L'objectif de ce projet est d'entraîner un modèle Transformers sur des échanges parlementaires en français afin de les associer à des partis politiques.

## Installation

### Dépendances

Bibliothèques utilisées :

- **transformers** : fournit les outils pour la tokénisation, la classification et l'entraînement du modèle.
- **datasets** : charge les données.
- **evaluate** : évalue quantitativement le modèle.
- **accelerate** : optimise le calcul.
- **numpy** : fournit des fonctions mathématiques.
- **scikit-learn** : permet de produire un rapport détaillé par classe.
- **torch** : permet l'entraînement du modèle avec PyTorch.
- **sentencepiece** et **protobuf** : nécessaires au chargement de certains modèles/tokeniseurs.

### Notes d'installation

Pour installer les dépendances, lancez la commande suivante :

```bash
pip install transformers datasets evaluate accelerate numpy scikit-learn torch sentencepiece protobuf
```

## API

| Fonction | Entrée(s) | Sortie(s) | Description |
|---|---|---|---|
| `compute_metrics` | `eval_pred(logits, labels)` | `dict(accuracy, f1_macro)` | Renvoie la précision et la F-mesure macro |
| `preprocess_function` | `examples` contenant le texte brut | `dict(tokens)` | Tokénise le texte et convertit les séquences en identifiants numériques |
| `group_small_parties` | `example(row)` | `example(modifié)` | Regroupe les partis dont la fréquence est inférieure au seuil choisi, par exemple 5 % |

## Tests

Les tests ont été effectués sur les serveurs de l'UGA.

### Premier test

Le premier test a été effectué avec le modèle **cmarkea/distilcamembert-base**, une version allégée de CamemBERT. Le modèle a été entraîné sur **3 époques**. Dans cette première version, le seuil de regroupement des petits partis était fixé à **2 %**.

Résultats obtenus :

- **précision = 54,06 %**
- **F1 macro = 29,16 %**
- **perte d'évaluation = 1,426**

La précision indique que le modèle attribue correctement la classe dans un peu plus d'un cas sur deux. Cependant, le F1 macro est faible. Cela montre que les performances ne sont pas équilibrées entre les classes : le modèle semble surtout mieux reconnaître les classes majoritaires, tandis que les classes minoritaires sont beaucoup moins bien prédites.

Ce résultat ne semble pas principalement dû à un surapprentissage, car l'écart entre la perte d'entraînement et la perte d'évaluation reste limité. Le problème principal est plutôt le déséquilibre des classes.

### Second test

Ce test s'est également réalisé sur **3 époques** avec **cmarkea/distilcamembert-base**.

Paramètres modifiés par rapport à l'essai précédent :

- le seuil de regroupement des petits partis est passé de **2 % à 5 %** ;
- les petits partis ont été regroupés dans une catégorie **AUTRE** ;
- le découpage train/test a été fait de manière stratifiée, afin de conserver la proportion des classes ;
- le modèle sélectionne la meilleure version selon le **F1 macro** ;
- la longueur maximale des séquences a été fixée à **256 tokens**.

Résultats obtenus :

- **précision = 53,67 %**
- **F1 macro = 43,52 %**
- **perte d'évaluation = 1,262**

La précision reste proche de celle du premier test. En revanche, le F1 macro augmente nettement. Cela montre que le modèle est devenu plus performant sur les classes moins représentées, même si la précision globale n'a pas beaucoup changé.

### Troisième test

Le troisième test a été effectué avec **camembert-base**, un modèle plus lourd et plus performant que DistilCamemBERT. La longueur maximale des séquences a aussi été augmentée à **512 tokens**, afin de donner plus de contexte au modèle.

Le modèle a été entraîné sur **3 époques**.

Résultats obtenus :

- **précision = 56,41 %**
- **F1 macro = 47,25 %**
- **perte d'évaluation = 1,236**

Cette version améliore à la fois la précision et le F1 macro. Le modèle semble donc mieux exploiter les indices linguistiques et thématiques présents dans les discours.

Un rapport détaillé par classe a été généré dans `rapport_v3.txt`. Il montre cependant que les résultats restent très inégaux selon les classes. Les classes les mieux reconnues sont notamment **LAREM**, **FI** et **LR**. En revanche, **MODEM** et **SOC** obtiennent des scores plus faibles, notamment au niveau du rappel, ce qui montre que le modèle reconnaît difficilement les discours appartenant à ces partis.

En dehors du déséquilibre du corpus, cette différence pourrait s'expliquer par des proximités thématiques entre certains partis, qui les rendent difficiles à différencier. Il est aussi possible qu'un certain nombre de discours soient trop neutres et ne contiennent donc pas d'indices explicites permettant au modèle de les labéliser correctement.

### Quatrième test

Le quatrième test reprend la configuration du troisième test, mais ajoute une **pondération personnalisée des classes**.

L'objectif était de compenser la faible fréquence de certaines classes, afin de forcer le modèle à porter plus d'attention aux petits partis, comme MODEM, SOC ou GDR.

Le modèle a été entraîné sur **3 époques**.

Résultats obtenus :

- **précision = 51,14 %**
- **F1 macro = 47,35 %**
- **perte d'évaluation = 1,413**

La pondération des classes n'a pas réellement amélioré les performances globales. Le F1 macro reste presque stable, mais la précision diminue. Cela suggère que le modèle prend davantage de risques sur les petites classes, mais au détriment des prédictions globales.

Cette expérimentation n'a donc pas été retenue comme meilleure configuration.

### Cinquième test

Le cinquième test reprend la meilleure configuration précédente, c'est-à-dire **camembert-base** avec une longueur maximale de **512 tokens**, mais le nombre d'époques a été augmenté de **3 à 4**.

La pondération personnalisée des classes n'a pas été conservée.

Résultats obtenus :

- **précision = 56,69 %**
- **F1 macro = 48,29 %**
- **perte d'évaluation = 1,333**

C'est avec cet essai que nous avons obtenu les meilleurs scores globaux. L'amélioration reste légère, mais le F1 macro augmente par rapport au troisième test. Nous retenons donc ce cinquième test comme le meilleur compromis entre précision et F1 macro.

### Tableau récapitulatif des tests

| Test | Dossier | Modèle | Époques | Précision | F1 macro | Perte d'évaluation |
|---|---|---|---:|---:|---:|---:|
| Test 1 | `output` | DistilCamemBERT | 3 | 0.5406 | 0.2916 | 1.4264 |
| Test 2 | `output_v2` | DistilCamemBERT | 3 | 0.5368 | 0.4352 | 1.2625 |
| Test 3 | `output_v3` | CamemBERT-base | 3 | 0.5641 | 0.4725 | 1.2364 |
| Test 4 | `output_v4` | CamemBERT-base + pondération | 3 | 0.5114 | 0.4735 | 1.4133 |
| Test 5 | `output_v5` | CamemBERT-base | 4 | 0.5669 | 0.4829 | 1.3327 |

### Conclusion

Tous ces tests nous ont permis d'affiner les paramètres du modèle. Les améliorations les plus efficaces ont été le passage de **DistilCamemBERT** à **CamemBERT-base**, l'augmentation de la longueur maximale des séquences à **512 tokens**, le regroupement des petits partis dans une catégorie **AUTRE**, et l'utilisation d'un découpage stratifié.

La meilleure configuration obtenue est celle du cinquième test, avec **camembert-base**, `max_length=512` et **4 époques**. Elle atteint environ **56,69 % de précision** et **48,29 % de F1 macro**.

Cependant, les performances restent limitées. Cela s'explique principalement par le fort déséquilibre des classes, l'hétérogénéité de la catégorie **AUTRE**, et la proximité thématique entre certains partis politiques. Pour améliorer davantage le modèle, il faudrait probablement travailler plus finement sur le nettoyage des données, la constitution des classes et éventuellement tester d'autres stratégies d'échantillonnage.

## Bugs identifiés

Lors de la génération du rapport détaillé, un script a d'abord planté à cause d'un problème d'encodage de caractères non ASCII. Le problème a été résolu en exécutant le script avec `python3` dans l'environnement virtuel du projet.

## Structure des données

### Dataset avant et après tokénisation

**Données brutes avant tokénisation**

Notre dataset contient des discours parlementaires enregistrés en France entre 2015 et 2022.

Lien vers le dataset :  
https://data.crossda.hr/file.xhtml?persistentId=doi:10.23669/1ZTELP/TRB27P&version=1.0

Le dataset est au format `.tsv` et contient des colonnes de métadonnées : identifiant du discours, date, langue, nom du locuteur, parti politique, texte, etc. Pour l'entraînement du modèle, nous avons retenu principalement la colonne `speaker_party`, qui correspond au parti politique du locuteur, et la colonne `text`, qui contient le texte du discours.

Après analyse du jeu de données par un script Python, on obtient les statistiques suivantes.

**Statistiques avant nettoyage**

| Métrique | Données brutes |
|:---:|:---:|
| Nb total de discours | 714,439 |
| Longueur médiane discours | 13 mots |

Pour la répartition des classes :

| Parti | Proportion [%] |
|:---:|:---:|
| LAREM | 29.11 |
| LR | 23.30 |
| - | 15.15 |
| FI | 5.96 |
| SOC | 5.55 |
| MODEM | 5.19 |
| GDR | 4.52 |
| DEM | 2.90 |
| UDI-AGIR | 2.85 |
| LT | 1.98 |
| NG | 1.28 |
| UDI-I | 0.69 |
| AGIR-E | 0.44 |
| UDI-A-I | 0.31 |
| LC | 0.31 |
| UDI-I | 0.30 |
| EDS | 0.13 |
| RE | 0.00 |

=> 18 classes au total

Avec ces statistiques sur le dataset brut, on peut voir qu'il était nécessaire de faire un nettoyage :

- pour éliminer les prises de parole trop courtes ;
- pour éliminer les discours sans parti politique associé ;
- pour réduire le fort déséquilibre des classes.

Tous les discours d'une longueur inférieure à 15 mots ont été filtrés. Les discours sans parti politique associé ont aussi été supprimés. Dans les dernières expériences, les partis politiques dont la proportion était inférieure à **5 %** ont été regroupés dans une catégorie **AUTRE**.

**Statistiques après nettoyage**

Ces statistiques ont été calculées avec le script `clean_data.py`.

| Métrique | Stats |
|:---:|:---:|
| Nb total de discours | 262,257 |
| % de données conservées | 36.7 % |
| Longueur médiane discours | 72 mots |

| Parti | Proportion [%] |
|:---:|:---:|
| LAREM | 32.49 |
| LR | 23.37 |
| AUTRE | 16.08 |
| FI | 9.63 |
| SOC | 6.85 |
| GDR | 6.13 |
| MODEM | 5.45 |

=> 7 classes après nettoyage

Il faut aussi tenir compte du fait que le modèle ne regarde entièrement que les textes d'une longueur inférieure à **512 tokens**. Après calcul, seules **5.01 %** de nos données nettoyées étaient tronquées par cette limite, ce qui semble acceptable.

Le dataset nettoyé a ensuite été divisé avec **80 % des données pour l'entraînement** et **20 % pour le test**. Le découpage a été effectué de manière stratifiée, afin de conserver la proportion des classes dans les deux ensembles.

**Après tokénisation**

L'algorithme de tokénisation utilisé dépend du modèle choisi. Les premiers essais utilisent DistilCamemBERT, une version plus légère du modèle CamemBERT pour le français. Les derniers essais utilisent CamemBERT-base, afin d'obtenir de meilleurs résultats.

En sortie du tokéniseur, on obtient une liste de tokens convertis en identifiants numériques. Cette liste est ensuite passée dans une fonction de préparation des données, qui applique un padding pour que toutes les séquences d'un batch aient une longueur compatible. Un `attention_mask` est aussi ajouté afin d'indiquer au modèle les éléments de padding à ignorer.

Enfin, les labels sont associés aux séquences pour permettre au modèle d'apprendre à prédire le parti politique du locuteur.

### Sorties du modèle

Les résultats bruts du modèle sortent sous forme de logits, organisés en tenseur. Ces nombres donnent pour chaque classe la force de prédiction : plus la valeur est élevée, plus il est probable que l'input appartienne à cette classe.

Dans notre cas, le tenseur de sortie a pour taille `[8, 7]` lorsque le batch size est de 8 : 8 exemples dans le batch et 7 classes possibles. Une ligne contient donc les scores d'un exemple pour chaque classe.

Pour obtenir la prédiction finale, on sélectionne la classe qui possède le score le plus élevé. Les métriques sont ensuite calculées à partir de ces prédictions et des labels attendus.

### Fonctions

Les fonctions utilisées servent à transformer les données pour les rendre utilisables par le modèle :

- `group_small_parties` réduit le nombre de classes afin de rendre la distribution plus équilibrée ;
- `preprocess_function` convertit l'input, sous forme de chaînes de caractères, en identifiants numériques compréhensibles par le modèle ;
- `compute_metrics` traite la sortie du modèle pour obtenir les métriques de performance, notamment la précision et le F1 macro.

### Évaluation de la complexité

Le modèle repose sur une architecture de type Transformer. Dans ce type de modèle, le mécanisme d'attention compare les tokens d'une séquence entre eux. Pour une séquence de longueur `n`, cette opération a une complexité approximative de `O(n²)`.

Comme cette opération est répétée pour un grand nombre de discours, on peut estimer la complexité globale à `O(N × n²)`, avec `N` le nombre de discours et `n` la longueur maximale des séquences.
