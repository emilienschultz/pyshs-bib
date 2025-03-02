# PySHS : traitement de données en sciences sociales avec Python

La bibliothèque PySHS a pour but de faciliter l'analyse de données avec Python de questionnaire en sciences humaines et sociales pour les francophones.

Elle vise à combler (partiellement) le manque de fonctions de traitement de données en Python pour les enquêtes par questionnaires, notamment en comparaison avec R.

## Contenu

### Traiter des données issues d'enquêtes par questionnaires (avec pondération)

- Description d'un tableau de données
- Tri à plat et tableau(x) croisé(s) avec pondération
- Régression logistique sous forme d'un tableau simple
- Sauvegarde en format tableur de tableaux
- Importation de la fonction `catdes` de FactoMineR (association variable catégorielle/autres variables)
- Visualisation statique d'une ACM de Prince

N'hésitez pas à compléter :) 

## Installation

### Avec pip

```sh
$ pip install pyshs --upgrade
```

### Version de développement

```sh
$ git clone https://github.com/emilienschultz/pyshs-bib.git
$ cd pyshs
$ pip install -e .
```

## Utilisation

### Tri à plat d'une variable qualitative pondérée

```python
>> import pyshs
>> data = pd.read_excel("enquete.xlsx")
>> pyshs.tri_a_plat(data,"age","weight")
```

| age     |   Effectif redressé |   Pourcentage (%) |
|:--------|--------------------:|------------------:|
| [0-25[  |               260.4 |              13.0 |
| [25-45[ |               731.1 |              36.5 |
| [45-65[ |               755.1 |              37.7 |
| [65+    |               256.4 |              12.8 |
| Total   |                2003 |               100 |


### Autres utilisations

Voir le Notebook **Exemple PySHS.ipynb** pour voir les fonctions disponibles.
