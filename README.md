# Bibliothèque PySHS

La bibliothèque PySHS a pour but de réunir des outils utiles à un public de praticiens des sciences humaines et sociales francophones pour traiter des données. Elle a pour but de s'enrichir progressivement pour permettre à Python de devenir une alternative (réaliste) à R.

## Contenu

### Traiter des données d'enquête par questionnaire

- Tri à plat et tableau croisé avec pondération

## Installation

:warning: PySHS est uniquement compatible avec **Python 3**.

**Via PyPI**

```sh
$ pip install pyshs
```

## Exemples

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


### Tableau croisé d'une variable qualitative pondérée

```python
>> import pyshs
>> data = pd.read_excel("enquete.xlsx")
>> pyshs.tableau_croise(data,"age","sexe","weight")
```

| age     | female        | male          |    All |
|:--------|:--------------|:--------------|-------:|
| [0-25[  | 158.9 (61.0%) | 101.5 (39.0%) |  260.4 |
| [25-45[ | 370.7 (50.7%) | 360.4 (49.3%) |  731.1 |
| [45-65[ | 372.7 (49.4%) | 382.4 (50.6%) |  755.1 |
| [65+    | 127.2 (49.6%) | 129.2 (50.4%) |  256.4 |
| All     | 1029.5        | 973.5         | 2003   |