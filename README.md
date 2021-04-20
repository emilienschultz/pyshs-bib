# Bibliothèque PySHS

La bibliothèque PySHS a pour but de réunir des outils utiles à un public de praticiens des sciences humaines et sociales francophones pour traiter des données. Elle a pour but de s'enrichir progressivement pour permettre à Python de devenir une alternative (réaliste) à R avec des fonctions facilement utilisable sur les opérations habituelles.

## Contenu

### Traiter des données d'enquête par questionnaire

- Tri à plat et tableau croisé avec pondération
- Tableau croisant une variable dépendante avec une série de variables indépendantes

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

### Tableau croisant une variable dépendante et plusieurs variables indépendantes

```python
>> import pyshs
>> data = pd.read_excel("enquete.xlsx")
>> var_indep = {"sexe":"Genre","age":"Age","zones":"Lieu d'habitation"}
>> pyshs.tableau_croise_multiple(data,"confiance_scientifiques",var_indep,"weight")
```
|                                      | 1 - Oui       | 2 - Non       | Total           |
|:-------------------------------------|:--------------|:--------------|:----------------|
| Femmes                               | 899.2 (87.3%) | 130.3 (12.7%) | 1029.5 (51.4 %) |
| Hommes                               | 867.6 (89.1%) | 105.9 (10.9%) | 973.5 (48.6 %)  |
| Age [0-25[                           | 231.5 (88.9%) | 28.9 (11.1%)  | 260.4 (13.0 %)  |
| Age [25-45[                          | 640.9 (87.7%) | 90.2 (12.3%)  | 731.1 (36.5 %)  |
| Age [45-65[                          | 662.9 (87.8%) | 92.2 (12.2%)  | 755.1 (37.7 %)  |
| Age [65+                             | 231.5 (90.3%) | 24.9 (9.7%)   | 256.4 (12.8 %)  |
| Lieu d'habitation: Paris             | 261.0 (89.7%) | 29.9 (10.3%)  | 290.8 (14.5 %)  |
| Lieu d'habitation: Rural             | 391.6 (87.3%) | 57.0 (12.7%)  | 448.7 (22.4 %)  |
| Lieu d'habitation: Urban <100k       | 277.0 (86.6%) | 43.0 (13.4%)  | 320.1 (16.0 %)  |
| Lieu d'habitation: Urban <20k        | 182.1 (84.5%) | 33.5 (15.5%)  | 215.6 (10.8 %)  |
| Lieu d'habitation: Urban >100k       | 655.1 (90.0%) | 72.7 (10.0%)  | 727.8 (36.3 %)  |

