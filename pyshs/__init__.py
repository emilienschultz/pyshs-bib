"""
PySHS - Faciliter le traitement de données de questionnaires en SHS
Langue : Français
Dernière modification : 27/08/2023
Auteur : Émilien Schultz
Contributeurs.rices :
- Matthias Bussonnier
- Léo Mignot
- Fatima Gauna
- Thibault Clérice 
- Jean-Baptiste Bertrand 

Pour le moment le module PySHS comprend :

- une fonction de description du tableau
- une fonction de comparaison de deux colonnes pour voir le recodage
- une fonction pour le tri à plat (pondéré)
- une fonction pour les tableaux croisés (pondérés)
- une fonction pour des tableaux croisés multiples (pondérés) afin de voir le lien variable dépendante/indépendantes
- une fonction pour un tableau croisé à trois variables pour en contrôler une lors de l'analyse
- une fonction de mise en forme des résultats de la régression logistique de Statsmodel pour avoir un tableau avec les références
- une fonction pour produire la régression logistique binomiale et la mettre en forme
- une fonction moyenne & écart-type pondéré
- une fonction d'écriture de tableaux excels
- le portage de la fonction R de FactoMineR *catdes* (uniquement quali)

En beta :
- une fonction de test du ratio de vraissemblance de deux régressions
- une fonction de visualisation des ACM de prince en statique

"""

from ._core import (
    description,
    tri_a_plat,
    verification_recodage,
    tableau_croise,
    tableau_croise_controle,
    tableau_croise_multiple,
    significativite,
    tableau_reg_logistique,
    construction_formule,
    regression_logistique,
    likelihood_ratio,
    vers_excel,
    moyenne_ponderee,
    ecart_type_pondere,
    catdes,
    plot_acm)

__version__ = "0.3.7"
