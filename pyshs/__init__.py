"""
PySHS - Faciliter le traitement de données de questionnaires en SHS
Langue : Français
Dernière modification : 2/03/2025
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
    catdes,
    construction_formule,
    description,
    ecart_type_pondere,
    likelihood_ratio,
    moyenne_ponderee,
    plot_acm,
    regression_logistique,
    significativite,
    tableau_croise,
    tableau_croise_controle,
    tableau_croise_multiple,
    tableau_reg_logistique,
    tri_a_plat,
    verification_recodage,
    vers_excel,
)

__version__ = "0.3.9"
__author__ = "Émilien Schultz"
