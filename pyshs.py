"""
Module PySHS pour faciliter le traitement statistique en SHS avec Python
Dernière modification : 18/04/2021
Auteur : Émilien Schultz

Pour le moment le module PySHS comprend
- une fonction pour le tri à plat (pondérés)
- une fonction pour les tableaux croisés (pondérés)
- une fonction pour des tableaux croisés multiples (pondérés) afin de voir le lien variable dépendante/indépendantes
- une fonction de mise en forme des résultats de la régression logistique
"""


import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

__version__ = "0.1.1"

def tri_a_plat(df,variable,weight = False):
    """
    Tri à plat pour une variable qualitative
    
    Paramètres
    ----------
    df : DataFrame
    variable : string, nom de la colonne
    weight : optionnel, colonne de la pondération

    Retours
    -------
    DataFrame mis en forme du tri à plat avec total et pourcentages
    
    Remarques
    -------
    Pas de gestion des valeurs manquantes actuellement

    """
    # Tester le format de l'entrée
    if not isinstance(df, pd.DataFrame):
        print("Attention, ce n'est pas un tableau Pandas")
        return None
    if not variable in df.columns:
        print("Attention, la variable n'est pas dans le tableau")
        return None
    
    # Cas de données non pondérées
    if not weight:
        effectif = df[variable].value_counts()
        pourcentage = round(100*df[variable].value_counts(normalize=True),1)
        tableau = pd.DataFrame([effectif,pourcentage]).T
        tableau.columns = ["Effectif","Pourcentage (%)"]
    
    # Cas des données pondérées
    else:   
        effectif = round(df.groupby(variable)[weight].sum(),1)
        total = effectif.sum()
        pourcentage = round(100*effectif/total,1)
        tableau = pd.DataFrame([effectif,pourcentage]).T
        tableau.columns = ["Effectif redressé","Pourcentage (%)"]

    # Retourner le tableau ordonné
    return tableau.sort_index()



# à ajouter éventuellement de la mise en forme

def tableau_croise(df,c1,c2,weight=False,p=False,debug=False):
    """
    Tableau croisé pour deux variables qualitatives, avec
    présentation des pourcentages par ligne
    
    Paramètres
    ----------
    df : DataFrame
    c1,c2 : string, noms des deux colonnes à croiser
    weight : optionnel, colonne de la pondération
    p : optionnel, retour de la probabilité critique calculée avec un chi2
    debug : optionnel, retour des tableaux bruts non mis en forme

    Retours
    -------
    DataFrame mis en forme du tableau croisé avec total et pourcentages
    
    Si p, ajout de la probabilité critique
    Si debug, retour de tous les tableaux intermédiaires
    
    Remarques
    -------
    Pas de gestion des valeurs manquantes actuellement

    """
    
    # Si les données ne sont pas pondérées, pondération à 1
    if not weight:
        df = df.copy() # Pour ajouter une colonne
        df["weight"]=1
        weight = "weight"
        
    # Tableau effectif avec distribution marginales
    t_absolu = round(pd.crosstab(df[c1],df[c2],df[weight],aggfunc = sum,margins=True),1).fillna(0)
    # Tableau effectif sans distribution marginales
    t_absolu_sm = round(pd.crosstab(df[c1],df[c2],df[weight],aggfunc = sum),1).fillna(0)
    # Tableau pourcentages par ligne
    t_pourcentage = t_absolu_sm.apply(lambda x: 100*x/sum(x),axis=1)
    
    # Mise en forme du tableau avec les pourcentages
    t = t_absolu.copy()
    for i in range(0,t_pourcentage.shape[0]):
        for j in range(0,t_pourcentage.shape[1]):
            t.iloc[i,j] = str(t_absolu.iloc[i,j]) \
            +" ("+str(round(t_pourcentage.iloc[i,j],1))+"%)"
    
    # Retour des tableaux non mis en forme
    if debug:
        return t,t_absolu,t_absolu_sm,t_pourcentage
    
    # Retour du tableau avec la p-value
    if p:
        return t,chi2_contingency(t_absolu_sm)[1]
    
    # Retour du tableau mis en forme
    return t


def tableau_croise_multiple(df,dep,indeps,weight=False, chi2 = True):
    """
    Tableau croisé multiple une variable dépendantes/plusieurs indépendantes
    
    Paramètres
    ----------
    df : DataFrame
    dep : string, nom de la variable dépendante en colonne
    indeps : dictionnaire des variables indépendantes et leur label pour le tableau
    weight : optionnel, colonne de la pondération

    Retours
    -------
    DataFrame mis en forme du tableau croisé avec pourcentages par ligne et tri croisé sur total
    p-value indicative par un chi2
    
    """
    
    # Total de pondération pour le calcul du tri à plat par variable
    if not weight:
        total = len(df)
    else:
        total = df[weight].sum()
    
    t_all = {}
    
    # Boucle sur les variables indépendantes
    for i in indeps:
        # Tableau croisé pondéré
        t,p = tableau_croise(df,i,dep,weight,p=True)
        if chi2:
            t_all[indeps[i]+" (p = %.03f)" % p] = t.drop("All")
        else:
             t_all[indeps[i]] = t.drop("All")
    # Création d'un dataframe
    t_all = pd.concat(t_all)
    t_all["Total"] = t_all["All"].apply(lambda x : "%0.1f (%0.1f %%)" % (x, round(100*x/total,1)))
    t_all.columns.name = ""
    t_all.index.names = ["Variable","Modalités"]
    
    return t_all.drop("All",axis=1)


def significativite(x,digits=4):
    """
    Nombre d'étoiles associées à une p-value

    Paramètres
    ----------
    x : float, valeur de la p-value

    Retours
    -------
    string : p-value arrondie et étoiles
    """
    if pd.isnull(x):
        return None

    # Arrondir
    x = round(x,digits)

    # Retourner la valeur avec le nombre d'étoiles associées
    if x < 0.001:
        return str(x)+ "***"
    if x < 0.01:
        return str(x)+ "**"
    if x < 0.05:
        return str(x)+ "*"

    return str(x)

def presentation_logistique(regression,sig=False):
    """
    Mise en forme des résultats de régression logistique

    Paramètres
    ----------
    regression: modèle de régression de statsmodel
    sig: optionnel, booléen

    Retours
    -------
    DataFrame : tableau de la régression logistique
    """

    # Passage des coefficients aux Odds Ratio
    df = np.exp(regression.conf_int())
    df['odd ratio'] = round(np.exp(regression.params), 2)
    df["p-value"] = round(regression.pvalues, 3)       
    df["IC"] = df.apply(lambda x : "%.2f [%.2f-%.2f]" \
                         % (x["odd ratio"],x[0],x[1]),axis=1)

   # Ajout de la significativité
    if sig:
        df["p-value"] = df["p-value"].apply(significativite)

    df = df.drop([0,1], axis=1)
    return df




# Fonctions temporaires


def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))
    
