# modification du 02/03/2025

import math
import warnings
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import]
import plotly.graph_objects as go  # type: ignore[import]
import seaborn as sns  # type: ignore[import]
import statsmodels.api as sm  # type: ignore[import]
import statsmodels.formula.api as smf  # type: ignore[import]
from adjustText import adjust_text  # type: ignore[import]
from pandas import DataFrame
from pandas.api.types import is_numeric_dtype  # type: ignore[import]
from samplics.categorical import CrossTabulation  # type: ignore[import]
from scipy.stats import chi2_contingency, hypergeom, norm  # type: ignore[import]
from scipy.stats.distributions import chi2  # type: ignore[import]
from statsmodels.formula.api import ols  # type: ignore[import]

# gestion de la langue
langue = "fr"


def description(df: DataFrame, arrondir: int = 2) -> DataFrame:
    """
    Description d'un tableau de données.

    Parameters
    ----------
    df : DataFrame
    arrondir : int, optionnel
        nombre de décimales (défaut : 2)

    Returns
    -------
    DataFrame
        Description des variables du tableau

    """
    tableau = []

    # gestion de la langue dans le tableau
    if langue == "fr":
        textes = ["Numérique", "Catégorielle"]
    elif langue == "en":
        textes = ["Numeric", "Category"]
    else:
        warnings.warn(
            "Langue non gérée, les types de variables ne seront pas indiqués",
            UserWarning,
        )
        textes = ["Numeric", "Category"]

    for i in df.columns:
        if is_numeric_dtype(df[i]):
            l = [
                i,
                textes[0],
                None,
                None,
                round(df[i].mean(), arrondir),
                round(df[i].std(), arrondir),
                pd.isnull(df[i]).sum(),
            ]
        else:
            l = [
                i,
                textes[1],
                len(df[i].unique()),
                df[i].mode().iloc[0],
                None,
                None,
                pd.isnull(df[i]).sum(),
            ]
        tableau.append(l)
    if langue == "fr":
        columns = [
            "Variable",
            "Type",
            "Modalités",
            "Mode",
            "Moyenne",
            "Écart-type",
            "Valeurs manquantes",
        ]
    elif langue == "en":
        columns = [
            "Variable",
            "Type",
            "Modalities",
            "Mode",
            "Mean",
            "Standard error",
            "Missing values",
        ]

    df = pd.DataFrame(
        tableau,
        columns=columns,
    ).set_index(columns[0])
    return df.fillna(" ")


def tri_a_plat(
    df: DataFrame, variable: str, poids: str | None = None, arrondir: int = 1
) -> DataFrame:
    """
    Tri à plat pour une variable qualitative.
    Pondération possible.

    Parameters
    ----------
    df : DataFrame
    variable : string
        nom de la colonne
    poids : string (optionnal)
        colonne de pondération
    arrondir : int, optionnel
        nombre de décimales (défaut : 1)

    Returns
    -------
    DataFrame
        Tri à plat mis en forme

    Notes
    -----
    Pas de gestion des valeurs manquantes.

    """
    # Tester le format de l'entrée
    if not isinstance(df, pd.DataFrame):
        warnings.warn("Attention, ce n'est pas un tableau Pandas", UserWarning)
        return None

    if variable not in df.columns:
        warnings.warn("Attention, la variable n'est pas dans le tableau", UserWarning)
        return None

    # Cas de données non pondérées
    if not poids:
        effectif = df[variable].value_counts()
        pourcentage = round(100 * df[variable].value_counts(normalize=True), arrondir)
        tableau = pd.DataFrame([effectif, pourcentage]).T

    # Cas des données pondérées
    else:
        effectif = round(df.groupby(variable)[poids].sum(), arrondir)
        total = effectif.sum()
        pourcentage = round(100 * effectif / total, arrondir)
        tableau = pd.DataFrame([effectif, pourcentage]).T

    # Mise en forme du tableau
    if langue == "fr":
        columns = ["Effectif", "Pourcentage (%)"]
    elif langue == "en":
        columns = ["Frequency", "Percentage (%)"]
    tableau.columns = columns

    # Retourner le tableau ordonné en forçant l'index en texte
    tableau.index = [str(i) for i in tableau.index]
    tableau = tableau.sort_index()

    # Ajout de la ligne total
    tableau.loc["Total"] = [effectif.sum(), round(pourcentage.sum(), arrondir)]

    return tableau


def verification_recodage(tableau: DataFrame, c1: str, c2: str) -> None:
    """
    Comparer une variable recodée avec la variable initiale.

    Parameters
    ----------
    corpus : DataFrame
    c1 : str
        nom de la colonne 1
    c2 : str
        nom de la colonne 2

    Returns
    -------
    None

    Notes
    -----
    Pour le moment uniquement de l'affichage.

    """

    # Vérifier que les deux colonnes sont distinctes
    if c1 == c2:
        warnings.warn("Ce sont les mêmes colonnes", UserWarning)
        return None

    # Vérifier que les deux variables sont bien dans le corpus
    for c in [c1, c2]:
        if c not in tableau.columns:
            warnings.warn("La variable %s n'est pas dans le tableau" % c, UserWarning)
            return None

    # Vérification s'il y a des valeurs manquantes dans la colonne d'arrivée
    s = pd.isnull(tableau[c2]).sum()
    if s > 0:
        warnings.warn(
            "Il y a %d valeurs nulles dans la colonne recodée" % s, UserWarning
        )

    # renommer et modifier les labels pour éviter les homonymies
    df = tableau[[c1, c2]].copy()
    df[c1] = df[c1].fillna("None").apply(lambda x: str(x) + "(1)")
    df[c2] = df[c2].fillna("None").apply(lambda x: str(x) + "(2)")

    # tableau croisé des deux variables
    t = pd.crosstab(df[c2], df[c1])

    # création des relations
    t_flat = t.unstack()  # Déplier le tableau croisé
    links = []
    for i, j in zip(t_flat.index, t_flat):
        links.append([i[0], i[1], j])

    # éléments pour définir le diagramme de Sankey avec plotly
    all_nodes = list(t.index) + list(t.columns)
    source_indices = [all_nodes.index(i[0]) for i in links]
    target_indices = [all_nodes.index(i[1]) for i in links]
    values = [i[2] for i in links]
    node_colors = ["orange"] * len(t.index) + ["blue"] * len(t.columns)

    # Création de la figure
    fig = go.Figure(
        data=[
            go.Sankey(
                # Define nodes
                node=dict(label=all_nodes, color=node_colors),
                # Add links
                link=dict(
                    source=source_indices,
                    target=target_indices,
                    value=values,
                    # color = edge_colors,
                ),
            )
        ]
    )

    fig.update_layout(
        title_text="Colonne %s à colonne %s" % (c1, c2),
        font_size=10,
        height=500,
        width=600,
    )
    fig.show()
    return None


def tableau_croise(
    df: DataFrame,
    c1: str,
    c2: str,
    poids: str | None = None,
    p: bool = False,
    verb: bool = False,
    arrondir: int = 1,
) -> DataFrame:
    """
    Tableau croisé pour deux variables qualitatives et % par ligne.

    Parameters
    ----------
    df : DataFrame
        tableau de données
    c1,c2 : string
        nom des colonnes
    poids : string (optionnel),
        pondération
    p : bool (optionnel)
        ajout d'un test de chi2
    verb : bool (optionnel)
        sortie "verbeuse"
        tableaux complet, absolu, pourcentages, p-value
    arrondir : int, optionnel
        nombre de décimales (défaut : 1)

    Returns
    -------
    crosstab : DataFrame
        Tableau croisé mis en forme

    Notes
    -----
    Pas de gestion des valeurs manquantes actuellement

    """

    # Tester le format de l'entrée
    if not isinstance(df, pd.DataFrame):
        warnings.warn("Attention, ce n'est pas un tableau Pandas", UserWarning)
        return None
    if c1 not in df.columns or c2 not in df.columns:
        warnings.warn(
            "Attention, une des variables n'est pas dans le tableau", UserWarning
        )
        return None

    # Si les données ne sont pas pondérées, création d'une pondération unitaire
    if poids:
        pondere = True
    else:
        df = df.copy()  # Pour ne pas modifier l'objet
        df["poids"] = 1
        poids = "poids"
        pondere = False

    # Tableau effectif avec distribution marginales
    t_absolu = round(
        pd.crosstab(df[c1], df[c2], values=df[poids], aggfunc="sum", margins=True),
        arrondir,
    ).fillna(0)
    # Tableau pourcentages par ligne (enlever la colonne totale)
    t_pourcentage = t_absolu.drop("All", axis=1).apply(
        lambda x: round(100 * x / sum(x), arrondir), axis=1
    )

    # Mise en forme du tableau avec les pourcentages
    t = t_absolu.copy().astype(str)
    for i in range(0, t_pourcentage.shape[0]):
        for j in range(0, t_pourcentage.shape[1]):
            t.iloc[i, j] = (
                str(t_absolu.iloc[i, j])
                + " ("
                + str(round(t_pourcentage.iloc[i, j], arrondir))
                + "%)"
            )

    # Ajout des 100% pour la ligne colonne
    t["All"] = t["All"].apply(lambda x: "{} (100%)".format(x))

    # Traduire All par Total
    t = t.rename(index={"All": "Total"}, columns={"All": "Total"})
    t_absolu = t_absolu.rename(index={"All": "Total"}, columns={"All": "Total"})
    t_pourcentage = t_pourcentage.rename(
        index={"All": "Total"}, columns={"All": "Total"}
    )

    # Calcul du test
    if pondere:  # correction Rao-Scott avec samplics
        tab = CrossTabulation()
        tab.tabulate(
            vars=df[[c1, c2]].rename(columns={c1: "var1", c2: "var2"}),
            samp_weight=df[poids],
            remove_nan=True,
        )
        val_p = tab.stats["Pearson-Adj"]["p_value"]
    else:  # sans correction
        # TO DO : utiliser le calcul de samplics ?
        val_p = chi2_contingency(t_absolu.drop("Total").drop("Total", axis=1))[1]

    # Retour des tableaux non mis en forme
    if verb:
        return t, t_absolu, t_pourcentage, val_p

    # Retour du tableau avec la p-value
    if p:
        return t, val_p

    # Retour du tableau mis en forme
    return t


def tableau_croise_controle(
    df, cont, c, r, poids=False, chi2=False, arrondir=1, proba_simplifiee=False
):
    """
    Tableau croisé avec variable de contrôle.

    Parameters
    ----------
    df : DataFrame
        tableau de données
    cont : string
        colonne de contrôle
    c,r : strings
        colonnes à croiser
    poids : string (optionnel),
        poids optionnel
    arrondir : int, optionnel
        nombre de décimales (défaut : 1)
    proba_simplifiee : bool, optionnel
        simplifier les probas (défaut : False)

    Returns
    -------
    crosstab : DataFrame
        Tableau croisé mis en forme

    Notes
    -----
    Pas de gestion des valeurs manquantes actuellement

    """

    # Tester le format de l'entrée
    if not isinstance(df, pd.DataFrame):
        warnings.warn("Attention, ce n'est pas un tableau Pandas", UserWarning)
        return None
    if cont not in df.columns or c not in df.columns or r not in df.columns:
        warnings.warn(
            "Attention, une des variables n'est pas dans le tableau", UserWarning
        )
        return None

    # Si les données ne sont pas pondérées, création d'une pondération unitaire
    if not poids:
        df = df.copy()  # Pour ne pas modifier l'objet
        df["poids"] = 1
        poids = "poids"

    tab = {}
    mod = df[cont].unique()  # modalités de contrôle
    for i in mod:
        d = df[df[cont] == i]  # sous-ensemble
        t, p = tableau_croise(d, c, r, poids, p=True, arrondir=arrondir)

        # Construire le tableau avec ou sans le chi2
        if chi2:
            if proba_simplifiee:
                tab[i + " " + significativite(p, arrondir=arrondir, value=False)] = t
            else:
                tab[i + " (p = %.1e)" % p] = t
        else:
            tab[i] = t

    # Mise en forme du tableau
    tab = pd.concat(tab)
    tab.index.names = [cont, c]

    # Retour du tableau mis en forme
    return tab


def tableau_croise_multiple(
    df,
    var_dep,
    var_indeps,
    poids=False,
    chi2=True,
    axis=0,
    ss_total=True,
    contenu="complet",
    arrondir=2,
    proba_simplifiee=False,
):
    """
    Tableau croisé multiples variables.
    Variable dépendantes vs. plusieurs variables indépendantes.

    Parameters
    ----------
    df : DataFrame
        Tableau de données
    var_dep : str or dic
        Variable dépendante en colonne
        Pour les dictionnaires : {nom:label}
    var_indeps : dict or list
        dictionnaire des variables indépendantes
        Pour les dictionnaires : {nom:label}
    poids : str, optionnal
        poids optionnel
    axis : int, optionnal
        sens des pourcentages,axis = 1 pour les colonnes
    ss_total : bool, optionnal
        présence de sous totaux
    contenu : str, optionnal
        données complètes, brutes ou pourcentages
    arrondir : int, optionnel
        nombre de décimales (défaut : 2)
    proba_simplifiee : bool, optionnel
        simplifier les probas (défaut : False)

    Returns
    -------
    DataFrame
        mise en forme du tableau croisé.

    Notes
    -----
    Manque une colonne tri à plat.

    """

    # Tester le format de l'entrée
    if not isinstance(df, pd.DataFrame):
        warnings.warn("Attention, ce n'est pas un tableau Pandas", UserWarning)
        return None
    if not isinstance(var_indeps, (list, dict)):
        warnings.warn(
            "Les variables ne sont pas renseignées sous le bon format", UserWarning
        )
        return None
    if var_dep not in df.columns:
        warnings.warn(
            "La variable {} n'est pas dans le tableau".format(var_dep), UserWarning
        )
        return None
    for i in var_indeps:
        if i not in df.columns:
            warnings.warn(
                "La variable {} n'est pas dans le tableau".format(i), UserWarning
            )
            return None

    # Noms des variables
    if isinstance(var_indeps, list):
        var_indeps = {i: i for i in var_indeps}

    t_all = {}

    # Compteur des totaux par croisement
    check_total = []

    # Boucle sur les variables indépendantes
    for i in var_indeps:
        # Tableau croisé pondéré (deux orientations possibles)
        if axis == 0:
            t_comp, t_ab, t_per, p = tableau_croise(
                df, i, var_dep, poids, verb=True, arrondir=arrondir
            )

        else:
            t_comp, t_ab, t_per, p = tableau_croise(
                df, var_dep, i, poids, verb=True, arrondir=arrondir
            )
            t_comp = t_comp.T
            t_ab = t_ab.T
            t_per = t_per.T

        # Sélection du contenu du tableau
        if contenu == "complet":
            t = t_comp
        elif contenu == "absolu":
            t = t_ab
        elif contenu == "pourcentage":
            t = t_per
        else:
            warnings.warn("Erreur dans le format du tableau", UserWarning)
            return None

        # Enlever les sous-totaux si besoin
        if not ss_total:
            t = t.drop("Total")

        dis = tri_a_plat(df, i, poids=poids, arrondir=arrondir)

        check_total.append(t.iloc[-1, -1])
        t["Distribution"] = dis[dis.columns[1]].apply(lambda x: "{}%".format(x))
        if chi2:
            if proba_simplifiee:
                t_all[
                    f"{var_indeps[i]} {significativite(p, arrondir=arrondir, value=False)}"
                ] = t
            else:
                t_all[f"{var_indeps[i]} (p = {p:.1e})"] = t
        else:
            t_all[var_indeps[i]] = t

    # Création d'un DataFrame
    t_all = pd.concat(t_all)
    t_all.columns.name = ""
    if langue == "fr":
        t_all.index.names = ["Variable", "Modalités"]
    elif langue == "en":
        t_all.index.names = ["Variable", "Modalities"]

    # Alerter sur les totaux différents
    if len(set(check_total)) != 1:
        warnings.warn(
            "Attention, les totaux par tableaux sont différents (valeurs manquantes)",
            UserWarning,
        )

    return t_all


def significativite(x, arrondir=2, value=False):
    """
    Mettre en forme la p-value.

    Parameters
    ----------
    x : float
        p-value
    arrondir : int, optionnel
        nombre de décimales (défaut : 4)

    Returns
    -------
    string
        valeur avec des étoiles.
    """

    # Tester le format
    if pd.isnull(x):
        return None

    # Arrondir
    x = round(x, arrondir)

    # Retourner la valeur avec le nombre d'étoiles associées
    if x < 0.001:
        if value:
            return "*** (p=" + str(x) + ")"
        else:
            return "*** (p < 0.001)"
    if x < 0.01:
        if value:
            return "** (p=" + str(x) + ")"
        else:
            return "** (p < 0.01)"
    if x < 0.05:
        if value:
            return "* (p=" + str(x) + ")"
        else:
            return "* (p < 0.05)"

    # Retourner la p-value mise en forme
    return "(p=" + str(x) + ")"


def tableau_reg_logistique(
    regression, data, var_indeps, sig=True, arrondir=2, notationscientifique=False
) -> DataFrame:
    """
    Mise en forme des résultats de la régression logistique.
    Lecture pour les SHS.

    Parameters
    ----------
    regression : statsmodel object from GLM
        modèle de régression
    data : DataFrame
        tableau des données
    var_indeps : dict or list
        liste des colonnes de la régression
    sig : bool, optionnel
        faire apparaître la significativité
    arrondir : int, optionnel
        nombre de décimales (défaut : 2)
    notationscientifique : bool, optionnel
        notation scientifique pour les p-values (défaut : False)

    Returns
    -------
    DataFrame
        tableau des résultats.


    Examples
    --------
    The dictionnary ind_var should be defined as {var:label}

    >>> import statsmodels.api as sm
    >>> modele = smf.glm(formula=f, data=data, family=sm.families.Binomial(),
                 freq_weights=data["weight"])
    >>> reg = modele.fit()
    >>> tableau_reg_logistique(reg,data,ind_var,sig=True)

    """

    # Séparation des variables et des effets d'interaction
    var_indeps_unique = {}
    for v in var_indeps:
        if "*" in v:  # cas d'une interaction
            for e in v.split("*"):
                if not e.strip() in var_indeps_unique:
                    var_indeps_unique[e.strip()] = e.strip()
        else:
            var_indeps_unique[v] = var_indeps[v]

    # Mise en forme du tableau général OR /
    table = np.exp(regression.conf_int())
    table["OR"] = round(np.exp(regression.params), arrondir)
    table["p"] = (
        round(regression.pvalues, arrondir)
        if not notationscientifique
        else regression.pvalues.apply(lambda x: f"{x:.2e}")
    )
    table["IC 95%"] = table.apply(
        lambda x: "%s [%s-%s]"
        % (
            str(round(x["OR"], arrondir)),
            str(round(x[0], arrondir)),
            str(round(x[1], arrondir)),
        ),
        axis=1,
    )

    # Ajout de la significativité
    if sig:
        table["p"] = table["p"].apply(lambda x: significativite(x, arrondir=arrondir))
    table = table.drop([0, 1], axis=1)

    # Transformation de l'index pour ajouter les références

    # Gestion à part de l'intercept qui n'a pas de modalité
    temp_intercept = list(table.loc["Intercept"])
    table = table.drop("Intercept")

    # Variables numériques
    var_num = data.select_dtypes(include=np.number).columns

    # Identification des références utilisées par la régression
    # Premier élément des modalités classées pour les variables non numériques
    refs = []
    for v in var_indeps_unique:
        if not v in var_num:
            r = sorted(data[v].dropna().unique())[0]  # premier élément
            refs.append(str(v) + "[T." + str(r) + "]")  # ajout de la référence

    # Ajout des références dans le tableau
    for i in refs:
        table.loc[i] = ["ref", " ", " "]

    # Création d'un MultiIndex Pandas par variable
    new_index = []
    for i in table.index:
        if ":" in i:  # cas où c'est une ligne d'interaction
            new_index.append(("var. interaction", i.replace("T.", "")))
        else:
            if "[T." in i:  # Si c'est une variable catégorielle
                tmp = i.split("[T.")
                var = (
                    tmp[0].replace("Q('", "").replace("')", "")
                )  # enlever la décoration
                new_index.append(
                    (var_indeps_unique[var], tmp[1][0:-1])
                )  # gérer l'absence dans le dictionnaire
            else:
                i = i.replace("Q('", "").replace("')", "")  # enlever la décoration
                new_index.append((var_indeps_unique[i], "numérique"))

    # Réintégration de l'Intercept dans le tableau
    new_index.append((".Intercept", ""))
    table.loc[".Intercept"] = temp_intercept

    # Réindexation du tableau
    if langue == "fr":
        names = ["Variable", "Modalité"]
    elif langue == "en":
        names = ["Variable", "Modality"]
    new_index = pd.MultiIndex.from_tuples(new_index, names=names)
    table.index = new_index
    table = table.sort_index()

    return table


def construction_formule(dep, indep):
    """
    Construit une formule de modèle linéaire.

    Parameters
    ----------
    dep : str
        variable dépendante
    indep: list
        liste des variables indépendantes

    Returns
    -------
    str : formule de régression

    """
    return dep + " ~ " + " + ".join(["Q('%s')" % i for i in indep])


def regression_logistique(
    df,
    dep_var,
    var_indeps,
    poids=False,
    table_only=True,
    arrondir=2,
    notationscientifique=False,
    sig=True,
):
    """
    Régression logistique binomiale pondérée.

    Parameters
    ----------
    df : DataFrame
        tableau des données
    dep_var : str
        variable dépendante
    var_indeps : dict or list
        liste des variables indépendantes
    poids : str, optionnel
        pondération
    table_only : bool
        seulement le tableau ou le modèle
    arrondir : int, optionnel
        nombre de décimales (défaut : 2)
    notationscientifique : bool, optionnel
        notation scientifique pour les p-values (défaut : False)

    Returns
    -------
    DataFrame : tableau des résultats
    """

    # S'il n'y a pas de pondération définie
    if not poids:
        df = df.copy()
        df["poids"] = 1
        poids = "poids"
    # Mettre les variables indépendantes en dictionnaire si nécessaire
    if isinstance(var_indeps, list):
        var_indeps = {i: i for i in var_indeps}

    # Construction de la formule
    f = construction_formule(dep_var, var_indeps)

    # Création du modèle
    modele = smf.glm(
        formula=f, data=df, family=sm.families.Binomial(), freq_weights=df[poids]
    )

    regression = modele.fit()

    # Retourner le tableau de présentation
    if table_only:
        tableau = tableau_reg_logistique(
            regression,
            df,
            var_indeps,
            sig=sig,
            arrondir=arrondir,
            notationscientifique=notationscientifique,
        )
        return tableau
    else:
        return regression


def likelihood_ratio(mod, mod_r):
    """
    Différence de déviance entre deux modèles logistiques (likelihood ratio)

    Parameters
    ----------
    mod : statsmodel object from GLM
        First model to compare
    mod_r : statsmodel object from GLM
        Second model to compare at

    Returns
    -------
    float : p-value of the likelihood ratio

    Notes
    -----
    Source : http://rnowling.github.io/machine/learning/2017/10/07/likelihood-ratio-test.html
    Testé en Rstats avec lmtest
    """
    val = [mod.llf, mod_r.llf]
    LR = 2 * (max(val) - min(val))  # rapport de déviance

    val = [mod.df_model, mod_r.df_model]
    diff_df = max(val) - min(val)  # différence de ddf

    p = chi2.sf(LR, diff_df)  # test de la significativité
    return p


def vers_excel(tables, file):
    """
    Écriture d'un ensemble de tableaux.
    Dans un fichier libre office (xlsc) avec titres.

    Parameters
    ----------
    tables : list or dict or DataFrame
        Données à écrire dans un fichier
    file: str
        chemin et nom du fichier de sortie (.xlsx)

    Returns
    -------
    None
    """

    # Transformation de l'entrée en dictionnaire
    if isinstance(tables, pd.DataFrame):
        tables = {"": tables}
    if isinstance(tables, list):
        tables = {"Tableau %d" % (i + 1): j for i, j in enumerate(tables)}
    if not isinstance(tables, dict):
        warnings.warn("Erreur dans le format des données rentrées", UserWarning)
        return None

    # Ouverture d'un fichier excel
    if not (file.endswith(".xlsx") or file.endswith(".xls")):
        warnings.warn("Le fichier à créer n'a pas la bonne extension", UserWarning)
        return None

    writer = pd.ExcelWriter(file, engine="openpyxl", mode="w")
    writer.book.create_sheet("Résultats")
    worksheet = writer.book.worksheets[0]
    writer.sheets["Résultats"] = worksheet
    curseur = 0  # ligne d'écriture
    # Boucle sur les tableaux
    for title in tables:
        worksheet.cell(curseur + 1, 1, title)
        tables[title].to_excel(writer, sheet_name="Résultats", startrow=curseur + 2)
        curseur += 2 + tables[title].shape[0] + 4
    writer.book.save(file)

    return None


def moyenne_ponderee(colonne, poids):
    """
    Calculer une moyenne pondérée.

    Parameters
    ----------
    colonne : Serie or list
        liste des données numériques
    poids : Serie or list
        liste des pondéérations

    Returns
    -------
    float : moyenne pondérée
    """
    return np.average(colonne, weights=poids)


def ecart_type_pondere(colonne, poids):
    """
    Ecart-type pondéré.

    Parameters
    ----------
    colonne : Serie or list
        liste des données numériques
    poids : Serie or list
        liste des pondéérations

    Returns
    -------
    float : écart-type pondéré
    """
    average = np.average(colonne, weights=poids)
    variance = np.average((colonne - average) ** 2, weights=poids)
    return math.sqrt(variance)


def _escape_quotes(variable: str) -> str:
    """
    Complète la fonction Q() de patsy pour échapper les caractères problématiques
    Parameters
    ----------
    variable : str
        nom de variable

    Returns
    -------
    str
        nom de variable modifiée

    """
    return variable.replace('"', '\\"')


def catdes(
    df: pd.DataFrame,
    vardep: str,
    varindep: List[str] | None = None,
    proba: float = 0.05,
    poids: str | None = None,
    mod: bool = False,
    arrondir: int = 2,
    notationscientifique: bool = False,
):
    """
    Calcule la relation entre une variable catégorielle et plusieurs variables.
    Implémentation de la fonction catdes de FactoMineR.
    Attention : encore en version BETA

    Parameters
    ----------
    df : DataFrame
    vardep : string
        variable catégorielle dépendante
    varindep : list (optionnal)
        liste des variables indépendantes
    proba : float (optionnal)
        niveau de significativité appliqué
    poids : string (optionnal)
        colonne de pondération
    mod : bool (optionnal)
        calculer la relation entre modalités
    arrondir : int, optionnel
        nombre de décimales (défaut : 2)
    notationscientifique : bool, optionnel
        notation scientifique pour les p-values (défaut : False)

    Returns
    -------
    DataFrame
        Tableau des associations entre variables quanti
    DataFrame
        Tableau des associations entre variables quali
    DataFrame (if mod = True)
        Tableau des associations entre modalités qualitatives.
    DataFrame (if mod = True)
        Tableau des associations entre modalités quantitatives.

    Notes
    -----
    Participation de Ahmadou Dicko pour passage R vers Python.
    Code initial en R dans FactoMineR.
    Passage en Python initié avec Inno3.
    Résultats testés comparativement avec R qui correspondent.
    Il manque encore les tests.

    """

    # Copie du tableau
    df = df.copy()

    # Variable dépendante catégorielle
    if is_numeric_dtype(df[vardep]):
        warnings.warn("Attention, la variable dépendante est numérique", UserWarning)
        return None

    # Construction de la liste de variables
    cols_num = []
    cols_cat = []
    if not varindep:
        # Cas où les variables ne sont pas proposées
        cols_cat = [
            c for c in df.columns if not is_numeric_dtype(df[c]) and c != vardep
        ]  # sans la variable dépendante
        cols_num = [
            c for c in df.columns if is_numeric_dtype(df[c]) and c != poids
        ]  # sans la pondération
    else:
        # Cas où les variables sont proposées
        for i in varindep:
            if is_numeric_dtype(df[i]):
                cols_num.append(i)
            else:
                cols_cat.append(i)

    # Pondération à 1 si pas de pondération
    if not poids:
        df["poids"] = [1] * len(df)
        poids = "poids"

    # Calcul de l'association par variables

    # Cas des variables catégorielles

    tableau_cat_var = []
    var_cat_corr = []

    # Pour chaque variable
    for v in cols_cat:
        # Calcul du tableau croisé
        t, a, p, p_val = tableau_croise(
            df, vardep, v, poids=poids, verb=True, arrondir=arrondir
        )
        a = a.drop(index="Total").drop(columns="Total")

        # Calcul du chi2
        k, p, f, t = chi2_contingency(a, correction=False)

        # Ajout aux résultats si significatif
        if p < proba:
            tableau_cat_var.append([v, p, f])
            var_cat_corr.append(v)

    # Mettre en forme le tableau
    tableau_cat_var = (
        pd.DataFrame(tableau_cat_var, columns=[vardep, "p", "df"])
        .set_index(vardep)
        .sort_values("p")
    )

    # Cas des variables numériques

    tableau_num_var = []
    var_num_corr = []

    # Pour chaque variable numérique
    for v in cols_num:
        # Calcul d'une ANOVA
        # Utilisation de Q() pour échapper les variables
        #  See https://patsy.readthedocs.io/en/latest/builtins-reference.html#patsy.builtins.Q
        model = ols(
            f'Q("{_escape_quotes(v)}") ~ C(Q("{_escape_quotes(vardep)}"))', data=df
        ).fit()
        aov_table = sm.stats.anova_lm(model, typ=2)

        # Paramètre de l'association
        eta2 = aov_table.iloc[0, 0] / aov_table.iloc[:, 0].sum()
        p = aov_table.iloc[0, 3]

        # Ajout aux résultats si significatif
        if p <= proba:
            tableau_num_var.append([v, eta2, p])
            var_num_corr.append(v)

    # Mettre en forme le tableau
    tableau_num_var = pd.DataFrame(
        tableau_num_var, columns=[vardep, "Eta 2", "p-value"]
    ).set_index(vardep)

    # Fin de la fonction si mod = False
    if not mod:
        return tableau_cat_var, tableau_num_var

    # Si mod = True, associations avec les modalités

    # Cas des variables catégorielles

    # Création des colonnes 0/1 par modalités
    tab_dep = pd.get_dummies(df[[vardep]])
    tab_ind = pd.get_dummies(df[var_cat_corr])
    tab_all = pd.get_dummies(
        df[list(set([vardep] + var_cat_corr + [poids]))]
    )  # assurer l'unicité des colonnes
    n = len(df)

    # Boucle sur les variables
    tableau_cat_mod = {}
    for categorie in tab_dep.columns:
        res_cat = []
        for modalite in tab_ind.columns:
            # Calcul d'un test hypergéométrique
            # Arrondi car pondération
            n_kj = round(
                (
                    tab_all[tab_all[categorie] == 1][modalite]
                    * tab_all[tab_all[categorie] == 1][poids]
                ).sum()
            )
            n_j = round((tab_all[modalite] * tab_all[poids]).sum())
            n_k = round((tab_all[categorie] * tab_all[poids]).sum())
            # Test dans catdes de FactomineR
            # 2 * P(N >= n_kj-1) + P(n_kj)
            prob_inf2 = hypergeom.cdf(n_kj - 1, n, n_j, n_k) * 2 + hypergeom.pmf(
                n_kj, n, n_j, n_k
            )
            # 2 * P(N < n_kj) + P(n_kj)
            prob_sup2 = (1 - hypergeom.cdf(n_kj, n, n_j, n_k)) * 2 + hypergeom.pmf(
                n_kj, n, n_j, n_k
            )
            # Prendre la valeur minimale
            p_min2 = min(prob_inf2, prob_sup2)
            # Calcul de la valeur test à partir d'une loi normale unitaire
            V = (1 - 2 * int(n_kj / n_j > n_k / n)) * norm.ppf(p_min2 / 2)
            # Calcul du chi2 sur le tableau croisé 2x2
            t, a, p, p_val = tableau_croise(
                tab_all, categorie, modalite, poids, verb=True, arrondir=arrondir
            )
            a = a.drop(index="Total").drop(columns="Total")
            k, p_chi2, f, t = chi2_contingency(a, correction=False)
            #  Ajout aux résultats si significatif
            if p_min2 / 2 < proba:
                res_cat.append(
                    [
                        modalite,
                        round(100 * n_kj / n_j, 2),
                        round(100 * n_kj / n_k, 2),
                        round(100 * n_j / n, 2),
                        round(V, 2),
                        p_min2 / 2,
                        p_chi2,
                    ]
                )

        # Mise en forme du tableau
        res_cat = (
            pd.DataFrame(
                res_cat,
                columns=[
                    "var",
                    "Cla/Mod (n_kj/n_j)",
                    "Mod/Cla (n_kj/n_k)",
                    "Proportion globale (n_j/n)",
                    "Valeur test",
                    "p hyper",
                    "p chi2",
                ],
            )
            .sort_values("Valeur test", ascending=False, key=abs)
            .set_index("var")
        )
        tableau_cat_mod[categorie] = res_cat

    # Mise en forme final du tableau
    # tableau_cat_mod = pd.concat(tableau_cat_mod)
    tableau_cat_mod = pd.concat(
        {
            tab: tableau_cat_mod[tab]
            for tab in tableau_cat_mod
            if not (tableau_cat_mod[tab]).empty
        }
    )

    # Cas des variables numériques

    var_dep_mod = df[vardep].unique()
    tableau_num_mod = {i: [] for i in var_dep_mod}

    # Pour chaque variable
    for v in var_num_corr:
        # Calcul de paramètres
        moy_mod = df.groupby(vardep).apply(lambda x: moyenne_ponderee(x[v], x[poids]))
        n_mod = df.groupby(vardep)[v].count()
        n = sum(n_mod)
        sd_mod = df.groupby(vardep).apply(lambda x: ecart_type_pondere(x[v], x[poids]))
        moy = moyenne_ponderee(df[v], df[poids])
        sd = ecart_type_pondere(df[v], df[poids])

        # Pour chaque modalités de la variable dépendante
        for m in var_dep_mod:
            # Calcul d'un test
            v_test = (
                (moy_mod.loc[m] - moy)
                / sd
                * math.sqrt(n_mod.loc[m])
                / math.sqrt((n - n_mod.loc[m]) / (n - 1))
            )
            p_value = (1 - norm.cdf(abs(v_test))) * 2

            # Ajout à la sortie si significatif au seuil
            if p_value <= proba:
                tableau_num_mod[m].append(
                    [v, v_test, p_value, moy_mod.loc[m], moy, sd_mod.loc[m], sd]
                )

    # Mise en forme des tableaux
    # Ce n'est pas très joli ...
    tableau_num_mod = pd.concat(
        {
            i: pd.DataFrame(
                tableau_num_mod[i],
                columns=[
                    "var",
                    "Valeur test",
                    "p-value",
                    "Moy mod",
                    "Moy glob",
                    "Std mod",
                    "Std glob",
                ],
            )
            .set_index("var")
            .sort_values("Valeur test", ascending=False, key=abs)
            for i in tableau_num_mod
        }
    )

    # Retourner les tableaux
    return tableau_cat_var, tableau_num_var, tableau_cat_mod, tableau_num_mod


# ----------------------------------------------------------------------
# Classes et fonctions encore en développement


def tableau_reg_logistique_distribution(
    df, dep_var, var_indeps, poids=False, arrondir=2, notationscientifique=False
):
    # Noms des variables
    if isinstance(var_indeps, list):
        var_indeps = {i: i for i in var_indeps}

    # régression logistique
    reg = regression_logistique(
        df,
        dep_var,
        var_indeps,
        poids=poids,
        table_only=True,
        arrondir=arrondir,
        notationscientifique=notationscientifique,
    )

    # Distribution
    dis = {}
    for i in var_indeps:
        dis[var_indeps[i]] = tri_a_plat(df, i, poids=poids, arrondir=arrondir)[
            "Pourcentage (%)"
        ].drop("Total")
    dis = pd.concat(dis, axis=0)
    dis.index.names = ["Variable", "Modalité"]

    # garder uniquement l'étoile ...
    reg["s"] = reg["p"].apply(
        lambda x: "" if pd.isnull(x) else "".join([i for i in x if i == "*"])
    )

    tab = reg[["IC 95%", "s"]].join(dis)

    # ajout étoile
    tab["IC 95%"] = tab.apply(lambda x: str(x["IC 95%"]) + " " + x["s"], axis=1)

    return tab[["Pourcentage (%)", "IC 95%"]]


def cramers_corrected_stat(confusion_matrix):
    """calculate Cramers V statistic for categorial-categorial association.
    uses correction from Bergsma and Wicher,
    Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape

    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def plot_acm(model, data, x=0, y=1, dims=(11, 8), keep_prefix=True, adjust_labels=True):
    """
    Parameters
    ----------
    model : prince.mca.MCA
        en: MCA model (fitted) from the PRINCE library.
        fr: Modèle d'ACM (déjà fitté) de la librairie PRINCE.
    data : pandas.core.frame.DataFrame
        en: Pandas dataframe containing data.
        fr: DataFrame pandas contenant les données.
    x : int, optionel
        en: Identifier of the component to plot horizontally. By default, 0.
        fr: Identifiant de l'axe factoriel qu'on souhaite représenter horizontalement. 0 par défaut.
    y : int, optionel
        en: Identifier of the component to plot vertically. By default, 1.
        fr: Identifiant de l'axe factoriel qu'on souhaite représenter verticalement. 1 par défaut.
    dims : tuple, optional
        en: Plot size. (11, 8) by default.
        fr: Taille du graphique. (11, 8) par défaut.
    keep_prefix : boolean, optional
        en: To prepend (or not) the name of the variable to the value (e.g. if set to True, will display something like "COLOR_purple",
        otherwise will display "purple").
        fr: Indique si le texte affiché à côté de chaque point projeté contient un préfixe contenant le nom de la variable.
        Si défini à "False", seul l'intitulé de la modalité apparait.
    adjust_text : boolean, optional
        en: Preventing text from overlapping if set to True.
        fr: Si défini à "True", réorganise automatiquement les labels de chaque point de manière à ce qu'ils ne se chevauchent pas.

    Returns
    -------
    plot : matplotlib.axes._subplots.AxesSubplot
        Graphique matplotlib.

    Notes
    -------
    Proposé et codé par Jean-Baptiste Bertrand
    """
    variables = data.columns
    # variance = model.eigenvalues_summary
    variance = model.percentage_of_variance_
    coord = model.column_coordinates(data)
    # coord contient les coordonnées de chaque modalité sur chaque axe factoriel
    # on ajoute une colonne "variable" pour le lien modalités/variable
    coord["variable"] = ""
    new_index = []

    for i in range(0, len(coord)):
        row = coord.iloc[i]
        var_name = row.name
        for variable in variables:
            values = data[variable].dropna().unique()
            for value in values:
                comb = "_".join([variable, value])
                if comb == var_name:
                    coord.iloc[i, coord.columns.get_loc("variable")] = variable
                    new_index.append(coord.iloc[i].name.replace(variable + "_", ""))
    if not keep_prefix:
        coord.index = new_index
    # On représente ensuite les modalités sous forme d'un nuage de points
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=dims)
    plot = sns.scatterplot(
        x=coord[x],
        y=coord[y],
        hue=coord["variable"],
        ax=ax,
    )
    # on modifie l'intitulé des axex
    ax.set_xlabel(f"Dimension {str(x)} ({round(variance[x], 1)} %)")
    ax.set_ylabel(f"Dimension {str(y)} ({round(variance[y], 1)} %)")
    # on ajoute des lignes pointillées aux origines, pour faciliter l'analyse
    plot.axhline(y=0, color="black", linestyle="--", linewidth=0.5)
    plot.axvline(x=0, color="black", linestyle="--", linewidth=0.5)

    # on ajoute l'intitulé des points
    txts = []
    for line in range(0, coord.shape[0]):
        txt = plot.text(
            coord[x][line] + 0.03,
            coord[y][line],
            coord.index[line],
        )
        txts.append(txt)
    if adjust_labels == True:
        adjust_text(txts)

    # derniers réglages :
    # on supprime l'encadré autour du graphique
    # on déplace la légende à l'extérieur du graphique
    sns.despine()
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    return plot
