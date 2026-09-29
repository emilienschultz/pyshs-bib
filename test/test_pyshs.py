import pyshs
import pandas as pd
import pytest


# A fixture is a magical function that will called for each test with a
# parameter of the same name, and the result of the function
# will be given to the test function.
# We do not use a global variable as tests could "mutate" the global variable.
@pytest.fixture
def df_test():

    return pd.DataFrame(
        [
            [1, "A", "c", "python", 0],
            [10, "B", "c", "python", 0],
            [20, "A", "b", "R", 1],
            [0, "A", "d", "R", 1],
        ],
        columns=["C1", "C2", "C3", "C4", "C5"],
    )


# tests unitaires pour chacune des fonctions de la bibliothèque

def test_description(df_test):
    assert isinstance(pyshs.description(df_test), pd.DataFrame)

def test_tri_a_plat(df_test):
    assert isinstance(pyshs.tri_a_plat(df_test, "C2", "C1"),pd.DataFrame)
    assert pyshs.tri_a_plat(df_test, "C2", "C1").shape == (3,2)

def test_tableau_croise(df_test):
    assert isinstance(pyshs.tableau_croise(df_test, "C2", "C3", "C1"),pd.DataFrame)
    assert len(pyshs.tableau_croise(df_test, "C2", "C3",verb=True))==4

def test_tableau_croise_multiple(df_test):
    assert isinstance(
        pyshs.tableau_croise_multiple(
            df_test, "C4", {"C2": "colonne 1", "C2": "colonne 2"}
        ),
        pd.DataFrame,
    )

def test_regression_logistique(df_test):
    assert isinstance(pyshs.regression_logistique(df_test, "C5", ["C1"]), pd.DataFrame)

def test_regression_logistique_pseudo_r2_mcfadden(df_test, capsys):
    import statsmodels.formula.api as smf
    pyshs.regression_logistique(df_test, "C5", ["C1"], arrondir=4)
    attendu = round(smf.logit("C5 ~ C1", data=df_test).fit(disp=0).prsquared, 4)
    assert f"Pseudo R² (McFadden) : {attendu}" in capsys.readouterr().out

def test_regression_logistique_multinomiale(df_test):
    df = df_test.copy()
    df["C6"] = ["A", "B", "C", "A"]
    result = pyshs.regression_logistique_multinomiale(df, "C6", ["C1"])
    assert isinstance(result, pd.DataFrame)
    assert result.index.nlevels == 2
    assert result.columns.nlevels == 2

def test_moyenne_ponderee():
    assert pyshs.moyenne_ponderee([1, 2, 3], [1, 1, 2]) == 2.25

def test_ecart_type_pondere():
    assert pyshs.ecart_type_pondere([1, 1, 1], [10, 1, 2]) == 0

def test_significativite_seuils_sur_valeur_brute():
    assert pyshs.significativite(0.004, arrondir=2) == "** (p < 0.01)"
    assert pyshs.significativite(0.0096, arrondir=2) == "** (p < 0.01)"
    assert pyshs.significativite(0.0004, arrondir=2) == "*** (p < 0.001)"
    assert pyshs.significativite(0.049, arrondir=2) == "* (p < 0.05)"
    assert pyshs.significativite(0.004, arrondir=2, value=True) == "** (p=0.0)"
    assert pyshs.significativite(0.2, arrondir=2) == "(p=0.2)"

def test_tableau_reg_logistique_reference_categorical():
    import numpy as np
    rng = np.random.default_rng(0)
    d = pd.DataFrame({"x": rng.choice(["a", "b", "c"], 300), "y": rng.integers(0, 2, 300)})
    d["x"] = pd.Categorical(d["x"], categories=["c", "a", "b"])
    tab = pyshs.regression_logistique(d, "y", ["x"])
    modalites = list(tab.loc["x"].index)
    assert sorted(modalites) == ["a", "b", "c"]
    assert tab.loc[("x", "c"), "OR"] == "ref"
    assert tab.loc[("x", "a"), "OR"] != "ref"

def test_tableau_reg_logistique_multinomiale_reference_categorical():
    import numpy as np
    rng = np.random.default_rng(0)
    d = pd.DataFrame({"x": rng.choice(["a", "b", "c"], 300), "y": rng.choice(["u", "v", "w"], 300)})
    d["x"] = pd.Categorical(d["x"], categories=["c", "a", "b"])
    tab = pyshs.regression_logistique_multinomiale(d, "y", ["x"])
    assert sorted(tab.loc["x"].index) == ["a", "b", "c"]

if __name__ == "__main__":

    print("just run 'pytest' to test this library")
