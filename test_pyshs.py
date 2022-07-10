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
    assert pyshs.tri_a_plat(df_test, "C2", "C1").loc["A", "Effectif redressé"] == 21


def test_tableau_croise(df_test):
    assert isinstance(pyshs.tableau_croise(df_test, "C2", "C3", "C1"),pd.DataFrame)


def test_tableau_croise_multiple(df_test):
    assert isinstance(
        pyshs.tableau_croise_multiple(
            df_test, "C4", {"C2": "colonne 1", "C2": "colonne 2"}
        ),
        pd.DataFrame,
    )


def test_regression_logistique(df_test):
    assert isinstance(pyshs.regression_logistique(df_test, "C5", ["C1"]), pd.DataFrame)


def test_moyenne_ponderee():
    assert pyshs.moyenne_ponderee([1, 2, 3], [1, 1, 2]) == 2.25


def test_ecart_type_pondere():
    assert pyshs.ecart_type_pondere([1, 1, 1], [10, 1, 2]) == 0


if __name__ == "__main__":

    print("just run 'pytest' to test this library")
