import pandas as pd


def test_import():
    data = pd.DataFrame([[0, 1], [2, 3]])

    assert data.iloc[1, 0] == 2
