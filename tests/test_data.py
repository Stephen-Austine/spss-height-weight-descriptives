from pathlib import Path

import pyreadstat
import pytest

DATA = Path(__file__).resolve().parents[1] / "data" / "dataset_2A.sav"


def load():
    return pyreadstat.read_sav(DATA)


def test_shape_and_missing():
    df, _ = load()
    assert df.shape == (12, 4)
    assert df.GENDER.isna().sum() == 2
    assert df.HEIGHT.isna().sum() == 2


def test_descriptives():
    df, _ = load()
    assert df.HEIGHT.mean() == pytest.approx(65.7)
    assert df.HEIGHT.std() == pytest.approx(2.830, abs=1e-3)
    assert df.WEIGHT.mean() == pytest.approx(136.9)
    assert df.HEIGHT.skew() == pytest.approx(0.528, abs=1e-3)


def test_gender_frequencies_and_labels():
    df, meta = load()
    assert df.GENDER.value_counts().to_dict() == {0.0: 6, 1.0: 4}
    assert meta.variable_value_labels["GENDER"] == {0.0: "male", 1.0: "female"}


def test_viewer_output_present():
    spv = DATA.parents[1] / "output" / "spss_output_A2.spv"
    assert spv.stat().st_size > 0
