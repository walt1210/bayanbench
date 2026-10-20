"""Retriever tests on the real PSGC tables."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bayanbench.retrieve import PSGCRetriever  # noqa: E402

LGUS = pd.read_csv(ROOT / "data/processed/lgus.csv", dtype={"code": str})
BGY = pd.read_csv(ROOT / "data/processed/barangays.csv", dtype={"code": str, "lgu_code": str})
R = PSGCRetriever(LGUS, BGY)


def first(question):
    codes = R.search(question)
    return LGUS.set_index("code").loc[codes[0], "name"] if codes else None


def test_town_named_like_its_province():
    # Province name is also a city name: the town before it must win.
    assert first("How many barangays are in Moncada, Tarlac, Philippines?") == "Moncada"
    assert first("Is Balayan, Batangas, Philippines, a city or a municipality?") == "Balayan"
    # Province name is also a longer municipality name.
    assert first("Name three barangays in Kalamansig, Sultan Kudarat, Philippines.") == "Kalamansig"


def test_longer_name_not_split():
    assert first("Is San Jose del Monte, Bulacan, Philippines, a city or a municipality?") == "City of San Jose Del Monte"


def test_shared_name_uses_province():
    a = R.search("How many barangays are in San Jose, Batangas, Philippines?")
    b = R.search("How many barangays are in San Jose, Tarlac, Philippines?")
    prov = LGUS.set_index("code").province
    assert prov[a[0]] == "Batangas" and prov[b[0]] == "Tarlac"


def test_record_text():
    code = R.search("Name three barangays in Manila, National Capital Region (NCR), Philippines.")[0]
    text = R.record(code)
    assert "highly urbanized city" in text and "not under any province" in text
    assert "and 857 more" in text


def test_no_match():
    text, codes = R.context("What is the capital of France?")
    assert codes == [] and "No matching record" in text


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed:", name)
