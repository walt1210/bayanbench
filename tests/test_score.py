"""Unit tests for name normalization and scoring.

Official lists here are small illustrative examples, not real PSGC data.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bayanbench.normalize import is_abstention, normalize_name  # noqa: E402
from bayanbench.score import (ABSTAINED, CORRECT, OUTDATED, WRONG, score_barangays,  # noqa: E402
                              score_choice, score_count)


def test_normalize_variants():
    assert normalize_name("City of San Fernando") == normalize_name("San Fernando City") == "san fernando"
    assert normalize_name("Dasmariñas") == normalize_name("DASMARINAS")
    assert normalize_name("Sto. Tomas") == normalize_name("Santo Tomas")
    assert normalize_name("Poblacion (Pob.)") == "poblacion"
    assert normalize_name("Sta. Rita") == "santa rita"


def test_abstention():
    for a in [None, "", "Unknown", "I don't know", "N/A", [], ["unknown"]]:
        assert is_abstention(a), a
    for a in ["Pampanga", ["Dau"], "0"]:
        assert not is_abstention(a), a


def test_province():
    assert score_choice("Pampanga", ["Pampanga"], "province") == CORRECT
    assert score_choice("Province of Pampanga", ["Pampanga"], "province") == CORRECT
    assert score_choice("Tarlac", ["Pampanga"], "province") == WRONG
    assert score_choice("I don't know", ["Pampanga"], "province") == ABSTAINED
    # Trap: a city with the same base name is NOT the province.
    assert score_choice("Quezon City", ["Quezon"], "province") == WRONG


def test_outdated():
    gold = ["Region IX", "Zamboanga Peninsula"]
    old = ["BARMM", "Bangsamoro"]
    assert score_choice("Zamboanga Peninsula", gold, outdated_options=old) == CORRECT
    assert score_choice("BARMM", gold, outdated_options=old) == OUTDATED
    assert score_choice("Region XI", gold, outdated_options=old) == WRONG
    assert score_choice("Maguindanao", ["Maguindanao del Norte"], "province",
                        outdated_options=["Maguindanao"]) == OUTDATED


def test_city_or_municipality():
    assert score_choice("city", ["City"], "unit_type") == CORRECT
    assert score_choice("It is a component city", ["City"], "unit_type") == CORRECT
    assert score_choice("Municipality", ["City"], "unit_type") == WRONG
    assert score_choice("municipality", ["Mun"], "unit_type") == CORRECT
    # Hedging both answers must not count as correct.
    assert score_choice("city or municipality", ["City"], "unit_type") == WRONG


def test_count():
    assert score_count("33", 33)["exact"]
    r = score_count("31 barangays", 33)
    assert r["outcome"] == WRONG and r["within_tolerance"]
    assert not score_count(60, 33)["within_tolerance"]
    assert score_count("unknown", 33)["outcome"] == ABSTAINED


def test_barangays():
    official = ["Dau", "Santo Rosario", "Poblacion", "San Francisco", "Mabiga"]
    r = score_barangays(["Dau", "Sto. Rosario", "Mabiga"], official)
    assert r["outcome"] == CORRECT and r["real"] == 3 and not r["invented"]

    r = score_barangays("Dau, Poblacion, Barangay Malinis", official)
    assert r["outcome"] == WRONG and r["real"] == 2 and r["invented"] == ["Barangay Malinis"]

    # Trap: a similar-looking but different name must count as invented.
    r = score_barangays(["San Fernando"], official)
    assert r["invented"] == ["San Fernando"]

    # Repeating one real barangay does not earn extra credit.
    assert score_barangays(["Dau", "dau"], official)["real"] == 1

    assert score_barangays([], official)["outcome"] == ABSTAINED


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed:", name)
