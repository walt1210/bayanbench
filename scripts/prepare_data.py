"""Turn the official PSGC Excel file into clean tables for the benchmark.

Input:  PSA PSGC publication datafile (e.g. PSGC-2Q-2026-Publication-Datafile.xlsx)
Output: data/processed/lgus.csv        one row per city or municipality
        data/processed/barangays.csv   one row per barangay, linked to its city or municipality

Hierarchy comes from the 10-digit PSGC code:
    region = first 2 digits, province = first 5, city or municipality = first 7.
Barangays of the City of Manila sit under its 14 districts ("SubMun"),
which are rolled up to the City of Manila.

Usage:
    python scripts/prepare_data.py data/raw/PSGC-2Q-2026-Publication-Datafile.xlsx
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bayanbench.normalize import normalize_name  # noqa: E402

ISLAND_GROUP = {  # by region code
    "13": "Luzon", "14": "Luzon", "01": "Luzon", "02": "Luzon", "03": "Luzon",
    "04": "Luzon", "17": "Luzon", "05": "Luzon",
    "06": "Visayas", "18": "Visayas", "07": "Visayas", "08": "Visayas",
    "09": "Mindanao", "10": "Mindanao", "11": "Mindanao", "12": "Mindanao",
    "16": "Mindanao", "19": "Mindanao",
}

# Region membership that changed recently. An answer naming the previous
# region is scored as "outdated", not as a hallucination.
# Sources: PSA press release on RA 12000 (Negros Island Region, June 2024);
# Executive Order 91 (Sulu to Region IX, 30 July 2025).
PREVIOUS_REGION = {
    "Negros Occidental": "Region VI (Western Visayas)",
    "City of Bacolod": "Region VI (Western Visayas)",
    "Negros Oriental": "Region VII (Central Visayas)",
    "Siquijor": "Region VII (Central Visayas)",
    "Sulu": "Bangsamoro Autonomous Region In Muslim Mindanao (BARMM)",
}
# Province that changed recently (Maguindanao split by RA 11550; plebiscite 2022).
PREVIOUS_PROVINCE = {
    "Maguindanao del Norte": "Maguindanao",
    "Maguindanao del Sur": "Maguindanao",
}

ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8,
         "IX": 9, "X": 10, "XI": 11, "XII": 12, "XIII": 13}
EXTRA_REGION_NAMES = {
    "13": ["NCR", "Metro Manila", "National Capital Region"],
    "14": ["CAR", "Cordillera", "Cordillera Administrative Region"],
    "04": ["CALABARZON", "Region IV-A", "Region 4A", "Region 4-A"],
    "17": ["MIMAROPA", "Region IV-B", "Region 4B", "Region 4-B", "Southwestern Tagalog Region"],
    "18": ["Negros Island Region", "NIR", "Region XVIII", "Region 18"],
    "19": ["BARMM", "Bangsamoro", "Bangsamoro Autonomous Region in Muslim Mindanao"],
    "16": ["Caraga", "Region XIII", "Region 13"],
}


def region_options(code2: str, full_name: str) -> list[str]:
    """Every acceptable way of naming a region."""
    opts = {full_name}
    m = re.match(r"Region ([IVX]+(?:-[AB])?)\s*\((.+)\)", full_name)
    if m:
        numeral, common = m.groups()
        opts |= {f"Region {numeral}", common}
        base = numeral.split("-")[0]
        if base in ROMAN and "-" not in numeral:
            opts |= {f"Region {ROMAN[base]}"}
    paren = re.search(r"\(([^)]+)\)", full_name)
    if paren:
        opts.add(paren.group(1))
    opts |= set(EXTRA_REGION_NAMES.get(code2, []))
    return sorted(opts)


def display_name(name: str) -> str:
    """Remove 'City' wording so the question does not reveal the unit type.

    Every city in PSGC has 'City' in its official name, in one of these forms:
    'City of Angeles', 'Mabalacat City', 'Science City of Muñoz',
    'Island Garden City of Samal'. All become the bare name.
    """
    name = re.sub(r"^(?:[A-Za-z ]+ )?City of\s+", "", name)
    name = re.sub(r"\s+City$", "", name)
    return name.strip()


def main(xlsx: str, out_dir: str = "data/processed") -> None:
    df = pd.read_excel(xlsx, sheet_name="PSGC", dtype={"10-digit PSGC": str})
    df = df.iloc[:, [0, 1, 3, 4, 5, 6, 7, 8, 10]]
    df.columns = ["code", "name", "level", "old_names", "city_class",
                  "income", "urban", "pop2024", "status"]
    df["name"] = df["name"].astype(str).str.strip()

    reg = df[df.level == "Reg"]
    regions = dict(zip(reg.code.str[:2], reg.name))
    prov = df[df.level == "Prov"]
    provinces = dict(zip(prov.code.str[:5], prov.name))
    submun_parent = {c: c[:5] + "00000" for c in df[df.level == "SubMun"].code}

    lgu = df[df.level.isin(["City", "Mun"])].copy()
    lgu_codes = set(lgu.code)

    # --- barangays ---------------------------------------------------------
    bgy = df[df.level == "Bgy"].copy()
    parent = bgy.code.str[:7] + "000"
    bgy["lgu_code"] = parent.map(lambda p: p if p in lgu_codes else submun_parent.get(p))
    assert bgy.lgu_code.notna().all(), "barangay without a city or municipality"
    assert bgy.lgu_code.isin(lgu_codes).all()

    # --- cities and municipalities ------------------------------------------
    lgu["display_name"] = lgu.name.map(display_name)
    lgu["name_key"] = lgu.display_name.map(normalize_name)
    lgu["unit_type"] = lgu.level.map({"City": "city", "Mun": "municipality"})
    lgu["region_code"] = lgu.code.str[:2]
    lgu["region"] = lgu.region_code.map(regions)
    lgu["island_group"] = lgu.region_code.map(ISLAND_GROUP)
    lgu["province"] = lgu.code.str[:5].map(provinces)            # NaN for independent cities
    lgu["income_class"] = lgu.income.astype(str).str.replace("*", "", regex=False).str.strip()
    lgu.loc[~lgu.income_class.isin(["1st", "2nd", "3rd", "4th", "5th"]), "income_class"] = None
    counts = bgy.groupby("lgu_code").size()
    lgu["n_barangays"] = lgu.code.map(counts).fillna(0).astype(int)
    urban_share = bgy.assign(u=bgy.urban.eq("U")).groupby("lgu_code").u.mean()
    lgu["urban_share"] = lgu.code.map(urban_share).round(3)

    lgu["is_ncr"] = lgu.region_code.eq("13")
    lgu["independent_city"] = lgu.province.isna()            # HUCs, Isabela City, NCR, new SGA towns
    lgu["new_lgu"] = lgu.code.str.startswith("19999")         # BARMM Special Geographic Area, created 2024
    key_counts = lgu.name_key.value_counts()
    lgu["ambiguous_name"] = lgu.name_key.map(key_counts).gt(1)

    def recent_change(row):
        reasons = []
        if row.province in PREVIOUS_REGION or row["name"] in PREVIOUS_REGION:
            reasons.append("region")
        if row.province in PREVIOUS_PROVINCE:
            reasons.append("province")
        if row.new_lgu:
            reasons.append("new")
        return ";".join(reasons) or None

    lgu["recent_change"] = lgu.apply(recent_change, axis=1)
    lgu["previous_region"] = lgu.apply(
        lambda r: PREVIOUS_REGION.get(r["name"]) or PREVIOUS_REGION.get(r.province), axis=1)
    lgu["previous_province"] = lgu.province.map(PREVIOUS_PROVINCE)
    lgu["region_options"] = lgu.apply(
        lambda r: json.dumps(region_options(r.region_code, r.region)), axis=1)
    lgu["prev_region_options"] = lgu.previous_region.map(
        lambda pr: json.dumps(region_options(
            next(k for k, v in regions.items() if v == pr), pr)) if isinstance(pr, str) else None)

    # Context shown in questions so that shared names are unambiguous.
    lgu["context"] = lgu.province.fillna(lgu.region)
    lgu["context_key"] = lgu.name_key + "|" + lgu.context.map(normalize_name)
    lgu["ambiguous_in_context"] = lgu.context_key.duplicated(keep=False)

    cols = ["code", "name", "display_name", "name_key", "unit_type", "city_class",
            "income_class", "pop2024", "urban_share", "n_barangays", "province",
            "region_code", "region", "island_group", "is_ncr", "independent_city",
            "new_lgu", "ambiguous_name", "ambiguous_in_context", "recent_change",
            "previous_region", "previous_province", "region_options",
            "prev_region_options", "context"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lgu[cols].to_csv(out / "lgus.csv", index=False)
    bgy[["code", "name", "lgu_code", "urban", "pop2024", "status"]].to_csv(
        out / "barangays.csv", index=False)

    print(f"{len(lgu):,} cities and municipalities ({(lgu.unit_type == 'city').sum()} cities)")
    print(f"{len(bgy):,} barangays; every one linked to a city or municipality")
    print(f"shared names: {lgu.ambiguous_name.sum()}; still ambiguous with province shown: "
          f"{lgu.ambiguous_in_context.sum()}")
    print(f"independent of a province: {lgu.independent_city.sum()}; "
          f"recently changed: {lgu.recent_change.notna().sum()}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(*sys.argv[1:])
