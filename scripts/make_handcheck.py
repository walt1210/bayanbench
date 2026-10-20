"""Build a hand-check spreadsheet from scored results of several models.

Picks a fixed-seed sample: for each question type, an equal number of answers
the scorer marked as right (correct, outdated, abstained) and as wrong, spread
across models. A person marks each verdict as agree or disagree. The share of
agreements estimates how trustworthy the automatic scoring is.

Usage:
    python scripts/make_handcheck.py results/pilot results/pilot/handcheck_pilot.xlsx
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
TYPE_ORDER = ["province", "region", "unit_type", "barangay_count", "name_barangays"]
TYPE_LABEL = {"province": "Province", "region": "Region", "unit_type": "City or municipality",
              "barangay_count": "Barangay count", "name_barangays": "Name three barangays"}


def main(results_dir: str, out_path: str, per_type: int = 10, seed: int = 11):
    frames = [pd.read_csv(f, dtype={"lgu_code": str})
              for f in sorted(glob.glob(f"{results_dir}/*_scored.csv"))]
    df = pd.concat(frames, ignore_index=True)
    bgy = pd.read_csv(PROC / "barangays.csv", dtype={"lgu_code": str})
    by_lgu = bgy.groupby("lgu_code")["name"].apply(list).to_dict()
    rng = np.random.default_rng(seed)

    # Per question type: spread rows evenly across models, prefer verdicts that need
    # judgment (correct and wrong), and include at most one easy "abstained" row.
    picks = []
    models = sorted(df.model.unique())
    for t in TYPE_ORDER:
        g = df[df.qtype == t]
        chosen = []
        for m in models:
            gm = g[g.model == m]
            for outcome in ["correct", "outdated", "wrong", "unreadable"]:
                pool = gm[gm.outcome == outcome]
                if len(pool) and len(chosen) < per_type:
                    chosen.append(pool.index[rng.integers(len(pool))])
        rest = g[~g.index.isin(chosen) & ~g.outcome.eq("abstained")]
        while len(chosen) < per_type - 1 and len(rest):
            i = rest.index[rng.integers(len(rest))]
            chosen.append(i)
            rest = rest.drop(i)
        abst = g[g.outcome.eq("abstained") & ~g.index.isin(chosen)]
        if len(chosen) < per_type and len(abst):
            chosen.append(abst.index[rng.integers(len(abst))])
        picks.append(df.loc[chosen[:per_type]])
    hc = pd.concat(picks).reset_index(drop=True)

    wb = Workbook()
    ws = wb.active
    ws.title = "Hand check"
    headers = ["#", "Model", "Question type", "Question", "Official answer", "AI answer",
               "Scorer verdict", "Scorer notes", "Your verdict", "Reason (if disagree)"]
    ws.append(headers)
    for i, r in hc.iterrows():
        official = r.gold
        if r.qtype == "name_barangays":
            names = by_lgu.get(r.lgu_code, [])
            official = f"{len(names)} official barangays: " + ", ".join(names)
        notes = ""
        if isinstance(r.get("detail"), str) and r.detail:
            try:
                d = json.loads(r.detail)
                notes = "; ".join(f"{k}: {', '.join(v)}" for k, v in d.items() if v)
            except (json.JSONDecodeError, TypeError, AttributeError):
                notes = r.detail
        answer = r.answer
        try:
            answer = json.loads(answer)
            answer = ", ".join(answer) if isinstance(answer, list) else str(answer)
        except (json.JSONDecodeError, TypeError):
            pass
        ws.append([i + 1, r.model, TYPE_LABEL[r.qtype], r.question, official, answer,
                   r.outcome, notes, "", ""])

    widths = [5, 15, 16, 48, 60, 40, 12, 40, 14, 30]
    for col, w in zip("ABCDEFGHIJ", widths):
        ws.column_dimensions[col].width = w
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="2D4A3E")
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
        row[8].fill = PatternFill("solid", fgColor="FFF4D6")
    dv = DataValidation(type="list", formula1='"agree,disagree"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"I2:I{len(hc) + 1}")
    ws.freeze_panes = "E2"

    guide = wb.create_sheet("How to check")
    for line in [
        "For each row, decide whether the scorer verdict is right for the AI answer given.",
        "You do not need to know the town. Compare the AI answer with the official answer.",
        "",
        "correct    the AI answer matches the official answer",
        "wrong      the AI answer does not match (for barangays: at least one name is not in the official list)",
        "abstained  the AI said it did not know",
        "outdated   the AI gave an answer that was true before a recent official change",
        "",
        "Mark 'disagree' when, for example, the AI gave a correct answer in a different spelling",
        "and the scorer called it wrong, or the scorer accepted a wrong answer.",
        "Spelling variants such as Sta. and Santa should be accepted.",
    ]:
        guide.append([line])
    guide.column_dimensions["A"].width = 110
    wb.save(out_path)
    print(f"{len(hc)} rows -> {out_path}")
    print(hc.groupby(["qtype", "outcome"]).size().to_string())


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
