"""Day 6: figures for the README and website (PNG, light background).

Reads the scored core results (Day 3) and the results with PSGC records (Day 5).
Writes figures/*.png and figures/figure_data/*.csv (the numbers behind each figure).

Map boundaries: faeldon/philippines-json-maps (MIT), 2023 PSGC provinces,
stored in data/external/ph_provinces_2023/.
"""
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import PatchCollection  # noqa: E402
from matplotlib.patches import Patch, Polygon  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
DATA = FIG / "figure_data"

# Reference palette (validated: categorical adjacent pairs pass in light mode;
# aqua and yellow sit below 3:1 contrast, so every series also has a direct label).
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
BLUE_RAMP = ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"]  # steps 100/250/400/500/650

MODELS = ["llama3.1-8b", "sealion-v3-8b", "gemma4-12b", "qwen3.5-9b"]
LABEL = {"llama3.1-8b": "Llama 3.1 8B", "sealion-v3-8b": "SEA-LION v3 8B",
         "gemma4-12b": "Gemma 4 12B", "qwen3.5-9b": "Qwen 3.5 9B"}
COLOR = dict(zip(MODELS, SERIES))
# Gemini was run without records only (free-tier limits), 25 questions per request.
GEMINI = "gemini-3.6-flash-batch25"
LABEL[GEMINI] = "Gemini 3.6 Flash"
COLOR[GEMINI] = SERIES[4]
QUINTILE_LABEL = ["Smallest\n(under 20k)", "20k to 31k", "31k to 45k", "45k to 76k", "Largest\n(over 76k)"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.grid": False,
    "axes.spines.top": False, "axes.spines.right": False, "lines.solid_capstyle": "round",
})


def title(fig, text, sub):
    fig.text(0.02, 0.975, text, fontsize=14, fontweight="bold", color=INK, va="top")
    fig.text(0.02, 0.915, sub, fontsize=10, color=INK2, va="top")


def source(fig, text):
    fig.text(0.02, 0.015, text, fontsize=8, color=MUTED, va="bottom")


def hgrid(ax):
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def load(kind):
    """kind: 'core' (no records) or 'core_rag' (with PSGC records)."""
    lgu = pd.read_csv(ROOT / "data/processed/lgus.csv", dtype={"code": str})
    lgu["pop_quintile"] = pd.qcut(lgu.pop2024.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    frames = []
    models = MODELS + ([GEMINI] if kind == "core" and
                       (ROOT / f"results/core/results_{GEMINI}_core_scored.csv").exists() else [])
    for m in models:
        d = pd.read_csv(ROOT / f"results/{kind}/results_{m}_{kind}_scored.csv", dtype={"lgu_code": str})
        d["model"] = m
        frames.append(d)
    df = pd.concat(frames, ignore_index=True).merge(
        lgu[["code", "pop_quintile", "island_group", "province", "region_code"]],
        left_on="lgu_code", right_on="code", how="left")
    df["knows_location"] = df.outcome.isin(["correct", "outdated"]).astype(int)
    return df


def fig_small_towns(before):
    loc = before[before.qtype.isin(["province", "region"]) & before.recent_change.isna()]
    t = loc.groupby(["model", "pop_quintile"]).knows_location.mean().mul(100).unstack()
    t.round(1).to_csv(DATA / "fig1_location_by_population_quintile.csv")
    fig, ax = plt.subplots(figsize=(9, 5.6))
    fig.subplots_adjust(left=0.08, right=0.72, top=0.80, bottom=0.2)
    x = np.arange(1, 6)
    ends = []
    for m in [m for m in MODELS + [GEMINI] if m in t.index]:
        y = t.loc[m].values
        ax.plot(x, y, color=COLOR[m], linewidth=2, solid_joinstyle="round")
        ax.scatter(x, y, s=42, color=COLOR[m], edgecolor=SURFACE, linewidth=2, zorder=3)
        ends.append((y[-1], m, y[0]))
    # direct labels at the right end, nudged apart only if they would overlap
    ends.sort()
    placed = []
    for y_end, m, y0 in ends:
        yy = max(y_end, placed[-1] + 6) if placed else y_end
        placed.append(yy)
        ax.annotate(f"{LABEL[m]}  {y0:.0f}% → {y_end:.0f}%", xy=(5, y_end), xytext=(5.18, yy),
                    va="center", fontsize=9.5, color=INK,
                    arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.8) if abs(yy - y_end) > 1 else None,
                    annotation_clip=False)
    ax.set_xticks(x, QUINTILE_LABEL)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20), [f"{v}%" for v in range(0, 101, 20)])
    ax.set_xlim(0.8, 5.15)
    ax.set_xlabel("Town population (2024), in five equal groups", labelpad=8)
    hgrid(ax)
    title(fig, "Small open models know where big towns are, not small ones",
          "Share of province and region questions answered correctly, by town size (no lookup)")
    source(fig, "BayanBench core sample: 400 cities and municipalities. Answer key: PSA PSGC, 2Q 2026. "
                "Recently changed places excluded.")
    fig.savefig(FIG / "fig1_small_towns.png", dpi=200)
    plt.close(fig)


def fig_regression():
    reg = pd.read_csv(ROOT / "results/analysis/location_regression.csv")
    keep = {"population (+1 SD of log10)": "Town 2.5 times larger",
            "city (vs municipality)": "City (vs municipality)",
            "income class (+1 step poorer)": "One income class poorer",
            "Visayas (vs Luzon)": "Visayas (vs Luzon)",
            "Mindanao (vs Luzon)": "Mindanao (vs Luzon)"}
    r = reg[reg.term.isin(keep)].copy()
    r["label"] = r.term.map(keep)
    r = r.set_index("label").loc[list(keep.values())].reset_index()
    r[["label", "odds_ratio", "or_low", "or_high", "p"]].round(4).to_csv(DATA / "fig2_regression.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    fig.subplots_adjust(left=0.27, right=0.95, top=0.76, bottom=0.2)
    y = np.arange(len(r))[::-1]
    ax.axvline(1, color=AXIS, linewidth=1)
    sig = r.p < 0.05
    for yi, (_, row), s in zip(y, r.iterrows(), sig):
        c = SERIES[0] if s else MUTED
        ax.plot([row.or_low, row.or_high], [yi, yi], color=c, linewidth=2)
        ax.scatter(row.odds_ratio, yi, s=60, color=c, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(row.or_high * 1.06, yi, f"{row.odds_ratio:.2f}" + ("" if s else "  (not significant)"),
                va="center", fontsize=9, color=INK2)
    ax.set_xscale("log")
    ticks = [0.25, 0.5, 1, 2, 4, 8]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(0.25, 12)
    ax.set_yticks(y, r.label)
    ax.tick_params(axis="y", length=0, labelcolor=INK, labelsize=10)
    ax.spines["left"].set_visible(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Odds of a correct answer (1 = no difference; log scale)", labelpad=8)
    ax.text(0.95, -0.85, "← less often correct", ha="right", va="center", fontsize=8.5, color=MUTED)
    ax.text(1.05, -0.85, "more often correct →", ha="left", va="center", fontsize=8.5, color=MUTED)
    ax.set_ylim(-1.3, len(r) - 0.5)
    title(fig, "Town size and Mindanao matter; income does not, once size is known",
          "Logistic regression on 2,228 province and region answers (283 towns), all factors at once")
    source(fig, "Also controls for urban share, question type and model. Lines: 95% confidence intervals, "
                "standard errors clustered by town. Blue: p < 0.05.")
    fig.savefig(FIG / "fig2_what_predicts_errors.png", dpi=200)
    plt.close(fig)


def fig_before_after(before, after):
    def rate(d, models):
        out = {}
        for m in models:
            s = d[d.model == m]
            out[m] = {
                "Province": (s[s.qtype == "province"].outcome == "correct").mean() * 100,
                "Region": (s[s.qtype == "region"].outcome == "correct").mean() * 100,
                "Exact barangay count": (s[s.qtype == "barangay_count"].outcome == "correct").mean() * 100,
                "Three real barangays": (s[s.qtype == "name_barangays"].outcome == "correct").mean() * 100,
            }
        return pd.DataFrame(out).T
    has_g = GEMINI in set(before.model)
    rows = MODELS + ([GEMINI] if has_g else [])
    b = rate(before, rows)
    a = rate(after, MODELS)
    pd.concat({"no_lookup": b, "with_psgc_records": a}, axis=1).round(1).to_csv(DATA / "fig3_before_after.csv")
    cols = list(b.columns)
    fig, axes = plt.subplots(1, 4, figsize=(11, 5.0), sharey=True)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.74, bottom=0.17, wspace=0.12)
    y = np.arange(len(rows))[::-1].astype(float)
    if has_g:
        y[:-1] += 0.4  # small gap above the Gemini row
    for ax, c in zip(axes, cols):
        if has_g:
            ax.axhline(0.7, color=GRID, linewidth=0.8)
        for yi, m in zip(y, rows):
            if m in a.index:
                ax.plot([b.loc[m, c], a.loc[m, c]], [yi, yi], color=GRID, linewidth=3, zorder=1)
                ax.scatter(a.loc[m, c], yi, s=55, color=SERIES[0], edgecolor=SURFACE, linewidth=2, zorder=3)
            ax.scatter(b.loc[m, c], yi, s=55, color=MUTED, edgecolor=SURFACE, linewidth=2, zorder=3)
            ax.text(b.loc[m, c], yi + 0.3, f"{b.loc[m, c]:.0f}", ha="center", fontsize=8, color=INK2)
        ax.set_title(c, fontsize=10, color=INK, loc="left", pad=8)
        ax.set_xlim(-5, 108)
        ax.set_xticks([0, 50, 100], ["0%", "50%", "100%"])
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y, [LABEL[m] + ("*" if m == GEMINI else "") for m in rows])
    axes[0].tick_params(axis="y", labelcolor=INK, labelsize=10)
    axes[0].set_ylim(-0.6, y.max() + 0.7)
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=8, color=MUTED, label="No lookup"),
               plt.Line2D([], [], marker="o", linestyle="", markersize=8, color=SERIES[0],
                          label="With official PSGC records")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.015, 0.875), ncol=2, frameon=False,
               fontsize=9.5)
    title(fig, "Giving the model the official record fixes almost every error",
          "Percent correct, without and with official PSGC records in the prompt")
    source(fig, "Core sample, 1,825 questions per model. Records are found from the question text alone (PSA PSGC, "
                "2Q 2026). *Gemini, a much larger hosted model, was run without records only.")
    fig.savefig(FIG / "fig3_lookup_fix.png", dpi=200)
    plt.close(fig)


def fig_barangays(before):
    nb = before[before.qtype == "name_barangays"]
    g = nb.groupby("model")[["n_named", "n_real", "n_real_specific", "n_generic_wrong",
                             "n_elsewhere", "n_nowhere"]].sum()
    parts = pd.DataFrame({
        "Real, in that town": g.n_real_specific,
        "Real, but a generic name (Poblacion, Barangay 1)": g.n_real - g.n_real_specific,
        "Generic name that is not real there": g.n_generic_wrong,
        "Real, but in another town": g.n_elsewhere,
        "Made up (not found anywhere)": g.n_nowhere,
    })
    rows = MODELS + ([GEMINI] if GEMINI in g.index else [])
    pct = parts.div(g.n_named, axis=0).mul(100).loc[rows]
    pct.round(1).assign(names_given=g.n_named.loc[rows]).to_csv(DATA / "fig4_barangay_names.csv")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    fig.subplots_adjust(left=0.15, right=0.97, top=0.72, bottom=0.12)
    y = np.arange(len(rows))[::-1].astype(float)
    if GEMINI in rows:
        y[:-1] += 0.35
    left = np.zeros(len(rows))
    for col, color in zip(pct.columns, SERIES):
        vals = pct[col].values
        ax.barh(y, vals, left=left, height=0.55, color=color, edgecolor=SURFACE, linewidth=2, label=col)
        for yi, l, v in zip(y, left, vals):
            if v >= 8:
                ax.text(l + v / 2, yi, f"{v:.0f}%", ha="center", va="center", fontsize=8.5,
                        color="white" if color in (SERIES[0],) else INK)
        left += vals
    for yi, m in zip(y, rows):
        ax.text(101, yi, f"{int(g.n_named[m]):,} names", va="center", fontsize=8.5, color=MUTED)
    ax.set_yticks(y, [LABEL[m] for m in rows])
    ax.tick_params(axis="y", length=0, labelcolor=INK, labelsize=10)
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.spines["left"].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.13, 1.27), ncol=3, frameon=False, fontsize=8.5,
              handlelength=1.2, columnspacing=1.2)
    title(fig, "Asked to name three barangays, small models mostly guess",
          "What each barangay name given by the model turned out to be (no lookup)")
    source(fig, "Every name the models gave for 400 towns, checked against the 42,010 barangays in the "
                "PSA PSGC, 2Q 2026 (no lookup).")
    fig.savefig(FIG / "fig4_barangay_names.png", dpi=200)
    plt.close(fig)


REGION_SHORT = {"01": "Ilocos", "02": "Cagayan Valley", "03": "Central Luzon", "04": "CALABARZON",
                "05": "Bicol", "06": "Western Visayas", "07": "Central Visayas", "08": "Eastern Visayas",
                "09": "Zamboanga Peninsula", "10": "Northern Mindanao", "11": "Davao", "12": "SOCCSKSARGEN",
                "13": "NCR", "14": "CAR", "16": "Caraga", "17": "MIMAROPA", "18": "Negros Island", "19": "BARMM"}


def _region_shapes():
    """Province polygons (2023 boundaries) tagged with their CURRENT region (PSGC 2Q 2026).

    Re-tagging by today's province list puts Negros Occidental, Negros Oriental and
    Siquijor in the Negros Island Region and Sulu in Region IX.
    """
    lgu = pd.read_csv(ROOT / "data/processed/lgus.csv", dtype={"code": str, "region_code": str})
    prov_region = dict(zip(lgu.province, lgu.region_code.str.zfill(2)))
    shapes = []
    for f in sorted(glob.glob(str(ROOT / "data/external/ph_provinces_2023/*.json"))):
        for ft in json.load(open(f, encoding="utf-8"))["features"]:
            name = ft["properties"]["adm2_en"] or ""
            if name.startswith("NCR"):
                rc = "13"
            elif name.startswith("City of Isabela"):
                rc = "09"
            elif name in prov_region:
                rc = prov_region[name]
            else:  # BARMM Special Geographic Area (no name in the 2023 file)
                rc = str(ft["properties"]["adm1_psgc"]).zfill(10)[:2]
            g = ft["geometry"]
            polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
            for poly in polys:
                shapes.append((rc, np.array(poly[0])))
    return shapes


def fig_map(before, after):
    def by_region(d):
        loc = d[d.qtype.isin(["province", "region"]) & d.model.isin(MODELS)].copy()
        loc["rc"] = loc.region_code.astype(str).str.zfill(2)
        s = loc.groupby("rc").agg(rate=("knows_location", "mean"), towns=("lgu_code", "nunique"),
                                  answers=("qid", "size"))
        s["rate"] *= 100
        s["region"] = s.index.map(REGION_SHORT)
        return s
    b, a = by_region(before), by_region(after)
    pd.concat({"no_lookup": b, "with_psgc_records": a}, axis=1).round(1).to_csv(DATA / "fig5_map_by_region.csv")
    shapes = _region_shapes()
    bins = [0, 40, 50, 60, 70, 100.01]
    labels = ["Under 40%", "40 to 50%", "50 to 60%", "60 to 70%", "70% or more"]
    min_towns = 5
    fig = plt.figure(figsize=(10, 7.6))
    ax = fig.add_axes([0.0, 0.1, 0.5, 0.72])
    bx = fig.add_axes([0.66, 0.16, 0.31, 0.64])
    patches, colors = [], []
    for rc, ring in shapes:
        if rc in b.index and b.loc[rc, "towns"] >= min_towns:
            colors.append(BLUE_RAMP[min(np.digitize(b.loc[rc, "rate"], bins) - 1, 4)])
        else:
            colors.append("#f0efec")
        patches.append(Polygon(ring, closed=True))
    ax.add_collection(PatchCollection(patches, facecolor=colors, edgecolor=colors, linewidth=0.4))
    ax.set_xlim(116.8, 127.0)
    ax.set_ylim(4.4, 21.2)
    ax.set_aspect(1 / np.cos(np.radians(12.5)))
    ax.axis("off")
    # ranked bars: the exact number for every region, same colors as the map
    r = b[b.towns >= min_towns].sort_values("rate")
    y = np.arange(len(r))
    bx.barh(y, r.rate, height=0.62, color=[BLUE_RAMP[min(np.digitize(v, bins) - 1, 4)] for v in r.rate])
    for yi, v, ra in zip(y, r.rate, a.loc[r.index, "rate"]):
        bx.text(v + 1.5, yi, f"{v:.0f}%", va="center", fontsize=8.5, color=INK2)
    bx.set_yticks(y, r.region)
    bx.tick_params(axis="y", length=0, labelcolor=INK, labelsize=9)
    bx.set_xlim(0, 100)
    bx.set_xticks([0, 50, 100], ["0%", "50%", "100%"])
    bx.spines["left"].set_visible(False)
    bx.xaxis.grid(True, color=GRID, linewidth=0.8)
    bx.set_axisbelow(True)
    lo, hi = a.loc[r.index, "rate"].min(), a.loc[r.index, "rate"].max()
    legend = [Patch(facecolor=c, edgecolor="none", label=l) for c, l in zip(BLUE_RAMP, labels)]
    legend.append(Patch(facecolor="#f0efec", edgecolor=AXIS, linewidth=0.5,
                        label=f"Fewer than {min_towns} towns asked (NCR)"))
    fig.legend(handles=legend, loc="lower left", bbox_to_anchor=(0.02, 0.05), ncol=6, frameon=False,
               fontsize=8.5, handlelength=1.2, columnspacing=1.0)
    title(fig, "Where the models know their way around",
          f"Province and region questions answered correctly, by region, no lookup (average of 4 open models).\n"
          f"With official PSGC records, every region scores {lo:.0f}% to {hi:.0f}%.")
    source(fig, "BayanBench core sample, current regions (PSGC 2Q 2026); old official names count as correct. "
                "Boundaries: faeldon/philippines-json-maps (2023).")
    fig.savefig(FIG / "fig5_map.png", dpi=200)
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)
    before, after = load("core"), load("core_rag")
    fig_small_towns(before)
    fig_regression()
    fig_before_after(before, after)
    fig_barangays(before)
    fig_map(before, after)
    print("wrote", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    sys.exit(main())
