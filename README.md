# BayanBench: Does AI Know Your Town?

An audit of how well AI language models know Philippine cities, municipalities, and barangays, and whether smaller, poorer, or more remote towns receive less accurate answers.

> **Status:** Day 1 of 7. The answer key and question set are complete. Model testing, analysis, and the retrieval fix are in progress.

## Why this matters

Government offices and businesses in the Philippines are beginning to use AI chatbots. If a model answers confidently and correctly about large cities but invents facts about small towns, residents of small towns receive worse information. This project measures whether that happens, using official government data as the answer key.

## Related work

- **LocalBench** (2025) measures AI knowledge of 526 United States counties and notes that its findings may not generalize to other countries.
- **Large Language Models are Geographically Biased** (ICML 2024) studies bias at the country and regional level.
- **FilBench**, **Batayan**, and **Kalahi** evaluate Filipino language ability and cultural values.

No public evaluation of AI knowledge at the level of Philippine cities, municipalities, and barangays could be found.

## The answer key

All answers come from the **Philippine Standard Geographic Code (PSGC)** published by the Philippine Statistics Authority (2nd Quarter 2026 edition, published 30 June 2026): 18 regions, 82 provinces, 149 cities, 1,493 municipalities, and 42,010 barangays, with 2024 census population and income classification. The processed counts match the PSA national summary exactly.

## The questions

| Type | Example | Towns covered |
|---|---|---|
| Province | In which province of the Philippines is Mabalacat (a city or municipality) located? | 1,242 |
| Region | In which administrative region of the Philippines is Mabalacat located? | 1,273 |
| City or municipality | Is Mabalacat, Pampanga, Philippines, a city or a municipality? | 1,642 |
| Barangay count | How many barangays are in Mabalacat, Pampanga, Philippines? | 1,642 |
| Name barangays | Name three barangays in Mabalacat, Pampanga, Philippines. | 1,642 |

**Total: 7,441 questions.**

### Design decisions

- **No answer leaks.** Every PSGC city name contains the word "City" (for example "City of Angeles" or "Mabalacat City"), and no municipality name does. City names are therefore shown without that word.
- **Shared names.** 361 cities and municipalities share a name with another (there are ten named San Jose). Province and region questions use only unique names. Other questions show the province, which makes every town unambiguous.
- **Independent cities.** Highly urbanized cities such as Angeles are inside a province geographically but independent of it administratively, so they are not asked the province question.
- **Outdated versus invented.** Some official facts changed recently: the Negros Island Region was created in 2024 (RA 12000), Sulu moved from BARMM to Region IX in 2025 (Executive Order 91), and Maguindanao was divided in 2022. An answer that names the previous region or province is scored as **outdated**, not as a hallucination. Eight municipalities created in 2024 in the BARMM Special Geographic Area are flagged as new.

### Scoring

Every answer is scored automatically as **correct**, **abstained** (the model said it did not know), **outdated**, or **wrong** (a confident answer that does not match). Barangay names are matched after normalizing harmless differences such as "Sta." and "Santa" or "ñ" and "n", with a strict threshold so that a different place with a similar name is never accepted. The scorer has unit tests, including trap cases such as "Quezon City" not counting as the province of Quezon.

### Samples

| Sample | Towns | Questions | Purpose |
|---|---|---|---|
| Pilot | 50 | 232 | Debugging prompts and the scorer |
| Core | 400 | 1,825 | Main comparison within free API limits |
| Full | 1,642 | 7,441 | All cities and municipalities |

The core sample is stratified by island group (Luzon, Visayas, Mindanao) and population quintile. Its median population, 37,665, matches the national median of 37,549.

## Quick start

```bash
pip install -r requirements.txt

# Download the PSGC publication datafile from psa.gov.ph (Classification > PSGC)
python scripts/prepare_data.py data/raw/PSGC-2Q-2026-Publication-Datafile.xlsx
python scripts/build_questions.py
python tests/test_score.py
```

## Project structure

```
data/processed/lgus.csv          1,642 cities and municipalities with population, income class, flags
data/processed/barangays.csv     42,010 barangays linked to their city or municipality
data/processed/questions.jsonl   7,441 questions with official answers
data/processed/samples.json      pilot and core samples
src/bayanbench/normalize.py      place-name normalization
src/bayanbench/score.py          automatic scoring
scripts/prepare_data.py          PSGC Excel file to clean tables
scripts/build_questions.py       question generation and sampling
tests/test_score.py              scorer tests
```

## Limitations

- **Point in time.** The answer key reflects PSGC as of 30 June 2026. Models trained earlier may have learned facts that were correct at the time. Known recent changes are scored as outdated, but other recent changes, such as municipalities that recently became cities, are not yet identified.
- **Model access.** The comparison is limited to models available through free tiers and models small enough to run on free cloud hardware.

## Data source

Philippine Statistics Authority, Philippine Standard Geographic Code, 2nd Quarter 2026 publication datafile. The datafile lists no access constraints.

Walter Vince T. Ursua · [Portfolio](https://walt1210.github.io) · [GitHub](https://github.com/walt1210)
