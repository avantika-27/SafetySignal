# Boeing 737 MAX: Retrospective Case Study  
## Automation, training, and lessons for ASRS-style safety analytics

**Document purpose.** This case study situates the Boeing 737 MAX accidents (Lion Air Flight 610, October 2018; Ethiopian Airlines Flight 302, March 2019) in a **historical and organizational** context, summarizes **widely cited technical and human-factors themes** from public investigations, and explains how **this project’s** clustering, spike detection, regulation retrieval, and **benchmark evaluation** relate—without replacing official accident reports.

**Disclaimer.** Narratives below condense **public** findings (e.g., investigation summaries, regulatory actions). This is **not** an independent investigation. Operational and certification conclusions belong to state authorities and manufacturers.

---

## 1. Executive summary

The 737 MAX introduced flight-control behavior intended to improve handling characteristics at high angle of attack, implemented through the **Maneuvering Characteristics Augmentation System (MCAS)**. When **angle-of-attack (AoA)** information was erroneous or misunderstood, MCAS could command **repeated nose-down trim**, contributing—in combination with crew response, training, and procedure design—to **loss of control** in the two cited accidents.

Public investigations emphasized:

- **Sensor integrity and redundancy** (e.g., AoA disagree, reliance on single inputs where applicable).  
- **Automation transparency** (system behavior not fully aligned with prior 737 training).  
- **Procedures and training** for **runaway stabilizer** / non-normal trim scenarios.  
- **Organizational and certification processes** (design assumptions, documentation, and differences training).

For **analytics pipelines** like this capstone, the relevant parallel is **not** “predict MCAS” from ASRS text alone, but:

1. **Detect sustained thematic spikes** (e.g., trim, automation surprise, upset recovery) in **voluntary reports**.  
2. **Surface plausible regulatory citations** (14 CFR Part 91 / relevant ACs) for **awareness**, not adjudication.  
3. **Evaluate** retrieval and alerting against **weak labels** (benchmark CSV + synthetic NTSB-style narratives) using **recall, FPR, and causal-tag agreement**—as **diagnostic** metrics requiring calibration on larger, curator-verified sets.

---

## 2. Chronology (high level)

| Period | Milestone |
|--------|-----------|
| 2017 | 737 MAX entry into service |
| Oct 2018 | Lion Air JT610 (Indonesia) — public investigations and interim measures |
| Mar 2019 | Ethiopian ET302 — global grounding of the MAX fleet; intensified regulatory review |
| 2019–2021 | Return-to-service packages, training mandates, design changes (details per FAA EADs / ADs and OEM bulletins) |

Exact dates and fleet actions should be taken from **primary** FAA, EASA, and investigative authority documents when cited academically.

---

## 3. Technical themes (public record)

### 3.1 MCAS and trim authority

MCAS was designed to address handling characteristics under certain flight conditions by **commanding stabilizer trim**. Investigations described scenarios where **erroneous AoA** could lead to **undesired MCAS activation** and **nose-down commands**. Crews faced **time-compressed** situations requiring correct diagnosis (e.g., **runaway stabilizer** memory items, **cutout** use, manual trim) while maintaining aircraft control.

**Implication for text analytics:** ASRS narratives may reference **trim**, **uncommanded motion**, **automation**, **airspeed disagree**, or **manual electric trim** without naming “MCAS.” Thematic clustering and **spike detection** can still highlight **clusters** where such language **co-occurs** with phase (e.g., departure, initial climb).

### 3.2 Training and procedures

Public debate focused on **differences training**, **checklist design**, and **crew resource management** under surprise automation behavior. The **MAX retrospective** is as much about **organizational learning** as about a single subsystem.

**Implication:** A useful “causal agreement” metric between **gold tags** (e.g., `MCAS`, `trim`, `training`) and **cluster causal summaries** measures **alignment of language**, not legal causation.

---

## 4. Mapping to this project’s pipeline

| Component | Relation to MAX themes |
|-----------|-------------------------|
| **Clustering / UMAP** | Groups narratives by **shared language** (phase, anomaly strings, synopsis themes). |
| **Spike / z-score trends** | Flags **clusters** whose **normalized rates** jump—useful for **monitoring**, not accident prediction. |
| **Regulation RAG (Chroma)** | Retrieves **CFR / AC** chunks by embedding similarity to **cluster + root-cause** text—**human review** required. |
| **Analyst review UI** | Supports **approve/dismiss**, **thresholds**, **notes**, and **audit trail**—essential for any production use. |
| **Bulletins** | Exports structured summaries; must not be mistaken for regulatory determinations. |

---

## 5. Benchmark evaluation (737 / NTSB-style)

The repository includes:

- `evaluation/gold_b737_benchmark.csv` — **ASRS ACNs** (737-focused) with **weak** `y_true_alert` labels, plus **synthetic** rows inspired by public JT610/ET302 themes.  
- `evaluation/synthetic_ntsb_narratives.json` — short **paraphrased** narratives for lexical and **tag-recall** checks.  
- `evaluation/run_b737_evaluation.py` — computes **recall, precision, FPR**, and **mean causal agreement** (gold tags vs. cluster `causal_summary` / labels when joins exist).  
- `evaluation/reports/b737_eval_report.json` — machine-readable output.

### 5.1 How to read the metrics

- **Recall:** Among benchmark cases labeled “should alert,” fraction flagged by the **pipeline** (or **lexical fallback** if the report is not in `clustered_reports.csv`).  
- **FPR:** Among negatives, fraction incorrectly flagged. **Small benchmarks** are sensitive to threshold choice; treat **FPR** as **illustrative** until validated on a large, independently labeled set.  
- **Causal agreement:** Mean fraction of **gold cause tags** that appear as substrings in the assigned cluster’s narrative fields—**not** a measure of NTSB causal findings.

### 5.2 How to run

From the project root:

```bash
python3 evaluation/run_b737_evaluation.py
# or
python3 run_evaluation.py
```

Artifacts:

- `evaluation/reports/b737_eval_report.json` — per-row predictions, **metrics_all** (ASRS + synthetic), **metrics_asrs_rows_only**, and **causal_agreement_mean_cluster_vs_gold_tags** when clustered joins exist.

### 5.3 Limitations (explicit)

- ASRS is **voluntary** and **not** a census of accidents.  
- **NTSB-style** rows here are **synthetic paraphrases**, not NTSB database records.  
- **Clustering** is **descriptive**; **MAX**-specific terminology may be **absent** in ASRS even when themes overlap.

---

## 6. Recommendations for future work

1. **Curate** a larger **gold set** with investigator-labeled outcomes (or public NTSB event IDs cross-walked to narrative corpora).  
2. **Calibrate** lexical and spike **thresholds** on a **time-split** validation set.  
3. **Separate** metrics for **ASRS-joinable** rows vs. **synthetic** NTSB text.  
4. **Document** operational responses to alerts in your SMS or change log (outside this dashboard).

---

## 7. References (types of sources to cite in academic work)

- Final investigation reports and public dockets from the relevant **accident investigation authorities**.  
- **FAA** Emergency ADs / ADs and **Boeing** service-related bulletins as applicable to the period studied.  
- **ICAO** Annex 13 principles for accident investigation independence.

---

*Generated as part of the Aviation Safety Intelligence capstone tooling. Update benchmarks and metrics as data and labels mature.*
