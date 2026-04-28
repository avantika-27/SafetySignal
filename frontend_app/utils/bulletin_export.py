"""
Professional safety bulletin export: Markdown, HTML, optional PDF.
Evidence from cluster metadata + trend stats; regulations from RAG; recommended actions (templates).
"""

from __future__ import annotations

import html
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from utils.alerts_engine import build_rule_based_alerts
from utils.data_loader import (
    DataBundle,
    load_cluster_regulations_rag,
    load_data_bundle,
    load_top_spike_rows_from_notebook,
)


def build_alerts_dataframe(bundle: DataBundle) -> pd.DataFrame:
    """Match Alerts page logic: spike-aligned rows when top-spike data exists."""
    alerts = build_rule_based_alerts(bundle.monthly_cluster_trends, bundle.cluster_labels_report)
    top_spike_rows = load_top_spike_rows_from_notebook(limit=3)
    if top_spike_rows.empty:
        return alerts

    top_spike_rows["cluster"] = top_spike_rows["cluster"].astype(int)
    alerts["cluster"] = alerts["cluster"].astype(int)
    base_by_cluster = alerts.set_index("cluster").to_dict("index") if not alerts.empty else {}
    label_map = (
        bundle.cluster_summary[["cluster", "cluster_label"]]
        .drop_duplicates("cluster")
        .set_index("cluster")["cluster_label"]
        .to_dict()
    )
    root_map = (
        bundle.cluster_labels_report[["cluster", "causal_summary"]]
        .drop_duplicates("cluster")
        .set_index("cluster")["causal_summary"]
        .to_dict()
        if "causal_summary" in bundle.cluster_labels_report.columns
        else {}
    )

    rebuilt = []
    for _, s in top_spike_rows.iterrows():
        cid = int(s["cluster"])
        base = base_by_cluster.get(cid, {})
        rebuilt.append(
            {
                "cluster": cid,
                "cluster_label": label_map.get(cid, base.get("cluster_label", f"Cluster {cid}")),
                "month_key": s["month_key"],
                "incidents": int(s["count"]),
                "alert_id": f"AVIATION SAFETY ALERT #{s['month_key']}-{cid:03d}",
                "priority": base.get("priority", "HIGH"),
                "confidence": base.get("confidence", 85),
                "growth_pct": base.get("growth_pct", 0.0),
                "root_cause": root_map.get(cid, base.get("root_cause", label_map.get(cid, f"Cluster {cid}"))),
                "regulation_mapping": base.get("regulation_mapping", ""),
                "relevance_score": base.get("relevance_score", 0.0),
                "severity": base.get("severity", "high"),
                "alert_type": base.get("alert_type", "spike"),
                "message": base.get("message", ""),
            }
        )
    out = pd.DataFrame(rebuilt)
    order = {int(c): i for i, c in enumerate(top_spike_rows["cluster"].tolist())}
    out["__order"] = out["cluster"].map(order).fillna(999).astype(int)
    return out.sort_values("__order").drop(columns=["__order"])


def _cluster_row(bundle: DataBundle, cluster_id: int) -> dict[str, Any]:
    """Cluster-level fields from cluster_labels_report (by cluster id)."""
    df = bundle.cluster_labels_report
    if df.empty or "cluster" not in df.columns:
        return {}
    m = df[df["cluster"].astype(int) == int(cluster_id)]
    if m.empty:
        return {}
    return m.iloc[0].to_dict()


def _slug(s: str) -> str:
    x = re.sub(r"[^\w\s\-]", "", str(s), flags=re.UNICODE)
    x = re.sub(r"\s+", "_", x.strip())[:80]
    return x or "bulletin"


def recommended_actions(
    cluster_label: str,
    top_phase: str | None,
    keywords: str | None,
    causal_summary: str | None = None,
) -> list[str]:
    """Cluster-specific action checklist from label/phase/keywords/causal summary."""
    phase = (top_phase or "").strip().lower()
    kw = (keywords or "").lower()
    label = (cluster_label or "").lower()
    summary = (causal_summary or "").lower()
    strong_signals = " ".join([label, kw, phase])
    weak_signals = summary
    kw_terms = [t.strip() for t in (keywords or "").split(",") if t.strip()]
    kw_focus = ", ".join(kw_terms[:4]) if kw_terms else "the dominant event cues in this cluster"
    label_focus = (cluster_label or "selected cluster").strip()
    phase_focus = phase if phase else "operations"

    category_rules: list[tuple[str, list[str], list[str]]] = [
        (
            "equipment",
            ["equipment problem", "system_failure", "engine", "hydraulic", "electrical", "smoke", "fire", "gear", "flap", "trim", "pressur", "oil"],
            [
                "Run a reliability review for affected fleets and tail numbers tied to this failure signature.",
                "Audit MEL deferrals and repeat write-ups; escalate any repeat defect pattern in this cluster within 24 hours.",
                "Add a pre-dispatch engineering gate for flights with recent defects matching this cluster profile.",
            ],
        ),
        (
            "ground",
            ["taxi", "ground", "pushback", "ramp", "runway", "gate"],
            [
                "Issue hotspot briefings for specific taxi/gate conflict points seen in this cluster.",
                "Require explicit ramp-ground handoff phraseology before movement-area entry for this scenario.",
                "Conduct jumpseat/line observations on ground movement SOP adherence for this exact pattern.",
            ],
        ),
        (
            "approach",
            ["approach", "final", "unstabilized", "cfit", "terrain", "descent"],
            [
                "Tighten stabilized-approach gates and trigger mandatory go-around when this profile appears.",
                "Run targeted simulator scenarios for this cluster's phase-specific energy/path management failures.",
                "Review approach briefing quality and last-minute ATC change handling on recent events in this cluster.",
            ],
        ),
        (
            "airspace",
            ["nmac", "conflict", "tcas", "ra", "airspace", "drone", "uas", "traffic", "separation"],
            [
                "Reinforce immediate TCAS/RA compliance and conflict-call procedures for this encounter type.",
                "Publish route/airport hotspot advisories where this cluster’s conflicts are concentrated.",
                "Coordinate with ATC liaison on recurring separation-loss signatures from this cluster.",
            ],
        ),
        (
            "navigation",
            ["gps", "jamming", "spoof", "navigation", "ads-b", "fms"],
            [
                "Deploy a GPS-degraded operations checklist: raw-data cross-check, alternate nav source, dispatch notification.",
                "Require pre-approach contingency briefing for nav-loss scenarios on routes impacted by this cluster.",
                "Track nav-integrity alerts by route/time and push weekly pattern summaries to crews.",
            ],
        ),
        (
            "weather",
            ["weather", "turbulence", "wake", "crosswind", "icing", "windshear"],
            [
                "Enforce weather threat-and-error brief items specific to this cluster before descent/approach.",
                "Set conservative operational triggers (delay/divert/go-around) for this weather profile.",
                "Audit PIREP usage and ATC weather dissemination timing against cluster events.",
            ],
        ),
        (
            "comms",
            ["atc issue", "clearance", "frequency", "hearback", "readback", "controller"],
            [
                "Retrain crews on high-congestion readback/hearback discipline for this communication pattern.",
                "Introduce mandatory clearance confirmation step when call quality or frequency congestion is degraded.",
                "Capture and review ATC communication breakdown examples from this cluster in recurrent training.",
            ],
        ),
    ]

    scored: list[tuple[int, str, list[str]]] = []
    for cat_name, pats, acts in category_rules:
        score = 0
        for p in pats:
            if p in strong_signals:
                score += 3
            elif p in weak_signals:
                score += 1
        if score > 0:
            scored.append((score, cat_name, acts))
    scored.sort(key=lambda x: x[0], reverse=True)
    primary_category = scored[0][1] if scored else "general"

    actions: list[str] = [
        f"Run a 2-week focused safety review for '{label_focus}' during {phase_focus}, using cases with terms: {kw_focus}.",
        f"Select 10 recent reports from cluster '{label_focus}' and identify the first procedural breakdown point in each event sequence.",
    ]
    if scored:
        actions.extend(scored[0][2][:3])
    if len(scored) > 1 and scored[1][0] >= max(4, int(scored[0][0] * 0.6)):
        actions.append(scored[1][2][0])
    if len(actions) < 5:
        actions.append(
            f"Create a local mitigation bulletin specific to cluster '{label_focus}' and brief all relevant crews this cycle."
        )
    actions.extend(
        [
            f"Assign owner + due date per mitigation and monitor next-month trend for cluster '{label_focus}' (primary driver: {primary_category}).",
            "Log outcomes in SMS with before/after evidence (rate shift, audit result, SOP compliance trend).",
        ]
    )

    return actions[:7]


def _evidence_block(row: pd.Series, cluster_meta: dict[str, Any]) -> list[str]:
    bullets: list[str] = []
    bullets.append(
        f"Spike window: {row.get('month_key', '')} | Reported incidents in window: {int(row.get('incidents', 0))}"
    )
    if pd.notna(row.get("growth_pct")):
        bullets.append(f"Trend vs prior baseline: {float(row['growth_pct']):+.1f}% (rule-based estimate).")
    sz = cluster_meta.get("cluster_size") or cluster_meta.get("size")
    if sz is not None and not (isinstance(sz, float) and pd.isna(sz)):
        bullets.append(f"Cluster corpus size (ASRS-aligned): {int(sz)} narratives.")
    if cluster_meta.get("top_phase"):
        bullets.append(f"Dominant flight phase (structured fields): {cluster_meta['top_phase']}.")
    if cluster_meta.get("evidence_bullets"):
        excerpt = str(cluster_meta["evidence_bullets"])[:900]
        if len(str(cluster_meta["evidence_bullets"])) > 900:
            excerpt += "…"
        bullets.append(f"Structured evidence summary: {excerpt}")
    elif cluster_meta.get("description"):
        bullets.append(f"Cluster description: {str(cluster_meta['description'])[:600]}{'…' if len(str(cluster_meta['description'])) > 600 else ''}")
    if cluster_meta.get("limitations"):
        bullets.append(f"Limitations: {cluster_meta['limitations']}")
    return bullets


def render_bulletin_markdown(
    row: pd.Series,
    cluster_meta: dict[str, Any],
    regulations: list[dict[str, Any]],
    generated_at_iso: str,
) -> str:
    cid = int(row["cluster"])
    label = str(row.get("cluster_label", ""))
    root = str(row.get("root_cause", ""))
    alert_id = str(row.get("alert_id", f"ALERT-{cid}"))
    priority = str(row.get("priority", "HIGH"))
    conf = int(row.get("confidence", 0))

    lines = [
        "# Aviation Safety Bulletin",
        "",
        f"> **{alert_id}** · Priority **{priority}** · Spike scores **{conf}%**",
        "",
        "## Executive summary",
        "",
        f"This bulletin summarizes an automated spike alert for thematic cluster **{label}** (cluster id `{cid}`). "
        f"It is derived from ASRS-style narrative analytics and trend detection; it is **not** a regulatory finding.",
        "",
        "## Key metrics",
        "",
        f"- **Reporting month:** {row.get('month_key', '')}",
        f"- **Incidents (spike row):** {int(row.get('incidents', 0))}",
        f"- **Growth vs baseline:** {float(row.get('growth_pct', 0.0)):+.1f}%",
        "",
        "## Evidence",
        "",
    ]
    for b in _evidence_block(row, cluster_meta):
        lines.append(f"- {b}")
    lines.extend(["", "## Root-cause narrative (descriptive)", "", root, ""])

    lines.extend(["## Recommended actions", ""])
    for i, a in enumerate(
        recommended_actions(
            label,
            cluster_meta.get("top_phase"),
            cluster_meta.get("keywords"),
            root,
        ),
        start=1,
    ):
        lines.append(f"{i}. {a}")

    lines.extend(["", "## Applicable regulations & citations", "", "_Retrieval scores are similarity-based; verify against authoritative sources._", ""])
    if regulations:
        for r in regulations:
            cite = r.get("citation", "Source")
            rel = float(r.get("relevance", 0.0))
            doc = str(r.get("doc_type", "")).strip()
            src = str(r.get("source_file", "")).strip()
            meta = " · ".join(x for x in [doc, src] if x)
            excerpt = str(r.get("text", ""))[:650]
            lines.append(f"- **{cite}** (relevance {rel:.2f}){(' — ' + meta) if meta else ''}")
            lines.append(f"  - > {excerpt.strip()}")
    else:
        lines.append("- No regulation chunks retrieved; check FAA corpus / Chroma index.")

    lines.extend(
        [
            "",
            "## Disclaimer",
            "",
            "This bulletin is generated for operational awareness and safety promotion. It does not constitute legal advice, "
            "FAA interpretation, or a substitute for official regulations, orders, or company manuals.",
            "",
            f"---",
            f"*Generated {generated_at_iso} UTC · Aviation Safety Intelligence pipeline*",
        ]
    )
    return "\n".join(lines)


def _html_doc(
    title: str,
    inner_html: str,
) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: Georgia, 'Times New Roman', serif; color: #1a1a1a; max-width: 820px; margin: 2rem auto; padding: 0 1.25rem; line-height: 1.55; }}
    h1 {{ font-size: 1.65rem; border-bottom: 2px solid #0d47a1; padding-bottom: 0.35rem; }}
    h2 {{ font-size: 1.15rem; color: #0d47a1; margin-top: 1.75rem; }}
    .meta {{ background: #f5f7fa; border-left: 4px solid #0d47a1; padding: 0.75rem 1rem; margin: 1rem 0; font-size: 0.95rem; }}
    ul.actions {{ margin: 0.5rem 0 0 1.2rem; }}
    .cite {{ margin: 0.6rem 0; padding: 0.6rem 0.75rem; background: #fafafa; border: 1px solid #e0e0e0; border-radius: 4px; }}
    .cite small {{ color: #555; }}
    blockquote {{ margin: 0.4rem 0 0 0.5rem; color: #333; font-style: italic; border-left: 3px solid #ccc; padding-left: 0.75rem; }}
    .footer {{ margin-top: 2rem; font-size: 0.85rem; color: #666; border-top: 1px solid #ddd; padding-top: 1rem; }}
    @media print {{ body {{ margin: 0; }} }}
  </style>
</head>
<body>
{inner_html}
</body>
</html>
"""


def render_bulletin_html(
    row: pd.Series,
    cluster_meta: dict[str, Any],
    regulations: list[dict[str, Any]],
    generated_at_iso: str,
) -> str:
    cid = int(row["cluster"])
    label = str(row.get("cluster_label", ""))
    root = str(row.get("root_cause", ""))
    alert_id = str(row.get("alert_id", f"ALERT-{cid}"))
    priority = str(row.get("priority", "HIGH"))
    conf = int(row.get("confidence", 0))

    reg_html = []
    if regulations:
        for r in regulations:
            cite = html.escape(str(r.get("citation", "Source")))
            rel = float(r.get("relevance", 0.0))
            doc = html.escape(str(r.get("doc_type", "")).strip())
            src = html.escape(str(r.get("source_file", "")).strip())
            meta = " · ".join(x for x in [doc, src] if x)
            excerpt = html.escape(str(r.get("text", ""))[:650].strip())
            meta_html = f'<small>{html.escape(meta)}</small>' if meta else ""
            reg_html.append(
                f'<div class="cite"><strong>{cite}</strong> (relevance {rel:.2f}) {meta_html}'
                f'<blockquote>{excerpt}</blockquote></div>'
            )
    else:
        reg_html.append("<p><em>No regulation chunks retrieved.</em></p>")

    actions = recommended_actions(
        label,
        cluster_meta.get("top_phase"),
        cluster_meta.get("keywords"),
        root,
    )
    actions_li = "".join(f"<li>{html.escape(a)}</li>" for a in actions)

    ev_lines = _evidence_block(row, cluster_meta)
    ev_ul = "".join(f"<li>{html.escape(str(x))}</li>" for x in ev_lines)

    inner = f"""
<h1>Aviation Safety Bulletin</h1>
<div class="meta">
  <strong>{html.escape(alert_id)}</strong><br/>
  Priority <strong>{html.escape(priority)}</strong> · Spike scores <strong>{conf}%</strong><br/>
  Cluster <strong>{html.escape(label)}</strong> (id <code>{cid}</code>)
</div>

<h2>Executive summary</h2>
<p>This bulletin summarizes an automated spike alert for thematic cluster <strong>{html.escape(label)}</strong>.
It is derived from ASRS-style narrative analytics; it is <strong>not</strong> a regulatory finding.</p>

<h2>Key metrics</h2>
<ul>
  <li><strong>Reporting month:</strong> {html.escape(str(row.get('month_key', '')))}</li>
  <li><strong>Incidents (spike row):</strong> {int(row.get('incidents', 0))}</li>
  <li><strong>Growth vs baseline:</strong> {float(row.get('growth_pct', 0.0)):+.1f}%</li>
</ul>

<h2>Evidence</h2>
<ul>{ev_ul}</ul>

<h2>Root-cause narrative (descriptive)</h2>
<p>{html.escape(root)}</p>

<h2>Recommended actions</h2>
<ul class="actions">{actions_li}</ul>

<h2>Applicable regulations & citations</h2>
<p><em>Retrieval scores are similarity-based; verify against authoritative sources.</em></p>
{''.join(reg_html)}

<h2>Disclaimer</h2>
<p>This bulletin is generated for operational awareness. It does not constitute legal advice or an FAA interpretation.</p>
<div class="footer">Generated {html.escape(generated_at_iso)} UTC · Aviation Safety Intelligence pipeline</div>
"""
    return _html_doc(alert_id, inner)


def write_pdf_from_html(html_content: str, pdf_path: Path) -> bool:
    """Try WeasyPrint, then xhtml2pdf; return True if PDF written."""
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import weasyprint
        weasyprint.HTML(string=html_content).write_pdf(pdf_path)
        return True
    except Exception:
        pass
    try:
        from xhtml2pdf import pisa

        with pdf_path.open("wb") as f:
            pisa_status = pisa.CreatePDF(html_content.encode("utf-8"), dest=f)
        return not pisa_status.err
    except Exception:
        pass
    return False


def export_bulletins(
    bundle: DataBundle | None,
    out_dir: Path,
    *,
    regs_top_k: int = 6,
    write_pdf: bool = False,
) -> list[Path]:
    """
    Write one `.md` and `.html` per alert (and optionally `.pdf`).
    Returns list of written paths (primary artifacts).
    """
    if bundle is None:
        bundle = load_data_bundle()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    alerts = build_alerts_dataframe(bundle)
    if alerts.empty:
        return []

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    written: list[Path] = []

    for _, row in alerts.iterrows():
        cid = int(row["cluster"])
        meta = _cluster_row(bundle, cid)
        label = str(row.get("cluster_label", ""))
        root = str(row.get("root_cause", ""))
        regs = load_cluster_regulations_rag(cluster_label=label, root_cause=root, top_k=regs_top_k)

        aid = str(row.get("alert_id", f"alert_{cid}"))
        base = _slug(aid)

        md = render_bulletin_markdown(row, meta, regs, now)
        md_path = out_dir / f"{base}.md"
        md_path.write_text(md, encoding="utf-8")
        written.append(md_path)

        html_content = render_bulletin_html(row, meta, regs, now)
        html_path = out_dir / f"{base}.html"
        html_path.write_text(html_content, encoding="utf-8")
        written.append(html_path)

        if write_pdf:
            pdf_path = out_dir / f"{base}.pdf"
            if write_pdf_from_html(html_content, pdf_path):
                written.append(pdf_path)

    return written
