"""Per-case governance report as a PDF.

Built from a case's existing governance result (no re-run) and the same
``governance_view`` the single-run screen uses, so the PDF carries the same
detail: verdict, governance decision, the full event stream, WHERE & WHY, and
the real-LLM detail. Patient name, patient ID, and member ID are masked
throughout, as in the UI; the raw model output appears only in masked form.
"""

from __future__ import annotations

import html
import io
import re
from datetime import datetime, timezone
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from governance_view import CONTRAST_ROWS, FW, build_governance_view
from governed_run import GovernedRun
from healthcare.llm_allergy_agent import redact_note
from healthcare.system import HealthcareCase

MODE_LABELS = {
    "deterministic": "Deterministic agents",
    "llm-context-loss": "Real LLM decision agent (context-loss demo)",
    "llm-pii-leak": "Real LLM allergy agent (PII-leak demo)",
}
VERDICT_COLORS = {  # (text, background), matching the app's verdict panel
    "continue": ("#176b45", "#e7f5ed"),
    "review": ("#9a6819", "#fbf3e2"),
    "block": ("#b12632", "#fff0f1"),
}
STATUS_COLORS = {"pass": "#176b45", "review": "#9a6819", "fail": "#b12632"}
FOOTER = (
    "Synthetic data only. Real-LLM results are probabilistic: the same case can come out differently on another "
    "run, model, or context window. The governance checks are deterministic, except NeMo, which is also an LLM "
    "judgement."
)
INK, MUTED, LINE = colors.HexColor("#17212b"), colors.HexColor("#667580"), colors.HexColor("#d7e0e5")


def mask_name(name: str) -> str:
    """Initial of each word, the rest starred ("Pat Example" -> "P** E******")."""
    return " ".join(part[:1] + "*" * (len(part) - 1) for part in name.split())


def mask_identifier(value: str, keep: int = 3) -> str:
    """Keep the prefix and the last ``keep`` characters; star the rest ("PAT-DEMO-001" -> "PAT-****-001")."""
    if not value:
        return "none"
    prefix, sep, rest = value.partition("-")
    if not sep:
        prefix, rest = "", value
    body = re.sub(r"[A-Za-z0-9]", "*", rest[:-keep]) + rest[-keep:] if len(rest) > keep else "*" * len(rest)
    return f"{prefix}{sep}{body}"


def _clean(text: Any) -> str:
    """Text safe for reportlab's built-in fonts and Paragraph markup."""
    value = str(text if text is not None else "").replace("→", "->").replace("✓", "yes")
    value = value.encode("cp1252", errors="replace").decode("cp1252")
    return html.escape(value, quote=False)


class _Report:
    def __init__(self, case: HealthcareCase) -> None:
        self.case = case
        styles = getSampleStyleSheet()
        self.body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=9, leading=12, textColor=INK)
        self.small = ParagraphStyle("small", parent=self.body, fontSize=8, leading=10, textColor=MUTED)
        self.cell = ParagraphStyle("cell", parent=self.body, fontSize=8, leading=10)
        self.cell_bold = ParagraphStyle("cellb", parent=self.cell, fontName="Helvetica-Bold")
        self.h1 = ParagraphStyle("h1", parent=styles["Title"], fontSize=18, leading=22, alignment=0, textColor=INK,
                                 spaceAfter=2)
        self.h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12, leading=15, textColor=INK,
                                 spaceBefore=10, spaceAfter=4)
        self.label = ParagraphStyle("label", parent=self.small, fontName="Helvetica-Bold")
        self.code = ParagraphStyle("code", parent=self.cell, fontName="Courier", fontSize=7.5, leading=9.5)
        self.flow: list[Any] = []

    def masked(self, text: Any) -> str:
        """Mask the record's identifiers (as the UI does) and make the text PDF-safe."""
        return _clean(redact_note(self.case, str(text if text is not None else "")))

    def para(self, text: str, style=None) -> None:
        self.flow.append(Paragraph(text, style or self.body))

    def heading(self, text: str) -> None:
        self.flow.append(Paragraph(_clean(text), self.h2))

    def table(self, rows, widths, header=True, styles=()) -> None:
        cells = [[cell if not isinstance(cell, str) else Paragraph(cell, self.cell) for cell in row] for row in rows]
        commands = [
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.4, LINE),
            ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]
        if header:
            commands.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f4")))
        table = Table(cells, colWidths=widths, repeatRows=1 if header else 0)
        table.setStyle(TableStyle(commands + list(styles)))
        self.flow.append(table)

    def key_values(self, pairs) -> None:
        self.table(
            [[Paragraph(_clean(k), self.label), Paragraph(v, self.cell)] for k, v in pairs],
            [1.7 * inch, 5.5 * inch], header=False,
        )


def build_case_report(
    case: HealthcareCase,
    mode: str,
    governed: GovernedRun,
    llm: Any = None,
    run_engine: str = "langgraph",
    run_engine_fallback: bool = False,
    row: int | None = None,
    domain: str = "Healthcare",
    plugin_label: str = "HEALTHCARE PLUG-IN",
) -> bytes:
    """PDF bytes for one governed case, from its already-computed governance result."""

    review, integrity = governed.review, governed.integrity
    llm_decision = llm if mode == "llm-context-loss" else None
    llm_allergy = llm if mode == "llm-pii-leak" else None
    view = build_governance_view(
        domain, review, integrity, governed.final_agent, run_engine, run_engine_fallback, llm_decision, llm_allergy,
    )
    r = _Report(case)
    text_color, background = VERDICT_COLORS[view.verdict_css]

    # ---------------------------------------------------------------- header
    r.para("Runtime Agent Governance · per-case governance report", r.small)
    r.flow.append(Paragraph(_clean(f"{domain} case {case.case_id}" + (f" (batch row {row})" if row else "")), r.h1))
    verdict = Table(
        [[Paragraph("FINAL VERDICT · ACTION GOVERNANCE GATE", ParagraphStyle(
            "vl", parent=r.label, textColor=colors.HexColor(text_color)))],
         [Paragraph(_clean(view.action_label), ParagraphStyle(
             "vv", parent=r.h1, fontSize=22, leading=26, textColor=colors.HexColor(text_color)))]],
        colWidths=[7.2 * inch],
    )
    verdict.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(background)),
        ("BOX", (0, 0), (-1, -1), 1.2, colors.HexColor(text_color)),
        ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    r.flow += [Spacer(1, 4), verdict, Spacer(1, 6)]
    r.key_values([
        ("Patient", _clean(mask_name(case.patient_name))),
        ("Patient ID", _clean(mask_identifier(case.patient_id))),
        ("Insurance member ID", _clean(mask_identifier(case.member_id, keep=2))),
        ("Age / diagnosis", _clean(f"{case.age} / {case.diagnosis}")),
        ("Medication / known allergy", _clean(f"{case.medication} / {case.known_allergy}")),
        ("Coverage", "authorized" if case.coverage_authorized else "not authorized"),
        ("Run mode", _clean(MODE_LABELS.get(mode, mode))),
        ("Pipeline engine", _clean(run_engine + (" (fallback)" if run_engine_fallback else ""))
         + _clean(f" · AGT {'live' if governed.agt_live else 'fallback'}")),
    ])

    # ---------------------------------------------------------------- governance decision
    r.heading("Governance decision")
    decided = view.decision.disposition or view.decision.finding
    r.key_values([
        ("Action Governance gate", f"<b>{_clean(view.action_label)}</b> (policy action "
                                   f"{_clean(view.action)}; matched rule "
                                   f"{_clean(review.policy_decision.get('matched_rule') or 'none')})"),
        ("Policy result", r.masked(review.policy_decision.get("reason", ""))),
        ("Agent decided · records require", _clean(f"{decided} · {review.attribution.expected_disposition}")),
        ("Why", r.masked(review.attribution.reason)),
        ("Attributed to", r.masked(view.origin)),
        ("Category", _clean(review.attribution.category or "none")),
    ])

    # ---------------------------------------------------------------- event breakdown
    r.heading("Event breakdown (the live governance run)")
    rows = [[Paragraph(h, r.label) for h in ("Layer", "Step", "Status", "Reason / detail")]]
    status_styles = []
    for index, (layer, phase, status, css, detail) in enumerate(view.events, start=1):
        rows.append([
            Paragraph("FRAMEWORK" if layer == FW else _clean(plugin_label), r.small),
            Paragraph(r.masked(phase), r.cell_bold),
            Paragraph(f'<font color="{STATUS_COLORS.get(css, "#17212b")}"><b>{_clean(status)}</b></font>', r.cell),
            Paragraph(r.masked(detail), r.cell),
        ])
        if css != "pass":
            status_styles.append(("BACKGROUND", (0, index), (-1, index), colors.HexColor("#fbf7ec")))
    r.table(rows, [1.15 * inch, 1.75 * inch, 1.0 * inch, 3.3 * inch], styles=status_styles)

    # ---------------------------------------------------------------- where & why
    r.heading("WHERE & WHY")
    if integrity.passed and view.action == "allow":
        r.para("No governance failure: the pipeline agents used consistent state and the case passed its checks.")
    elif integrity.passed:
        flagged = [c["agent"] for c in review.content_checks if c.get("rail") != "skipped" and not c.get("passed")]
        r.para(
            "Deterministic trace integrity passed: every agent consumed what its parents emitted. The governance "
            "decision comes from another check"
            + (_clean(f": the NeMo content rail flagged {', '.join(flagged)}.") if flagged else ".")
        )
    if view.producer is not None:
        r.table(
            [[Paragraph(h, r.label) for h in ("", "Agent", "Finding · version")],
             [Paragraph("Emitted", r.label), Paragraph(_clean(view.producer.agent), r.cell),
              Paragraph(r.masked(f"{view.producer.finding} · v{view.producer.finding_version}"), r.cell)],
             [Paragraph("Consumed", r.label), Paragraph(_clean(view.consumer.agent), r.cell),
              Paragraph(r.masked(f"{view.consumed_finding} · version {view.consumed_version}"), r.cell)]],
            [1.0 * inch, 2.2 * inch, 4.0 * inch],
            styles=[("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#fff0f1"))],
        )
        r.flow.append(Spacer(1, 4))
        r.para(f"<b>Trace integrity:</b> {r.masked(integrity.reason)}")
    r.para(
        f"<b>Attribution:</b> {r.masked(view.origin)} · category {_clean(review.attribution.category or 'none')}"
        + (f" · related: {_clean(', '.join(review.attribution.masked_by))}" if review.attribution.masked_by else "")
    )
    if review.attribution.branch_checks:
        r.flow.append(Spacer(1, 3))
        checks = [[Paragraph(h, r.label) for h in ("Node", "Agent", "Reported", "Records require", "Match", "Detail")]]
        for check in review.attribution.branch_checks:
            checks.append([
                Paragraph(_clean(check.node_id), r.cell), Paragraph(_clean(check.agent), r.cell),
                Paragraph(r.masked(check.reported), r.cell), Paragraph(r.masked(check.recomputed), r.cell),
                Paragraph("yes" if check.matches else "<b>no</b>", r.cell), Paragraph(r.masked(check.detail), r.cell),
            ])
        r.para("<b>Record-relative checks</b> (correctness plug-in)", r.body)
        r.table(checks, [0.45 * inch, 1.15 * inch, 1.65 * inch, 1.65 * inch, 0.45 * inch, 1.85 * inch])

    # ---------------------------------------------------------------- LLM detail
    nemo_by_agent = {c["agent"]: c for c in review.content_checks}
    if llm_decision is not None:
        r.heading("LLM detail: real context loss (decision agent)")
        r.para(_clean(
            f"{llm_decision.model} ran with num_ctx={llm_decision.num_ctx}. Ollama kept only the last "
            f"{llm_decision.kept_tokens} prompt tokens"
            + (" because the case file was longer than the window" if llm_decision.prompt_truncated else "")
            + ". The table compares what each branch emitted, whether that report was inside the kept window "
            "(measured with Ollama's own token counts), and what the agent itself said it saw."
        ))
        window = {True: "inside", False: '<font color="#b12632"><b>outside</b></font>', None: "unmeasured"}
        rows = [[Paragraph(h, r.label) for h in ("Branch", "Emitted", "In agent window", "Agent reported seeing")]]
        for agent, inside in llm_decision.in_window.items():
            step = view.steps[agent]
            rows.append([
                Paragraph(_clean(agent), r.cell), Paragraph(r.masked(f"{step.finding} · v{step.finding_version}"), r.cell),
                Paragraph(window[inside], r.cell),
                Paragraph(r.masked(llm_decision.consumed.get(agent, "not present (lost)")), r.cell),
            ])
        r.table(rows, [1.7 * inch, 1.9 * inch, 1.3 * inch, 2.3 * inch])
        r.flow.append(Spacer(1, 4))
        r.para(_clean(f"Agent decided: {llm_decision.disposition} · Records require: "
                      f"{review.attribution.expected_disposition}. Not in its context: "
                      f"{', '.join(llm_decision.not_consumed) or 'none'}."))
        nemo = nemo_by_agent.get(governed.final_agent)
        r.para(_clean(
            f"NeMo ran on the decision agent's output with the configured rail ({nemo['rail']}) and returned: "
            f"{nemo['reason']} That rail checks the agent's own output for prompt injection and personal data; it "
            "does not judge grounding, so the catch here comes from the deterministic trace check."
            if nemo else "NeMo did not run on the decision agent's output."
        ))
    if llm_allergy is not None:
        r.heading("LLM detail: real content failure (allergy-safety agent)")
        nemo = nemo_by_agent.get("allergy-safety")
        nemo_line = (
            "did not run" if nemo is None else
            f"unavailable ({nemo['reason']})" if nemo["rail"] == "unavailable" else
            f"{'BLOCKED' if not nemo['passed'] else 'ALLOWED'} by the content rail ({nemo['rail']}): {nemo['reason']}"
        )
        allergy_check = next((c for c in review.attribution.branch_checks if c.agent == "allergy-safety"), None)
        r.key_values([
            ("Agent", _clean(f"{llm_allergy.model}, "
                             + ("control prompt (adds one redaction instruction)" if llm_allergy.redacted
                                else "permissive prompt (no redaction instruction; never asks for identifiers)"))),
            ("Agent status · records require",
             _clean(f"{llm_allergy.status} · {(allergy_check.recomputed if allergy_check else '-').upper()}")),
            ("Handoff note (masked)", r.masked(llm_allergy.note)),
            ("Identifiers found verbatim", _clean(", ".join(llm_allergy.identifiers_in_note) or "none")
             + " (a display check; NeMo makes the governance call)"),
            ("NeMo content rail (allergy agent)", _clean(nemo_line)),
            ("Deterministic checks", _clean(
                f"trace check {'passed' if integrity.passed else 'flagged'}; correctness plug-in "
                f"{'passed' if review.attribution.passed else 'flagged'}. A personal-data leak is a content "
                "problem, not a provenance problem.")),
        ])
    if llm is not None:
        highlight = "context-loss" if llm_decision is not None else "pii-leak"
        r.flow.append(Spacer(1, 6))
        r.para("<b>Each failure is caught by the layer suited to it and missed by the other</b> (this run's row in bold)")
        rows = [[Paragraph(h, r.label) for h in ("Real-LLM failure", "Deterministic trace check", "NeMo content rail")]]
        for key, failure, trace, nemo_text in CONTRAST_ROWS:
            style = r.cell_bold if key == highlight else r.cell
            rows.append([Paragraph(_clean(failure), style), Paragraph(_clean(trace), style),
                         Paragraph(_clean(nemo_text), style)])
        r.table(rows, [2.4 * inch, 2.4 * inch, 2.4 * inch])
        r.heading("Appendix: raw model output (identifiers masked)")
        r.para(r.masked(llm.raw_output).replace("\n", "<br/>"), r.code)
        r.para("The full prompt the agent received is shown, unmasked, in the app's raw-output expander.", r.small)

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    def footer(canvas, doc) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(MUTED)
        text = canvas.beginText(0.6 * inch, 0.45 * inch)
        for line in _wrap(FOOTER, 175):
            text.textLine(line)
        canvas.drawText(text)
        canvas.drawRightString(letter[0] - 0.6 * inch, 0.3 * inch, f"Generated {generated} · page {doc.page}")
        canvas.restoreState()

    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer, pagesize=letter, leftMargin=0.6 * inch, rightMargin=0.6 * inch, topMargin=0.55 * inch,
        bottomMargin=0.85 * inch, title=f"Governance report {case.case_id}", author="Runtime Agent Governance demo",
        pageCompression=0,
    )
    document.build(r.flow, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()


def _wrap(text: str, width: int) -> list[str]:
    words, lines, line = text.split(), [], ""
    for word in words:
        if len(line) + len(word) + 1 > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return lines + ([line] if line else [])


__all__ = ["MODE_LABELS", "build_case_report", "mask_identifier", "mask_name"]
