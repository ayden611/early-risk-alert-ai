#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re
from datetime import datetime, timezone

ROOT = Path(".")
WEB = ROOT / "era" / "web"
DOCS = ROOT / "docs" / "validation"

START = "<!-- ERA_COMMAND_CENTER_EXPLAINABILITY_POLISH_V1_START -->"
END = "<!-- ERA_COMMAND_CENTER_EXPLAINABILITY_POLISH_V1_END -->"

CANDIDATE_NAMES = [
    "command_center.html",
    "command-center.html",
    "command.html",
    "dashboard.html",
    "clinical_command_center.html",
]

def score_candidate(path: Path) -> int:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return -1

    name = path.name.lower()
    score = 0

    if name in CANDIDATE_NAMES:
        score += 50
    if "command" in name:
        score += 20
    if "center" in name:
        score += 20
    if "dashboard" in name:
        score += 12

    markers = [
        "Command Center",
        "command-center",
        "patient",
        "risk",
        "SpO2",
        "spo2",
        "heart_rate",
        "clinical_priority",
        "priority",
        "alert",
    ]

    for marker in markers:
        if marker in text:
            score += 5

    if "</body>" in text.lower():
        score += 15

    return score


def find_command_center_file() -> Path:
    if not WEB.exists():
        raise SystemExit("ERROR: era/web directory not found.")

    candidates = list(WEB.glob("*.html"))
    if not candidates:
        raise SystemExit("ERROR: No HTML files found in era/web.")

    ranked = sorted(((score_candidate(p), p) for p in candidates), reverse=True, key=lambda x: x[0])

    best_score, best_path = ranked[0]

    if best_score < 25:
        print("Could not confidently detect command center HTML.")
        print("Candidates:")
        for score, path in ranked:
            print(score, path)
        raise SystemExit("ERROR: No likely command center HTML file found.")

    print("Selected Command Center HTML:", best_path, "score=", best_score)
    return best_path


SNIPPET = r'''
<!-- ERA_COMMAND_CENTER_EXPLAINABILITY_POLISH_V1_START -->
<style id="era-command-center-explainability-polish-css">
  :root {
    --era-xai-bg: rgba(5, 10, 20, 0.72);
    --era-xai-panel: rgba(12, 22, 38, 0.88);
    --era-xai-line: rgba(145, 190, 255, 0.22);
    --era-xai-text: #f7fbff;
    --era-xai-muted: #b8c6d8;
    --era-xai-green: #9cffc7;
    --era-xai-blue: #91bdff;
    --era-xai-gold: #ffd78a;
    --era-xai-orange: #ffb978;
    --era-xai-red: #ff8fa3;
  }

  .era-xai-module,
  .era-xai-strip {
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }

  .era-xai-module {
    width: min(1180px, calc(100% - 28px));
    margin: 16px auto;
    border: 1px solid var(--era-xai-line);
    border-radius: 24px;
    background:
      radial-gradient(circle at 10% 0%, rgba(145, 189, 255, 0.14), transparent 32rem),
      linear-gradient(145deg, rgba(14, 24, 42, 0.96), rgba(6, 11, 21, 0.94));
    color: var(--era-xai-text);
    box-shadow: 0 22px 65px rgba(0,0,0,0.35);
    padding: 18px;
    position: relative;
    z-index: 3;
  }

  .era-xai-module-top {
    display: flex;
    justify-content: space-between;
    gap: 16px;
    align-items: flex-start;
    flex-wrap: wrap;
  }

  .era-xai-kicker {
    color: var(--era-xai-green);
    letter-spacing: 0.14em;
    text-transform: uppercase;
    font-size: 11px;
    font-weight: 900;
    margin-bottom: 5px;
  }

  .era-xai-title {
    margin: 0;
    font-size: clamp(18px, 2.4vw, 28px);
    letter-spacing: -0.04em;
    line-height: 1.08;
    font-weight: 950;
  }

  .era-xai-copy {
    color: var(--era-xai-muted);
    margin: 7px 0 0;
    font-size: 13px;
    max-width: 760px;
  }

  .era-xai-proof-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(110px, 1fr));
    gap: 10px;
    margin-top: 15px;
  }

  .era-xai-proof {
    border: 1px solid var(--era-xai-line);
    border-radius: 16px;
    background: rgba(255,255,255,0.055);
    padding: 12px;
    min-height: 84px;
  }

  .era-xai-proof span {
    display: block;
    color: var(--era-xai-muted);
    font-size: 10px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 850;
  }

  .era-xai-proof b {
    display: block;
    font-size: 25px;
    line-height: 1;
    margin-top: 9px;
    letter-spacing: -0.05em;
  }

  .era-xai-proof small {
    display: block;
    color: var(--era-xai-muted);
    margin-top: 6px;
    font-size: 11px;
  }

  .era-xai-actions {
    display: flex;
    gap: 9px;
    flex-wrap: wrap;
    margin-top: 12px;
  }

  .era-xai-actions a {
    color: var(--era-xai-text);
    text-decoration: none;
    border: 1px solid var(--era-xai-line);
    border-radius: 999px;
    padding: 8px 11px;
    background: rgba(255,255,255,0.06);
    font-size: 12px;
    font-weight: 800;
  }

  .era-xai-note {
    margin-top: 12px;
    border: 1px solid rgba(255, 215, 138, 0.35);
    background: rgba(255, 215, 138, 0.095);
    color: #ffe2a5;
    border-radius: 14px;
    padding: 10px 12px;
    font-size: 12px;
    font-weight: 750;
  }

  .era-xai-strip {
    border: 1px solid var(--era-xai-line);
    border-radius: 18px;
    background: linear-gradient(145deg, rgba(13, 24, 42, 0.92), rgba(7, 12, 22, 0.86));
    margin-top: 12px;
    padding: 12px;
    color: var(--era-xai-text);
  }

  .era-xai-strip-title {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    margin-bottom: 9px;
  }

  .era-xai-strip-title strong {
    font-size: 12px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--era-xai-green);
  }

  .era-xai-strip-title span {
    color: var(--era-xai-muted);
    font-size: 11px;
  }

  .era-xai-badges {
    display: flex;
    gap: 7px;
    flex-wrap: wrap;
  }

  .era-xai-badge {
    border: 1px solid var(--era-xai-line);
    border-radius: 999px;
    padding: 6px 9px;
    background: rgba(255,255,255,0.06);
    font-size: 11px;
    font-weight: 850;
    line-height: 1;
  }

  .era-xai-badge b {
    font-weight: 950;
  }

  .era-xai-critical {
    border-color: rgba(255, 143, 163, 0.55);
    background: rgba(255, 143, 163, 0.14);
    color: #ffd4dc;
  }

  .era-xai-elevated {
    border-color: rgba(255, 185, 120, 0.55);
    background: rgba(255, 185, 120, 0.13);
    color: #ffe2c4;
  }

  .era-xai-watch {
    border-color: rgba(255, 215, 138, 0.55);
    background: rgba(255, 215, 138, 0.12);
    color: #ffe7ad;
  }

  .era-xai-low {
    border-color: rgba(156, 255, 199, 0.45);
    background: rgba(156, 255, 199, 0.10);
    color: #cfffe3;
  }

  .era-xai-subcopy {
    color: var(--era-xai-muted);
    margin-top: 9px;
    font-size: 11px;
  }

  @media (max-width: 900px) {
    .era-xai-proof-grid {
      grid-template-columns: 1fr 1fr;
    }
  }

  @media (max-width: 560px) {
    .era-xai-proof-grid {
      grid-template-columns: 1fr;
    }
  }
</style>

<script id="era-command-center-explainability-polish-js">
(function () {
  if (window.__ERA_COMMAND_CENTER_EXPLAINABILITY_POLISH_V1__) return;
  window.__ERA_COMMAND_CENTER_EXPLAINABILITY_POLISH_V1__ = true;

  const ERA_EVIDENCE = {
    alertReduction: "94.3%",
    medianLead: "4.0h",
    fpr: "4.2%",
    threshold: "t=6.0",
    highAcuity: "t=4.0",
    highAcuityReduction: "80.3%"
  };

  function textOf(el) {
    return (el && (el.innerText || el.textContent) || "").replace(/\s+/g, " ").trim();
  }

  function parseNumber(value) {
    if (value === null || value === undefined) return null;
    const m = String(value).match(/-?\d+(\.\d+)?/);
    return m ? Number(m[0]) : null;
  }

  function extractScore(card) {
    const attrs = [
      "data-risk-score",
      "data-score",
      "data-risk",
      "aria-label"
    ];

    for (const attr of attrs) {
      const v = card.getAttribute && card.getAttribute(attr);
      const n = parseNumber(v);
      if (Number.isFinite(n)) return normalizeScore(n);
    }

    const text = textOf(card);
    const patterns = [
      /risk\s*score\s*[:\-]?\s*(\d+(\.\d+)?)/i,
      /risk\s*[:\-]?\s*(\d+(\.\d+)?)/i,
      /score\s*[:\-]?\s*(\d+(\.\d+)?)/i,
      /priority\s*score\s*[:\-]?\s*(\d+(\.\d+)?)/i
    ];

    for (const p of patterns) {
      const m = text.match(p);
      if (m) return normalizeScore(Number(m[1]));
    }

    return null;
  }

  function normalizeScore(score) {
    if (!Number.isFinite(score)) return null;
    if (score > 10 && score <= 100) return Math.round((score / 10) * 10) / 10;
    return Math.round(score * 10) / 10;
  }

  function tierFromScore(score, text) {
    const t = (text || "").toLowerCase();

    if (/\bcritical\b/.test(t)) return "Critical";
    if (/\belevated\b|\bhigh\b/.test(t)) return "Elevated";
    if (/\bwatch\b|\bmonitor\b/.test(t)) return "Watch";

    if (score === null || score === undefined) return "Review";
    if (score >= 6) return "Critical";
    if (score >= 5) return "Elevated";
    if (score >= 4) return "Watch";
    return "Low";
  }

  function tierClass(tier) {
    const t = String(tier || "").toLowerCase();
    if (t.includes("critical")) return "era-xai-critical";
    if (t.includes("elevated") || t.includes("high")) return "era-xai-elevated";
    if (t.includes("watch") || t.includes("review")) return "era-xai-watch";
    return "era-xai-low";
  }

  function inferDriver(card) {
    const text = textOf(card).toLowerCase();

    if (/spo2|sp02|oxygen|desat|saturation/.test(text)) {
      if (/declin|drop|down|worsen/.test(text)) return "SpO₂ decline";
      if (/below|low|borderline|<\s*90/.test(text)) return "Borderline SpO₂";
      return "SpO₂ / oxygenation";
    }

    if (/respiratory|respiration|rr\b|breath/.test(text)) {
      if (/elevat|high|tachyp/.test(text)) return "RR elevation";
      return "Respiratory-rate pattern";
    }

    if (/bp|blood pressure|systolic|diastolic|hypertension|hypotension/.test(text)) {
      if (/trend|worsen|declin|drop|rise/.test(text)) return "BP trend concern";
      return "BP instability";
    }

    if (/heart rate|\bhr\b|pulse|tachy|brady/.test(text)) {
      return "HR instability";
    }

    if (/temperature|temp|fever/.test(text)) {
      return "Temperature context";
    }

    if (/multi|composite|combined|compound/.test(text)) {
      return "Composite multi-signal pattern";
    }

    return "Composite review pattern";
  }

  function inferTrend(card, score) {
    const text = textOf(card).toLowerCase();

    if (/worsening|worsen|rising|increasing|declining|deteriorat/.test(text)) return "Worsening";
    if (/improving|improv|recover|decreasing/.test(text)) return "Improving";
    if (/stable/.test(text)) return "Stable";

    if (score !== null && score >= 6) return "Elevated / needs review";
    if (score !== null && score >= 4) return "Watch";
    return "Stable / low review priority";
  }

  function scoreSort(cards) {
    return cards
      .map((card, index) => ({ card, index, score: extractScore(card) }))
      .sort((a, b) => {
        const as = a.score === null ? -Infinity : a.score;
        const bs = b.score === null ? -Infinity : b.score;
        if (bs !== as) return bs - as;
        return a.index - b.index;
      });
  }

  function likelyPatientCard(el) {
    if (!el || el.closest(".era-xai-module")) return false;

    const text = textOf(el);
    if (text.length < 25) return false;

    const lower = text.toLowerCase();

    const patientSignal =
      /patient|case|bed|unit|risk|spo2|sp02|heart rate|\bhr\b|blood pressure|\bbp\b|respiratory|\brr\b|alert|priority/.test(lower);

    if (!patientSignal) return false;

    const tag = el.tagName ? el.tagName.toLowerCase() : "";
    if (["body", "html", "main", "section"].includes(tag)) return false;

    return true;
  }

  function getPatientCards() {
    const selectors = [
      "[data-patient-id]",
      "[data-case-id]",
      "[data-risk-score]",
      ".patient-card",
      ".patient-row",
      ".patient",
      ".risk-card",
      ".alert-card",
      ".clinical-card",
      ".queue-card",
      ".card"
    ];

    const found = [];
    const seen = new Set();

    for (const selector of selectors) {
      document.querySelectorAll(selector).forEach((el) => {
        if (seen.has(el)) return;
        if (!likelyPatientCard(el)) return;
        seen.add(el);
        found.push(el);
      });
    }

    return found.slice(0, 40);
  }

  function createBadge(label, value, cls) {
    const badge = document.createElement("span");
    badge.className = "era-xai-badge " + (cls || "");
    badge.innerHTML = `${label}: <b>${value}</b>`;
    return badge;
  }

  function annotateCard(card, rank) {
    if (!card || card.dataset.eraXaiPolished === "1") {
      if (card) updateExisting(card, rank);
      return;
    }

    const score = extractScore(card);
    const text = textOf(card);
    const tier = tierFromScore(score, text);
    const driver = inferDriver(card);
    const trend = inferTrend(card, score);

    const strip = document.createElement("div");
    strip.className = "era-xai-strip";
    strip.setAttribute("data-era-xai-strip", "1");

    const title = document.createElement("div");
    title.className = "era-xai-strip-title";
    title.innerHTML = `
      <strong>Explainable Review Context</strong>
      <span>Decision support only</span>
    `;

    const badges = document.createElement("div");
    badges.className = "era-xai-badges";

    badges.appendChild(createBadge("Queue Rank", "#" + rank));
    badges.appendChild(createBadge("Priority Tier", tier, tierClass(tier)));
    badges.appendChild(createBadge("Risk Score", score === null ? "contextual" : score));
    badges.appendChild(createBadge("Primary Driver", driver));
    badges.appendChild(createBadge("Trend", trend));

    const sub = document.createElement("div");
    sub.className = "era-xai-subcopy";
    sub.textContent = "Use tier, queue rank, driver, trend, and timing context together. Do not rely on raw score alone.";

    strip.appendChild(title);
    strip.appendChild(badges);
    strip.appendChild(sub);

    card.appendChild(strip);
    card.dataset.eraXaiPolished = "1";
    card.dataset.eraXaiRank = String(rank);
  }

  function updateExisting(card, rank) {
    const strip = card.querySelector("[data-era-xai-strip='1']");
    if (!strip) {
      card.dataset.eraXaiPolished = "0";
      annotateCard(card, rank);
      return;
    }

    const badges = strip.querySelector(".era-xai-badges");
    if (!badges) return;

    const first = badges.querySelector(".era-xai-badge");
    if (first && first.textContent.includes("Queue Rank")) {
      first.innerHTML = `Queue Rank: <b>#${rank}</b>`;
    }

    card.dataset.eraXaiRank = String(rank);
  }

  function insertEvidenceModule() {
    if (document.getElementById("era-command-center-evidence-module")) return;

    const module = document.createElement("section");
    module.id = "era-command-center-evidence-module";
    module.className = "era-xai-module";
    module.innerHTML = `
      <div class="era-xai-module-top">
        <div>
          <div class="era-xai-kicker">Command Center Explainability Layer</div>
          <h2 class="era-xai-title">Review queue context — not just raw score.</h2>
          <p class="era-xai-copy">
            ERA displays priority tier, queue rank, primary driver, trend direction, and validation context to reduce score-saturation concerns and support clinician review.
          </p>
        </div>
        <div class="era-xai-actions">
          <a href="/validation-intelligence">Validation Intelligence</a>
          <a href="/validation-evidence">Evidence Packet</a>
          <a href="/validation-runs">Validation Runs</a>
        </div>
      </div>

      <div class="era-xai-proof-grid">
        <div class="era-xai-proof">
          <span>Alert Reduction</span>
          <b>${ERA_EVIDENCE.alertReduction}</b>
          <small>${ERA_EVIDENCE.threshold} conservative mode</small>
        </div>
        <div class="era-xai-proof">
          <span>Median Lead Time</span>
          <b>${ERA_EVIDENCE.medianLead}</b>
          <small>DUA-safe retrospective testing</small>
        </div>
        <div class="era-xai-proof">
          <span>ERA FPR</span>
          <b>${ERA_EVIDENCE.fpr}</b>
          <small>t=6.0 low-burden review queue</small>
        </div>
        <div class="era-xai-proof">
          <span>High-Acuity Option</span>
          <b>${ERA_EVIDENCE.highAcuity}</b>
          <small>${ERA_EVIDENCE.highAcuityReduction} alert reduction</small>
        </div>
      </div>

      <div class="era-xai-note">
        Decision support only. Retrospective validation context only. Not intended to diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.
      </div>
    `;

    const main = document.querySelector("main") || document.querySelector(".main") || document.body;
    const nav = document.querySelector("nav, header, .navbar, .topbar");

    if (nav && nav.parentElement) {
      nav.parentElement.insertBefore(module, nav.nextSibling);
    } else if (main.firstElementChild) {
      main.insertBefore(module, main.firstElementChild);
    } else {
      main.appendChild(module);
    }
  }

  function polishCommandCenter() {
    insertEvidenceModule();

    const cards = getPatientCards();

    if (!cards.length) return;

    const ranked = scoreSort(cards);
    const rankMap = new Map();

    ranked.forEach((item, idx) => {
      rankMap.set(item.card, idx + 1);
    });

    cards.forEach((card) => {
      annotateCard(card, rankMap.get(card) || 1);
    });
  }

  function start() {
    polishCommandCenter();

    let count = 0;
    const timer = setInterval(() => {
      polishCommandCenter();
      count += 1;
      if (count > 240) clearInterval(timer);
    }, 2500);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
</script>
<!-- ERA_COMMAND_CENTER_EXPLAINABILITY_POLISH_V1_END -->
'''


def patch_file(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")

    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.S)
    text = pattern.sub("", text)

    lower = text.lower()
    idx = lower.rfind("</body>")

    if idx == -1:
      raise SystemExit(f"ERROR: Could not find </body> in {path}. Not patching.")

    patched = text[:idx] + "\n" + SNIPPET + "\n" + text[idx:]

    path.write_text(patched, encoding="utf-8")
    print("PATCHED:", path)


def main():
    target = find_command_center_file()
    patch_file(target)

    DOCS.mkdir(parents=True, exist_ok=True)
    report = f"""# Command Center Explainability Polish

Generated: {datetime.now(timezone.utc).isoformat()}

## Updated File

- `{target}`

## What This Adds

- Command Center evidence module
- 94.3% alert reduction hero stat
- 4.0h median lead-time validation context
- 4.2% ERA FPR at conservative t=6.0
- Patient-card explainability strip
- Queue Rank
- Priority Tier
- Risk Score
- Primary Driver
- Trend Direction
- Score-saturation-safe display language

## Pilot-Safe Intent

This UI layer supports review context and workflow prioritization. It does not diagnose, direct treatment, replace clinician judgment, or independently trigger escalation.
"""
    (DOCS / "command_center_explainability_polish.md").write_text(report, encoding="utf-8")
    print("WROTE docs/validation/command_center_explainability_polish.md")


if __name__ == "__main__":
    main()
