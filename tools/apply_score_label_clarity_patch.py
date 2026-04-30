#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import ast
import json
import re

ROOT = Path(".")
ERA = ROOT / "era"
WEB = ERA / "web"
STATIC = ERA / "static"
INIT = ERA / "__init__.py"

CSS_FILE = STATIC / "era_score_label_clarity.css"
JS_FILE = STATIC / "era_score_label_clarity.js"
MANIFEST = ROOT / "data" / "validation" / "score_label_clarity_patch_manifest.json"

HEAD_START = "<!-- ERA_SCORE_LABEL_CLARITY_HEAD_START -->"
HEAD_END = "<!-- ERA_SCORE_LABEL_CLARITY_HEAD_END -->"
BODY_START = "<!-- ERA_SCORE_LABEL_CLARITY_BODY_START -->"
BODY_END = "<!-- ERA_SCORE_LABEL_CLARITY_BODY_END -->"

ROUTE_START = "# ERA_STATIC_ASSET_ROUTE_V1_START"
ROUTE_END = "# ERA_STATIC_ASSET_ROUTE_V1_END"

CSS = r'''
.era-metric-clarity-banner{
  margin:12px 0 14px 0;
  padding:10px 12px;
  border-radius:14px;
  border:1px solid rgba(132,255,180,.26);
  background:linear-gradient(90deg, rgba(132,255,180,.12), rgba(126,190,255,.08));
  font-size:13px;
  line-height:1.45;
  font-weight:750;
}
.era-metric-clarity-banner strong{
  color:#b8ffcc;
}
.era-review-score-chip{
  display:inline-flex;
  align-items:center;
  justify-content:center;
  padding:4px 8px;
  border-radius:999px;
  border:1px solid rgba(132,255,180,.28);
  background:rgba(132,255,180,.10);
  font-weight:900;
  white-space:nowrap;
}
.era-validation-metric-chip{
  display:inline-flex;
  align-items:center;
  justify-content:center;
  padding:4px 8px;
  border-radius:999px;
  border:1px solid rgba(126,190,255,.30);
  background:rgba(126,190,255,.10);
  font-weight:900;
  white-space:nowrap;
}
'''

JS = r'''
(function(){
  function ready(fn){
    if(document.readyState !== "loading"){ fn(); }
    else{ document.addEventListener("DOMContentLoaded", fn); }
  }

  function textOf(el){
    return (el && el.textContent ? el.textContent : "").trim();
  }

  function walkTextLeaves(root, fn){
    if(!root) return;
    const all = Array.from(root.querySelectorAll("*"));
    all.forEach(function(el){
      if(el.children && el.children.length) return;
      const t = textOf(el);
      if(!t) return;
      fn(el, t);
    });
  }

  function findQueueRoots(){
    const headings = Array.from(document.querySelectorAll("h1,h2,h3,h4,.title,.section-title"));
    const roots = [];

    headings.forEach(function(h){
      const t = textOf(h).toLowerCase();
      if(
        t.includes("prioritized patient review queue") ||
        t.includes("top review queue") ||
        t.includes("hospital command wall") ||
        t.includes("review queue")
      ){
        const root = h.closest("section, article, .card, .panel, div") || h.parentElement;
        if(root && !roots.includes(root)) roots.push(root);
      }
    });

    const explicit = document.getElementById("era-realistic-command-center-queue");
    if(explicit && !roots.includes(explicit)) roots.unshift(explicit);

    return roots;
  }

  function findValidationRoots(){
    const headings = Array.from(document.querySelectorAll("h1,h2,h3,h4,.title,.section-title"));
    const roots = [];

    headings.forEach(function(h){
      const t = textOf(h).toLowerCase();
      if(
        t.includes("validation intelligence") ||
        t.includes("mimic validation") ||
        t.includes("alert reduction") ||
        t.includes("validation evidence") ||
        t.includes("validation run")
      ){
        const root = h.closest("section, article, .card, .panel, div") || h.parentElement;
        if(root && !roots.includes(root)) roots.push(root);
      }
    });

    return roots;
  }

  function addQueueBanner(root){
    if(!root || root.querySelector(".era-metric-clarity-banner")) return;

    const banner = document.createElement("div");
    banner.className = "era-metric-clarity-banner";
    banner.innerHTML =
      "<strong>Metric clarity:</strong> Review scores are case-level queue rankings on a 0–10 scale. " +
      "Validation percentages such as 94.3% are aggregate alert-reduction metrics, not patient risk.";

    const heading = Array.from(root.querySelectorAll("h1,h2,h3,h4")).find(function(h){
      return /review queue|command wall|patient/i.test(textOf(h));
    });

    if(heading && heading.parentNode){
      if(heading.nextSibling) heading.parentNode.insertBefore(banner, heading.nextSibling);
      else heading.parentNode.appendChild(banner);
    } else {
      root.insertBefore(banner, root.firstChild);
    }
  }

  function addValidationBanner(root){
    if(!root || root.querySelector(".era-validation-metric-clarity-banner")) return;

    const banner = document.createElement("div");
    banner.className = "era-metric-clarity-banner era-validation-metric-clarity-banner";
    banner.innerHTML =
      "<strong>Validation metric note:</strong> Alert reduction, FPR, detection, and lead-time are aggregate retrospective evidence metrics. " +
      "They should not be read as individual patient-risk percentages.";

    const heading = Array.from(root.querySelectorAll("h1,h2,h3,h4")).find(function(h){
      return /validation|alert reduction|mimic/i.test(textOf(h));
    });

    if(heading && heading.parentNode){
      if(heading.nextSibling) heading.parentNode.insertBefore(banner, heading.nextSibling);
      else heading.parentNode.appendChild(banner);
    }
  }

  function normalizeQueueLabels(root){
    walkTextLeaves(root, function(el, t){
      let n = t;

      n = n.replace(/^Risk$/i, "Review Score");
      n = n.replace(/^Risk Score$/i, "Review Score");
      n = n.replace(/^Current Risk$/i, "Review Score");
      n = n.replace(/^Patient Risk$/i, "Review Score");
      n = n.replace(/^Risk %$/i, "Review Score");
      n = n.replace(/\bpatient risk score\b/ig, "patient review score");
      n = n.replace(/\brisk score\b/ig, "review score");
      n = n.replace(/\bcurrent risk\b/ig, "review score");

      n = n.replace(/\b([0-9](?:\.[0-9])?)\s*%\b/g, function(match, num){
        const value = parseFloat(num);
        if(value > 0 && value < 10) return num + " / 10";
        return match;
      });

      n = n.replace(/\b([0-9](?:\.[0-9])?)\s*\/\s*10\b/g, function(match){
        return match;
      });

      if(n !== t){
        el.textContent = n;
      }

      const finalText = textOf(el);
      if(/^[0-9](?:\.[0-9])?\s*\/\s*10$/.test(finalText)){
        el.classList.add("era-review-score-chip");
      }
    });
  }

  function normalizeValidationLabels(root){
    walkTextLeaves(root, function(el, t){
      let n = t;

      n = n.replace(/\b([0-9]{2,3}(?:\.[0-9])?)\s*%\s*risk\b/ig, "$1% alert reduction");
      n = n.replace(/^Risk$/i, "Alert Reduction");
      n = n.replace(/^Risk Reduction$/i, "Alert Reduction");
      n = n.replace(/^Patient Risk$/i, "Aggregate Metric");

      if(n !== t){
        el.textContent = n;
      }

      const finalText = textOf(el);
      if(/^[0-9]{2,3}(?:\.[0-9])?%$/.test(finalText)){
        const nearby = textOf(el.closest("section,article,div,td,li"));
        if(/alert reduction|validation|mimic|eicu|fpr|detection|lead/i.test(nearby)){
          el.classList.add("era-validation-metric-chip");
        }
      }
    });
  }

  function removeMisleadingRiskNearPercent(){
    Array.from(document.querySelectorAll("body *")).forEach(function(el){
      if(el.children && el.children.length) return;

      const t = textOf(el);
      if(!t) return;

      if(/\b9[0-9](?:\.[0-9])?%\s*risk\b/i.test(t)){
        el.textContent = t.replace(/\brisk\b/ig, "alert reduction");
      }
    });
  }

  function run(){
    const queueRoots = findQueueRoots();
    queueRoots.forEach(function(root){
      addQueueBanner(root);
      normalizeQueueLabels(root);
    });

    const validationRoots = findValidationRoots();
    validationRoots.forEach(function(root){
      addValidationBanner(root);
      normalizeValidationLabels(root);
    });

    removeMisleadingRiskNearPercent();
  }

  ready(function(){
    run();
    setTimeout(run, 300);
    setTimeout(run, 1200);
    setTimeout(run, 2500);
  });
})();
'''

def read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def write(p: Path, s: str) -> None:
    p.write_text(s, encoding="utf-8")

def remove_block(text: str, start: str, end: str) -> str:
    while start in text and end in text:
        a = text.index(start)
        b = text.index(end) + len(end)
        text = text[:a] + text[b:]
    return text

def find_command_center_file() -> Path | None:
    preferred = [
        WEB / "command_center.html",
        WEB / "command_center.py",
        ERA / "assistant_platform.py",
        INIT,
    ]

    for p in preferred:
        if p.exists():
            txt = read(p).lower()
            if "</body>" in txt and ("command center" in txt or "hospital command wall" in txt):
                return p

    best = None
    best_score = -1

    for p in ERA.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in {".html", ".py"}:
            continue

        txt = read(p).lower()
        if "</body>" not in txt:
            continue

        score = 0
        for token in [
            "command center",
            "hospital command wall",
            "prioritized patient review queue",
            "mimic validation intelligence",
            "site-specific pilot packet",
        ]:
            if token in txt:
                score += 2

        if "command" in p.name.lower():
            score += 3

        if score > best_score:
            best_score = score
            best = p

    return best if best_score >= 4 else None

def upsert_asset_tags(text: str) -> tuple[str, dict]:
    actions = {}

    head_block = (
        f"{HEAD_START}\n"
        '<link rel="stylesheet" href="/era-static/era_score_label_clarity.css?v=score-label1">\n'
        f"{HEAD_END}"
    )
    body_block = (
        f"{BODY_START}\n"
        '<script src="/era-static/era_score_label_clarity.js?v=score-label1"></script>\n'
        f"{BODY_END}"
    )

    text = remove_block(text, HEAD_START, HEAD_END)
    text = remove_block(text, BODY_START, BODY_END)

    if "</head>" in text:
        text = text.replace("</head>", head_block + "\n</head>", 1)
        actions["head_tags"] = "inserted"
    else:
        text = head_block + "\n" + text
        actions["head_tags"] = "prepended"

    if "</body>" in text:
        text = text.replace("</body>", body_block + "\n</body>", 1)
        actions["body_tags"] = "inserted"
    else:
        text = text + "\n" + body_block + "\n"
        actions["body_tags"] = "appended"

    return text, actions

def patch_static_route() -> dict:
    if not INIT.exists():
        return {"status": "missing_init"}

    text = read(INIT)
    original = text

    if "/era-static/<path:filename>" in text:
        return {"status": "already_present", "file": str(INIT)}

    try:
        tree = ast.parse(text)
    except Exception as exc:
        return {"status": "parse_error", "error": str(exc)}

    create_app_node = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "create_app":
            create_app_node = node
            break

    if create_app_node is None:
        return {"status": "create_app_not_found"}

    return_lines = []
    for node in ast.walk(create_app_node):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Name) and node.value.id == "app":
            return_lines.append(node.lineno)

    if not return_lines:
        return {"status": "return_app_not_found"}

    insert_line = max(return_lines)
    lines = text.splitlines()
    return_line = lines[insert_line - 1]
    indent = re.match(r"\s*", return_line).group(0)

    block = [
        f"{indent}{ROUTE_START}",
        f'{indent}@app.get("/era-static/<path:filename>")',
        f"{indent}def era_static_asset_route_v1(filename):",
        f"{indent}    from pathlib import Path",
        f"{indent}    from flask import send_from_directory",
        f'{indent}    static_dir = Path(__file__).resolve().parent / "static"',
        f"{indent}    return send_from_directory(str(static_dir), filename)",
        f"{indent}{ROUTE_END}",
        "",
    ]

    lines = lines[:insert_line - 1] + block + lines[insert_line - 1:]
    text = "\n".join(lines) + "\n"

    if text != original:
        write(INIT, text)
        return {"status": "patched", "file": str(INIT)}

    return {"status": "unchanged", "file": str(INIT)}

def main():
    STATIC.mkdir(parents=True, exist_ok=True)

    write(CSS_FILE, CSS.strip() + "\n")
    write(JS_FILE, JS.strip() + "\n")

    target = find_command_center_file()
    if not target:
        raise SystemExit("Could not safely find Command Center HTML-bearing file. No patch applied.")

    print("Command Center target:", target)

    old = read(target)
    new, actions = upsert_asset_tags(old)

    if new != old:
        write(target, new)
        print("Injected score-label clarity static assets.")
    else:
        print("No command-center changes needed.")

    route_action = patch_static_route()

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "target_file": str(target),
        "css_file": str(CSS_FILE),
        "js_file": str(JS_FILE),
        "actions": actions,
        "static_route": route_action,
        "purpose": "Clarify that case-level review scores are 0-10 and validation percentages are aggregate alert-reduction metrics.",
        "claim_policy": [
            "Review score is not patient risk percent.",
            "Alert reduction percent is aggregate validation evidence.",
            "Decision support only.",
            "No diagnosis, treatment direction, clinician replacement, or independent escalation."
        ]
    }

    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Manifest:", MANIFEST)

if __name__ == "__main__":
    main()
