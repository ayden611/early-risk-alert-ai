(function () {
  const snapshots = [
    {
      "ICU-12": { score: "8.6/10", tier: "Critical", driver: "SpO₂ decline", trend: "Worsening", lead: "~4.6 hrs" },
      "P101": { score: "8.6/10", tier: "Critical", driver: "SpO₂ decline", trend: "Worsening", lead: "~4.6 hrs" }
    },
    {
      "ICU-12": { score: "7.2/10", tier: "Elevated", driver: "SpO₂ recovery watch", trend: "Stable / Watch", lead: "~3.2 hrs" },
      "P101": { score: "7.2/10", tier: "Elevated", driver: "SpO₂ recovery watch", trend: "Stable / Watch", lead: "~3.2 hrs" }
    },
    {
      "ICU-12": { score: "8.0/10", tier: "Critical", driver: "SpO₂ decline", trend: "Worsening", lead: "~4.1 hrs" },
      "P101": { score: "8.0/10", tier: "Critical", driver: "SpO₂ decline", trend: "Worsening", lead: "~4.1 hrs" }
    },
    {
      "ICU-12": { score: "6.9/10", tier: "Watch", driver: "Oxygenation stable", trend: "Stable", lead: "~2.5 hrs" },
      "P101": { score: "6.9/10", tier: "Watch", driver: "Oxygenation stable", trend: "Stable", lead: "~2.5 hrs" }
    }
  ];

  let index = 0;

  function ready(fn) {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", fn);
    } else {
      fn();
    }
  }

  function isCommandCenter() {
    return (window.location.pathname || "").includes("command-center") ||
      ((document.body && document.body.innerText) || "").includes("Command Center");
  }

  function nearbyText(el) {
    let cur = el;
    let text = "";
    let depth = 0;
    while (cur && depth < 5) {
      text += " " + (cur.textContent || "");
      cur = cur.parentElement;
      depth++;
    }
    return text;
  }

  function replaceTextNodes(root, snap) {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, null);
    const nodes = [];
    let node;
    while ((node = walker.nextNode())) nodes.push(node);

    nodes.forEach(function (n) {
      const parentText = nearbyText(n.parentElement || root).toUpperCase();
      let s = n.nodeValue || "";

      const isIcu12Area =
        parentText.includes("ICU-12") ||
        parentText.includes("P101") ||
        parentText.includes("CLINICAL REVIEW ATTENTION") ||
        parentText.includes("TOP REVIEW-PRIORITY QUEUE");

      if (!isIcu12Area) return;

      s = s.replace(/\bRISK\b/g, "Review Score");
      s = s.replace(/\bRisk\b/g, "Review Score");

      s = s.replace(/Review Score\s*99%/gi, "Review Score " + snap["ICU-12"].score);
      s = s.replace(/review score\s*99%/gi, "Review Score " + snap["ICU-12"].score);
      s = s.replace(/risk\s*99%/gi, "Review Score " + snap["ICU-12"].score);

      if (/^\s*99%\s*$/.test(s)) {
        s = snap["ICU-12"].score;
      }

      n.nodeValue = s;
    });
  }

  function addClarityNote(snap) {
    if (document.getElementById("era-score-clarity-note")) return;

    const target =
      Array.from(document.querySelectorAll("section, div, article"))
        .find(el => {
          const t = (el.textContent || "");
          return t.includes("Prioritized patient review queue") || t.includes("Hospital Command Wall");
        }) || document.body;

    const note = document.createElement("div");
    note.id = "era-score-clarity-note";
    note.style.cssText = [
      "max-width:1500px",
      "margin:8px auto",
      "padding:9px 12px",
      "border:1px solid rgba(255,207,104,.35)",
      "border-radius:12px",
      "background:rgba(255,207,104,.08)",
      "color:#ffe7a8",
      "font-size:12px",
      "font-weight:800",
      "line-height:1.35"
    ].join(";");

    note.textContent =
      "Metric clarity: patient cards use a 0–10 Review Score for simulated queue prioritization. Aggregate validation percentages remain separate evidence metrics.";

    if (target && target.parentElement) {
      target.parentElement.insertBefore(note, target);
    }
  }

  function apply() {
    if (!isCommandCenter()) return;
    const snap = snapshots[index % snapshots.length];
    replaceTextNodes(document.body, snap);
    addClarityNote(snap);
  }

  ready(function () {
    apply();
    setInterval(function () {
      index = (index + 1) % snapshots.length;
      apply();
    }, 9000);

    const obs = new MutationObserver(function () {
      apply();
    });
    obs.observe(document.body, { childList: true, subtree: true });
  });
})();
