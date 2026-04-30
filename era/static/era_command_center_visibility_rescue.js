(function () {
  const snapshots = [
    {
      "ICU-12": { score: "8.6/10", tier: "Critical", driver: "SpO₂ decline", trend: "Worsening", lead: "~4.6 hrs" },
      "P101": { score: "8.6/10", tier: "Critical", driver: "SpO₂ decline", trend: "Worsening", lead: "~4.6 hrs" },
      "TELEMETRY-04": { score: "6.2/10", tier: "Watch", driver: "RR variability", trend: "Stable", lead: "~2.8 hrs" },
      "P102": { score: "6.2/10", tier: "Watch", driver: "RR variability", trend: "Stable", lead: "~2.8 hrs" },
      "STEPDOWN-09": { score: "7.4/10", tier: "Elevated", driver: "RR elevation", trend: "Stable / Watch", lead: "~3.1 hrs" },
      "P103": { score: "7.4/10", tier: "Elevated", driver: "RR elevation", trend: "Stable / Watch", lead: "~3.1 hrs" },
      "WARD-21": { score: "5.8/10", tier: "Watch", driver: "BP trend", trend: "Stable", lead: "~2.2 hrs" },
      "P104": { score: "5.8/10", tier: "Watch", driver: "BP trend", trend: "Stable", lead: "~2.2 hrs" }
    },
    {
      "ICU-12": { score: "7.2/10", tier: "Elevated", driver: "SpO₂ recovery watch", trend: "Stable / Watch", lead: "~3.2 hrs" },
      "P101": { score: "7.2/10", tier: "Elevated", driver: "SpO₂ recovery watch", trend: "Stable / Watch", lead: "~3.2 hrs" },
      "TELEMETRY-04": { score: "8.4/10", tier: "Critical", driver: "HR instability", trend: "Worsening", lead: "~4.0 hrs" },
      "P102": { score: "8.4/10", tier: "Critical", driver: "HR instability", trend: "Worsening", lead: "~4.0 hrs" },
      "STEPDOWN-09": { score: "7.0/10", tier: "Elevated", driver: "RR elevation", trend: "Watch", lead: "~3.0 hrs" },
      "P103": { score: "7.0/10", tier: "Elevated", driver: "RR elevation", trend: "Watch", lead: "~3.0 hrs" },
      "WARD-21": { score: "6.1/10", tier: "Watch", driver: "BP trend", trend: "Stable", lead: "~2.5 hrs" },
      "P104": { score: "6.1/10", tier: "Watch", driver: "BP trend", trend: "Stable", lead: "~2.5 hrs" }
    },
    {
      "ICU-12": { score: "8.0/10", tier: "Critical", driver: "SpO₂ decline", trend: "Worsening", lead: "~4.1 hrs" },
      "P101": { score: "8.0/10", tier: "Critical", driver: "SpO₂ decline", trend: "Worsening", lead: "~4.1 hrs" },
      "TELEMETRY-04": { score: "7.3/10", tier: "Elevated", driver: "HR instability", trend: "Stable / Watch", lead: "~3.4 hrs" },
      "P102": { score: "7.3/10", tier: "Elevated", driver: "HR instability", trend: "Stable / Watch", lead: "~3.4 hrs" },
      "STEPDOWN-09": { score: "8.1/10", tier: "Critical", driver: "RR elevation", trend: "Worsening", lead: "~3.8 hrs" },
      "P103": { score: "8.1/10", tier: "Critical", driver: "RR elevation", trend: "Worsening", lead: "~3.8 hrs" },
      "WARD-21": { score: "5.7/10", tier: "Watch", driver: "BP trend", trend: "Stable", lead: "~2.3 hrs" },
      "P104": { score: "5.7/10", tier: "Watch", driver: "BP trend", trend: "Stable", lead: "~2.3 hrs" }
    }
  ];

  let idx = 0;

  function ready(fn) {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", fn);
    } else {
      fn();
    }
  }

  function isCommandCenter() {
    const path = window.location.pathname || "";
    const text = document.body ? (document.body.innerText || "") : "";
    return path.includes("command-center") || text.includes("Hospital Command Wall") || text.includes("Prioritized patient review queue");
  }

  function setup() {
    if (!isCommandCenter()) return;

    const root = document.getElementById("era-command-tool-mode");
    if (!root) {
      setTimeout(setup, 500);
      return;
    }

    root.classList.add("era-cc-rescue-compact");
    if (!root.classList.contains("era-cc-expanded") && !root.classList.contains("era-cc-minimized")) {
      root.classList.remove("era-cc-expanded");
      root.classList.remove("era-cc-minimized");
    }

    ensureControls(root);
    ensureOriginalMarker(root);
    normalizeLegacyCards();

    setInterval(function () {
      idx = (idx + 1) % snapshots.length;
      normalizeLegacyCards();
    }, 11000);

    const observer = new MutationObserver(function () {
      root.classList.add("era-cc-rescue-compact");
      ensureControls(root);
      ensureOriginalMarker(root);
      normalizeLegacyCards();
    });

    observer.observe(document.body, { childList: true, subtree: true });
  }

  function ensureControls(root) {
    if (document.getElementById("era-command-center-rescue-controls")) return;

    const bar = document.createElement("div");
    bar.id = "era-command-center-rescue-controls";
    bar.innerHTML = `
      <div class="era-rescue-label">
        Compact review queue is active. The full original Command Center continues below.
      </div>
      <div class="era-rescue-buttons">
        <button type="button" class="era-rescue-btn" id="era-rescue-compact">Compact Queue</button>
        <button type="button" class="era-rescue-btn" id="era-rescue-expand">Expand Queue</button>
        <button type="button" class="era-rescue-btn" id="era-rescue-minimize">Minimize Queue</button>
        <button type="button" class="era-rescue-btn" id="era-rescue-scroll">Go to Full Command Center</button>
      </div>
    `;

    root.parentElement.insertBefore(bar, root);

    document.getElementById("era-rescue-compact").addEventListener("click", function () {
      root.classList.remove("era-cc-expanded", "era-cc-minimized");
      root.classList.add("era-cc-rescue-compact");
      root.scrollIntoView({ behavior: "smooth", block: "start" });
    });

    document.getElementById("era-rescue-expand").addEventListener("click", function () {
      root.classList.add("era-cc-expanded");
      root.classList.remove("era-cc-minimized");
      root.scrollIntoView({ behavior: "smooth", block: "start" });
    });

    document.getElementById("era-rescue-minimize").addEventListener("click", function () {
      root.classList.add("era-cc-minimized");
      root.classList.remove("era-cc-expanded");
    });

    document.getElementById("era-rescue-scroll").addEventListener("click", function () {
      const marker = document.getElementById("era-original-command-center-marker");
      if (marker) marker.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  }

  function ensureOriginalMarker(root) {
    if (document.getElementById("era-original-command-center-marker")) return;

    const marker = document.createElement("div");
    marker.id = "era-original-command-center-marker";
    marker.innerHTML = `
      <strong>Full Command Center continues below.</strong>
      The compact review queue above is an operational summary layer. Your original pilot controls, governance blocks,
      site-specific packet, validation panels, and command wall remain below this marker.
    `;

    root.parentElement.insertBefore(marker, root.nextSibling);
  }

  function normalizeLegacyCards() {
    const snap = snapshots[idx];

    const candidates = Array.from(document.querySelectorAll("section, article, div, li, tr"))
      .filter(el => !el.closest("#era-command-tool-mode"))
      .filter(el => {
        const t = (el.textContent || "").toUpperCase();
        return Object.keys(snap).some(k => t.includes(k));
      })
      .filter(el => (el.textContent || "").length < 2500);

    candidates.forEach(el => {
      const upper = (el.textContent || "").toUpperCase();
      const key = Object.keys(snap).find(k => upper.includes(k));
      if (!key) return;

      const row = snap[key];

      replaceLegacyText(el, row);
      addLegacyNote(el, row);
    });
  }

  function replaceLegacyText(el, row) {
    const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, null);
    const nodes = [];
    let node;
    while ((node = walker.nextNode())) nodes.push(node);

    nodes.forEach(n => {
      let s = n.nodeValue || "";

      s = s.replace(/\bRISK\b/g, "Review Score");
      s = s.replace(/\bRisk\b/g, "Review Score");

      s = s.replace(/Review Score\s*99%/gi, "Review Score " + row.score);
      s = s.replace(/Review score\s*99%/gi, "Review Score " + row.score);
      s = s.replace(/risk\s*99%/gi, "Review Score " + row.score);

      if (/^\s*99%\s*$/.test(s) && nearbyText(n.parentElement).toLowerCase().includes("review score")) {
        s = s.replace(/99%/g, row.score);
      }

      if (/^\s*(94|86|72|37|36|17|9)%\s*$/.test(s) && nearbyText(n.parentElement).toLowerCase().includes("review score")) {
        s = row.score;
      }

      if (/Review score\s+(94|86|72|37|36|17|9)%/i.test(s)) {
        s = s.replace(/Review score\s+(94|86|72|37|36|17|9)%/i, "Review Score " + row.score);
      }

      n.nodeValue = s;
    });
  }

  function addLegacyNote(el, row) {
    if (el.querySelector && el.querySelector(".era-legacy-score-fix-note")) {
      const existing = el.querySelector(".era-legacy-score-fix-note");
      existing.textContent = noteText(row);
      return;
    }

    const text = (el.textContent || "").toLowerCase();

    if (
      text.includes("clinical review attention") ||
      text.includes("current monitored review") ||
      text.includes("top review") ||
      text.includes("review score")
    ) {
      const note = document.createElement("div");
      note.className = "era-legacy-score-fix-note";
      note.textContent = noteText(row);

      if (el.firstElementChild) {
        el.insertBefore(note, el.firstElementChild.nextSibling);
      } else {
        el.prepend(note);
      }
    }
  }

  function noteText(row) {
    return "Current rotating review context: " +
      row.tier +
      " · Review Score " +
      row.score +
      " · Driver: " +
      row.driver +
      " · Trend: " +
      row.trend +
      " · Lead: " +
      row.lead +
      ".";
  }

  function nearbyText(el) {
    let cur = el;
    let txt = "";
    let depth = 0;
    while (cur && depth < 4) {
      txt += " " + (cur.textContent || "");
      cur = cur.parentElement;
      depth += 1;
    }
    return txt;
  }

  ready(setup);
})();
