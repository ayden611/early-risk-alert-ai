(function () {
  function ready(fn) {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", fn);
    else fn();
  }

  function isValidationPage() {
    const p = window.location.pathname || "";
    return (
      p.includes("validation-intelligence") ||
      p.includes("validation-evidence") ||
      p.includes("validation-runs") ||
      p.includes("evidence-packet")
    );
  }

  function replaceLoadingOnly() {
    Array.from(document.querySelectorAll("body *")).forEach(function (el) {
      if (el.children && el.children.length) return;
      const t = (el.textContent || "").trim();
      if (/^Loading\.{0,3}$/i.test(t)) {
        el.textContent = "Locked aggregate summary available";
      }
    });
  }

  function removeDuplicatePanels() {
    const panels = Array.from(document.querySelectorAll("#era-public-alignment-safe-panel"));
    panels.forEach(function (p, i) {
      if (i > 0) p.remove();
    });
  }

  function insertPanel() {
    if (document.getElementById("era-public-alignment-safe-panel")) return;

    const panel = document.createElement("section");
    panel.id = "era-public-alignment-safe-panel";
    panel.className = "era-public-alignment-box";
    panel.innerHTML = `
      <span class="era-public-alignment-kicker">Public evidence alignment</span>
      <h2>MIMIC-IV + eICU retrospective evidence, separated by event definition</h2>
      <p>
        Early Risk Alert AI now has locked retrospective aggregate evidence across MIMIC-IV and eICU.
        The public framing should stay conservative: threshold-direction behavior is consistent, while detection
        rates must be interpreted by event definition.
      </p>

      <div class="era-public-alignment-table-wrap">
        <table class="era-public-alignment-table">
          <thead>
            <tr>
              <th>Evidence Track</th>
              <th>Event Definition</th>
              <th>t=6.0 Alert Reduction</th>
              <th>t=6.0 FPR</th>
              <th>t=6.0 Detection</th>
              <th>Lead-Time Context</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>MIMIC-IV strict clinical-event evidence</strong><br>Full cohort + subcohorts B/C</td>
              <td>Strict clinical-event labels / event clusters</td>
              <td>94.0%–94.9%</td>
              <td>3.7%–4.4%</td>
              <td>14.1%–16.0%</td>
              <td>4.0 hrs</td>
            </tr>
            <tr>
              <td><strong>eICU harmonized clinical-event pass</strong><br>Full cohort</td>
              <td>Harmonized clinical-event labeling pass</td>
              <td>94.25%</td>
              <td>0.98%</td>
              <td>24.66%</td>
              <td>4.83 hrs</td>
            </tr>
            <tr>
              <td><strong>eICU harmonized subcohorts B/C</strong></td>
              <td>Harmonized clinical-event labeling pass, subcohort robustness</td>
              <td>95.26%–95.64%</td>
              <td>1.78%–2.02%</td>
              <td>64.03%–64.49%</td>
              <td>3.13–3.69 hrs</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div class="era-public-alignment-note">
        Guardrail: These are retrospective aggregate results. Do not claim prospective validation,
        diagnosis, treatment direction, clinician replacement, or independent escalation.
      </div>
    `;

    const main = document.querySelector("main") || document.body;
    const firstH = Array.from(document.querySelectorAll("h1,h2")).find(function (h) {
      const t = (h.textContent || "").toLowerCase();
      return t.includes("validation") || t.includes("evidence") || t.includes("runs");
    });

    if (firstH && firstH.parentElement) {
      firstH.parentElement.insertBefore(panel, firstH.nextSibling);
    } else {
      main.insertBefore(panel, main.firstChild);
    }
  }

  ready(function () {
    if (!isValidationPage()) return;
    replaceLoadingOnly();
    insertPanel();
    removeDuplicatePanels();

    setTimeout(function () {
      replaceLoadingOnly();
      insertPanel();
      removeDuplicatePanels();
    }, 1000);
  });
})();
