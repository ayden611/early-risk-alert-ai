(function(){
  function ready(fn){
    if(document.readyState === "loading"){ document.addEventListener("DOMContentLoaded", fn); }
    else{ fn(); }
  }

  function onValidationPage(){
    const p = window.location.pathname || "";
    return p.includes("validation-intelligence") ||
           p.includes("validation-evidence") ||
           p.includes("validation-runs") ||
           p.includes("evidence-packet");
  }

  function removeOldAlignmentDuplicates(){
    const ids = [
      "era-validation-alignment-v2",
      "era-eicu-metric-source-clarity",
      "era-validation-metric-note"
    ];

    ids.forEach(id => {
      const nodes = Array.from(document.querySelectorAll("#" + id));
      nodes.forEach((n, i) => {
        if(i > 0) n.remove();
      });
    });

    Array.from(document.querySelectorAll(".era-eicu-clarity-box")).forEach(el => {
      if(!el.id || el.id !== "era-validation-alignment-v2"){
        el.style.display = "none";
      }
    });
  }

  function replaceLoadingText(){
    Array.from(document.querySelectorAll("body *")).forEach(el => {
      if(el.children && el.children.length) return;
      const t = (el.textContent || "").trim();
      if(/^Loading\.{0,3}$/i.test(t)){
        el.textContent = "Locked aggregate summary displayed above";
      }
    });
  }

  function insertPanel(){
    if(!onValidationPage()) return;
    if(document.getElementById("era-validation-alignment-v2")) return;

    const root = document.querySelector("main") ||
                 document.querySelector(".main-content") ||
                 document.querySelector(".container") ||
                 document.body;

    const panel = document.createElement("section");
    panel.id = "era-validation-alignment-v2";
    panel.className = "era-validation-align-box";
    panel.innerHTML = `
      <span class="era-validation-align-kicker">Locked aggregate metric alignment</span>
      <h2>MIMIC-IV + eICU evidence stays separated by event definition</h2>
      <p>
        This panel is the public-facing consistency layer for validation pages. It separates MIMIC-IV strict
        clinical-event evidence, eICU harmonized clinical-event evidence, and the earlier eICU outcome-proxy check.
      </p>

      <div class="era-validation-table-wrap">
        <table class="era-validation-align-table">
          <thead>
            <tr>
              <th>Evidence track</th>
              <th>Event definition</th>
              <th>t=6.0 alert reduction</th>
              <th>t=6.0 FPR</th>
              <th>t=6.0 detection</th>
              <th>Lead-time context</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>MIMIC-IV strict clinical-event release</strong><br>Full cohort + subcohorts B/C</td>
              <td>Strict clinical-event labels / event clusters</td>
              <td>94.0%-94.9%</td>
              <td>3.7%-4.4%</td>
              <td>14.1%-16.0%</td>
              <td>4.0 hrs</td>
            </tr>
            <tr>
              <td><strong>eICU harmonized clinical-event full cohort</strong></td>
              <td>Harmonized clinical-event labeling pass</td>
              <td>94.25%</td>
              <td>0.98%</td>
              <td>24.66%</td>
              <td>4.83 hrs</td>
            </tr>
            <tr>
              <td><strong>eICU harmonized subcohorts B/C</strong></td>
              <td>Harmonized clinical-event labeling pass; subcohort robustness check</td>
              <td>95.26%-95.64%</td>
              <td>1.78%-2.02%</td>
              <td>64.03%-64.49%</td>
              <td>3.13-3.69 hrs</td>
            </tr>
            <tr>
              <td><strong>eICU outcome-proxy check</strong><br>Separate earlier second-dataset track</td>
              <td>Mortality/discharge-derived outcome-proxy context</td>
              <td>96.8%</td>
              <td>1.8%</td>
              <td>66.6%</td>
              <td>3.41 hrs</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div class="era-validation-align-warning">
        Guardrail: do not merge the eICU outcome-proxy result with the harmonized clinical-event pass.
        Detection rates should be interpreted with event-definition context. Retrospective aggregate evidence only;
        not prospective clinical-performance evidence.
      </div>
    `;

    const h = Array.from(document.querySelectorAll("h1,h2")).find(x => {
      const t = (x.textContent || "").toLowerCase();
      return t.includes("validation") || t.includes("evidence") || t.includes("runs");
    });

    if(h && h.parentNode){
      if(h.nextSibling) h.parentNode.insertBefore(panel, h.nextSibling);
      else h.parentNode.appendChild(panel);
    } else if(root.firstChild){
      root.insertBefore(panel, root.firstChild);
    } else {
      root.appendChild(panel);
    }
  }

  ready(function(){
    if(!onValidationPage()) return;
    removeOldAlignmentDuplicates();
    insertPanel();
    replaceLoadingText();
    setTimeout(function(){
      removeOldAlignmentDuplicates();
      insertPanel();
      replaceLoadingText();
    }, 800);
    setTimeout(function(){
      removeOldAlignmentDuplicates();
      insertPanel();
      replaceLoadingText();
    }, 1800);
  });
})();
