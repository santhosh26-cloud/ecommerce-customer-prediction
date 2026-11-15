(function () {
  const form = document.getElementById("predict-form");
  const inlineBtn = document.getElementById("ajax-predict");
  const resultBox = document.getElementById("ajax-result");

  if (!form || !inlineBtn || !resultBox) return;

  inlineBtn.addEventListener("click", async () => {
    const data = Object.fromEntries(new FormData(form).entries());

    inlineBtn.disabled = true;
    inlineBtn.textContent = "Predicting…";

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });
      const body = await res.json();

      if (!res.ok) {
        renderError(body.error || "Prediction failed.");
        return;
      }
      renderResult(body);
    } catch (err) {
      renderError(err.message || "Network error.");
    } finally {
      inlineBtn.disabled = false;
      inlineBtn.textContent = "Predict inline";
    }
  });

  function renderError(message) {
    resultBox.classList.remove("hidden");
    resultBox.innerHTML = `<h3>Something went wrong</h3><p>${escapeHtml(message)}</p>`;
  }

  function renderResult(body) {
    const probs = Object.entries(body.probabilities)
      .map(
        ([cls, p]) => `
          <div class="prob-row">
            <div class="prob-label"><span>${escapeHtml(cls)}</span><span>${p}%</span></div>
            <div class="bar"><div class="bar-fill" style="width:${p}%"></div></div>
          </div>`
      )
      .join("");

    const recos = (body.recommendations || [])
      .map((tip) => `<li><span class="reco-bullet"></span><span>${escapeHtml(tip)}</span></li>`)
      .join("");

    resultBox.classList.remove("hidden");
    resultBox.innerHTML = `
      <h3>Prediction: <span class="grad">${escapeHtml(body.prediction)}</span></h3>
      <p style="color: var(--muted); margin-top:-4px;">Confidence: <strong>${body.confidence}%</strong></p>
      <div class="probs" style="margin-top: 14px;">${probs}</div>
      ${recos ? `<ul class="reco-list" style="margin-top: 18px;">${recos}</ul>` : ""}
    `;
    resultBox.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
})();
