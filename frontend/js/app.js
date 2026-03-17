(function () {
  // ─── Helpers ────────────────────────────────────────────────
  function getApiBase() {
    return typeof window.API_BASE !== "undefined"
      ? window.API_BASE
      : window.location.origin;
  }

  function qs(id) {
    return document.getElementById(id);
  }

  // ─── DTI live update ─────────────────────────────────────────
  function updateDTI() {
    var salary = parseFloat(qs("salary").value) || 1;
    var debt = parseFloat(qs("monthly_debt").value) || 0;
    qs("dti-display").textContent = (debt / salary).toFixed(3);
  }

  qs("salary").addEventListener("input", updateDTI);
  qs("monthly_debt").addEventListener("input", updateDTI);
  updateDTI();

  // ─── Build payload ────────────────────────────────────────────
  function getPayload() {
    return {
      age: parseInt(qs("age").value),
      gender: qs("gender").value,
      employment_status: qs("employment_status").value,
      salary: parseFloat(qs("salary").value),
      credit_score: parseInt(qs("credit_score").value),
      previous_delinquencies: parseInt(qs("previous_delinquencies").value),
      existing_loans_count: parseInt(qs("existing_loans_count").value),
      savings_account: qs("savings_account").value,
      loan_amount: parseFloat(qs("loan_amount").value),
      loan_term_months: parseInt(qs("loan_term_months").value),
      monthly_debt: parseFloat(qs("monthly_debt").value),
    };
  }

  // ─── UI helpers ───────────────────────────────────────────────
  function showError(message) {
    var box = qs("error-box");
    box.textContent = message;
    box.classList.remove("hidden");
    qs("result-card").classList.add("hidden");
  }

  function showResult(data) {
    qs("error-box").classList.add("hidden");
    var card = qs("result-card");
    card.classList.remove("hidden");
    var content = qs("result-content");

    if (data.approved) {
      content.innerHTML =
        '<div class="result-inner">' +
          '<span class="badge badge-success">✓ Crédito Aprovado</span>' +
          '<p class="result-headline">Empréstimo aprovado!</p>' +
          '<p class="result-meta">' + escapeHtml(data.message) + "</p>" +
          '<div class="rate-row">' +
            '<span class="rate-label">Taxa de Juros Estimada</span>' +
            '<span class="rate-value">' + data.interest_rate.toFixed(2) + "%</span>" +
          "</div>" +
        "</div>";
    } else {
      content.innerHTML =
        '<div class="result-inner">' +
          '<span class="badge badge-danger">✗ Crédito Negado</span>' +
          '<p class="result-headline">Solicitação não aprovada</p>' +
          '<p class="result-meta">' + escapeHtml(data.message) + "</p>" +
        "</div>";
    }
  }

  function escapeHtml(s) {
    var div = document.createElement("div");
    div.textContent = s;
    return div.innerHTML;
  }

  // ─── Submit ───────────────────────────────────────────────────
  qs("submit-btn").addEventListener("click", function () {
    var btn = qs("submit-btn");
    btn.disabled = true;
    btn.textContent = "Analisando…";
    qs("error-box").classList.add("hidden");

    fetch(getApiBase() + "/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(getPayload()),
    })
      .then(function (response) {
        return response.json().then(function (body) {
          if (!response.ok) {
            var detail = body.detail;
            if (Array.isArray(detail)) {
              detail = detail.map(function (e) { return e.msg || e; }).join("; ");
            }
            throw new Error(detail || "Erro na requisição");
          }
          return body;
        });
      })
      .then(showResult)
      .catch(function (err) { showError("Erro: " + err.message); })
      .finally(function () {
        btn.disabled = false;
        btn.textContent = "Avaliar Empréstimo";
      });
  });

  // ─── Metrics modal ────────────────────────────────────────────
  qs("metrics-btn").addEventListener("click", openMetrics);
  qs("modal-close").addEventListener("click", closeModal);
  qs("modal-overlay").addEventListener("click", function (e) {
    if (e.target === qs("modal-overlay")) closeModal();
  });

  function openMetrics() {
    qs("modal-overlay").classList.remove("hidden");
    qs("modal-body").textContent = "Carregando…";

    fetch(getApiBase() + "/metrics")
      .then(function (r) { return r.json(); })
      .then(renderMetrics)
      .catch(function () {
        qs("modal-body").innerHTML =
          '<div class="error-box">Falha ao carregar métricas. ' +
          "Certifique-se de rodar <code>python src/evaluate.py</code> antes.</div>";
      });
  }

  function closeModal() {
    qs("modal-overlay").classList.add("hidden");
  }

  function renderMetrics(data) {
    var clf = data.classification;
    var reg = data.regression;

    qs("modal-body").innerHTML =
      '<p class="modal-section">Classificação</p>' +
      '<div class="mmetrics-grid">' +
        '<div class="m-item"><span class="m-lbl">Acurácia</span>' +
          '<span class="m-val">' + (clf.accuracy * 100).toFixed(1) + "%</span></div>" +
        '<div class="m-item"><span class="m-lbl">F1 Macro</span>' +
          '<span class="m-val">' + (clf.f1_macro * 100).toFixed(1) + "%</span></div>" +
        '<div class="m-item"><span class="m-lbl">Precisão</span>' +
          '<span class="m-val">' + (clf.precision_macro * 100).toFixed(1) + "%</span></div>" +
      "</div>" +
      '<table class="mtable" style="margin-top:.5rem">' +
        "<thead><tr><th>Métrica</th><th>Macro avg</th></tr></thead>" +
        "<tbody>" +
          "<tr><td>Recall</td><td>" + (clf.recall_macro * 100).toFixed(1) + "%</td></tr>" +
        "</tbody>" +
      "</table>" +
      '<p class="modal-section">Regressão (Taxa de Juros)</p>' +
      '<div class="mmetrics-grid">' +
        '<div class="m-item"><span class="m-lbl">R²</span>' +
          '<span class="m-val">' + reg.r2.toFixed(4) + "</span></div>" +
        '<div class="m-item"><span class="m-lbl">MAE</span>' +
          '<span class="m-val">' + reg.mae.toFixed(4) + "</span></div>" +
      "</div>";
  }
})();