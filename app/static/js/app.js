const API_BASE = "/api/v1";

function formatDriftStatus(detected) {
    return detected
        ? '<span class="status-alert">Обнаружен</span>'
        : '<span class="status-ok">Норма</span>';
}

function updateDriftBanner(status) {
    const banner = document.getElementById("drift-banner");
    if (!banner) return;

    if (!status) {
        banner.classList.add("hidden");
        return;
    }

    if (status.any_drift_detected) {
        banner.classList.remove("hidden");
        banner.classList.remove("warning");
        banner.textContent = status.notifications.join(" | ") || "Обнаружен дрейф данных";
    } else if (status.notifications && status.notifications.length) {
        banner.classList.remove("hidden");
        banner.classList.add("warning");
        banner.textContent = status.notifications.join(" | ");
    } else {
        banner.classList.add("hidden");
    }
}

function renderDriftCard(prefix, drift) {
    const statusEl = document.getElementById(`${prefix}-drift-status`);
    const scoreEl = document.getElementById(`${prefix}-drift-score`);
    if (!statusEl || !scoreEl || !drift) return;
    statusEl.innerHTML = formatDriftStatus(drift.detected);
    scoreEl.textContent = `score: ${drift.score.toFixed(3)} — ${drift.details}`;
}

async function loadDriftStatus() {
    try {
        const response = await fetch(`${API_BASE}/drift/status`);
        if (!response.ok) return;
        const status = await response.json();
        updateDriftBanner(status);
        renderDriftCard("data", status.data_drift);
        renderDriftCard("target", status.target_drift);
        renderDriftCard("concept", status.concept_drift);

        const notifications = document.getElementById("drift-notifications");
        if (notifications) {
            notifications.innerHTML = (status.notifications || [])
                .map((msg) => `<div class="alert">${msg}</div>`)
                .join("");
        }

        const reportLink = document.getElementById("latest-report-link");
        if (reportLink && status.report_path) {
            const filename = status.report_path.split(/[/\\]/).pop();
            reportLink.href = `${API_BASE}/drift/reports/${filename}`;
            reportLink.classList.remove("hidden");
        }
    } catch (err) {
        console.debug("Drift status not available yet", err);
    }
}

async function checkDrift() {
    const btn = document.getElementById("btn-check-drift");
    if (btn) btn.disabled = true;
    try {
        const response = await fetch(`${API_BASE}/drift/check`, { method: "POST" });
        const status = await response.json();
        updateDriftBanner(status);
        renderDriftCard("data", status.data_drift);
        renderDriftCard("target", status.target_drift);
        renderDriftCard("concept", status.concept_drift);

        const notifications = document.getElementById("drift-notifications");
        if (notifications) {
            notifications.innerHTML = (status.notifications || [])
                .map((msg) => `<div class="alert">${msg}</div>`)
                .join("");
        }

        const reportLink = document.getElementById("latest-report-link");
        if (reportLink && status.report_path) {
            const filename = status.report_path.split(/[/\\]/).pop();
            reportLink.href = `${API_BASE}/drift/reports/${filename}`;
            reportLink.classList.remove("hidden");
        }
    } finally {
        if (btn) btn.disabled = false;
    }
}

async function loadPredictions(limit = 50) {
    try {
        const response = await fetch(`${API_BASE}/predictions?limit=${limit}`);
        const data = await response.json();

        const countEl = document.getElementById("predictions-count");
        if (countEl) countEl.textContent = data.total;

        const bodies = [
            document.getElementById("recent-predictions-body"),
            document.getElementById("predictions-table-body"),
        ].filter(Boolean);

        const rows = data.items.map((item) => `
            <tr>
                <td>${item.id}</td>
                ${bodies[1] ? `<td>${new Date(item.created_at).toLocaleString("ru-RU")}</td>` : ""}
                <td class="truncate" title="${escapeHtml(item.text)}">${escapeHtml(item.text)}</td>
                <td>${item.rating}</td>
                ${bodies[1] ? `<td>${item.true_rating ?? "—"}</td>` : ""}
                <td>${(item.confidence * 100).toFixed(1)}%</td>
                <td>${item.sentiment}</td>
                <td>${item.is_anomaly ? '<span class="anomaly-badge">да</span>' : "нет"}</td>
            </tr>
        `).join("");

        bodies.forEach((body) => { body.innerHTML = rows || "<tr><td colspan='8'>Нет данных</td></tr>"; });
    } catch (err) {
        console.error("Failed to load predictions", err);
    }
}

async function startRetrain() {
    const btn = document.getElementById("btn-retrain");
    if (btn) btn.disabled = true;
    try {
        const response = await fetch(`${API_BASE}/retrain`, { method: "POST" });
        const data = await response.json();
        const statusEl = document.getElementById("retrain-status");
        if (statusEl) statusEl.textContent = `${data.status}: ${data.message}`;
    } finally {
        if (btn) btn.disabled = false;
        loadRetrainStatus();
    }
}

async function loadRetrainStatus() {
    try {
        const response = await fetch(`${API_BASE}/retrain/status`);
        const data = await response.json();
        const statusEl = document.getElementById("retrain-status");
        if (statusEl) statusEl.textContent = `Переобучение: ${data.status} — ${data.message}`;
    } catch (err) {
        console.debug("Retrain status unavailable", err);
    }
}

async function submitInference(event) {
    event.preventDefault();
    const text = document.getElementById("review-text").value.trim();
    const trueRating = document.getElementById("true-rating").value;
    const payload = { text };
    if (trueRating) payload.true_rating = parseInt(trueRating, 10);

    const response = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });
    const data = await response.json();

    const resultBox = document.getElementById("prediction-result");
    resultBox.classList.remove("hidden");
    document.getElementById("result-rating").textContent = data.rating;
    document.getElementById("result-sentiment").textContent = data.sentiment;
    document.getElementById("result-confidence").textContent = `${(data.confidence * 100).toFixed(1)}%`;

    const anomalyEl = document.getElementById("result-anomaly");
    if (data.is_anomaly) anomalyEl.classList.remove("hidden");
    else anomalyEl.classList.add("hidden");
}

async function loadExperiments() {
    const body = document.getElementById("experiments-table-body");
    if (!body) return;

    try {
        const response = await fetch(`${API_BASE}/experiments`);
        const data = await response.json();
        body.innerHTML = data.runs.map((run) => `
            <tr>
                <td><code>${run.run_id.slice(0, 8)}</code></td>
                <td>${escapeHtml(run.run_name)}</td>
                <td>${run.status}</td>
                <td>${run.accuracy != null ? run.accuracy.toFixed(4) : "—"}</td>
                <td><a href="/" onclick="return false;">см. MLflow UI</a></td>
            </tr>
        `).join("") || "<tr><td colspan='5'>Нет экспериментов или MLflow недоступен</td></tr>";
    } catch (err) {
        body.innerHTML = "<tr><td colspan='5'>Ошибка загрузки экспериментов</td></tr>";
    }
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

// Drift banner on all pages
document.addEventListener("DOMContentLoaded", () => {
    loadDriftStatus();
    setInterval(loadDriftStatus, 60000);
});
