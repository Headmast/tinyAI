const statusEls = {
  lastUpdated: document.getElementById('last-updated'),
  serviceStatus: document.getElementById('service-status'),
  serviceModel: document.getElementById('service-model'),
  totalRequests: document.getElementById('total-requests'),
  requestBreakdown: document.getElementById('request-breakdown'),
  totalTokens: document.getElementById('total-tokens'),
  tokenBreakdown: document.getElementById('token-breakdown'),
  activeRequests: document.getElementById('active-requests'),
  limits: document.getElementById('limits'),
  rateLimitHits: document.getElementById('rate-limit-hits'),
  badRequestErrors: document.getElementById('bad-request-errors'),
  upstreamErrors: document.getElementById('upstream-errors'),
};

const historyListEl = document.getElementById('history-list');
const historyLimitEl = document.getElementById('history-limit');

function fmtNum(v) {
  return new Intl.NumberFormat('ru-RU').format(v || 0);
}

function escapeHtml(text) {
  return String(text || '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;');
}

async function fetchJson(url) {
  const r = await fetch(url, { cache: 'no-store' });
  if (!r.ok) {
    throw new Error(`${url} -> ${r.status}`);
  }
  return r.json();
}

function renderStatus(status, metrics) {
  statusEls.serviceStatus.textContent = status.status || '-';
  statusEls.serviceModel.textContent = `model: ${status.model || '-'}`;

  statusEls.totalRequests.textContent = fmtNum(metrics.total_requests);
  statusEls.requestBreakdown.textContent = `ok: ${fmtNum(metrics.successful_requests)} / fail: ${fmtNum(metrics.failed_requests)}`;

  statusEls.totalTokens.textContent = fmtNum(metrics.total_tokens);
  statusEls.tokenBreakdown.textContent = `prompt: ${fmtNum(metrics.total_prompt_tokens)} / completion: ${fmtNum(metrics.total_completion_tokens)}`;

  statusEls.activeRequests.textContent = fmtNum(status.active_requests);
  const limits = status.limits || {};
  statusEls.limits.textContent = `limits: rpm ${limits.max_requests_per_minute ?? '-'} | ctx ${limits.max_context_tokens ?? '-'} | out ${limits.max_completion_tokens ?? '-'}`;

  statusEls.rateLimitHits.textContent = fmtNum(metrics.rate_limit_hits);
  statusEls.badRequestErrors.textContent = fmtNum(metrics.bad_request_errors);
  statusEls.upstreamErrors.textContent = fmtNum(metrics.upstream_errors);

  statusEls.lastUpdated.textContent = `Обновлено: ${new Date().toLocaleTimeString('ru-RU')}`;
}

function renderHistory(history) {
  const items = history.items || [];
  if (!items.length) {
    historyListEl.innerHTML = '<div class="muted">Пока нет запросов в истории сервиса.</div>';
    return;
  }

  historyListEl.innerHTML = items.map((item) => {
    const ok = item.success;
    const badgeClass = ok ? 'ok' : 'fail';
    const badgeText = ok ? 'OK' : 'FAIL';
    return `
      <article class="history-item">
        <div class="history-meta">
          <span class="badge ${badgeClass}">${badgeText}</span>
          <strong>${escapeHtml(item.endpoint || '-')}</strong>
          <span>${escapeHtml(item.timestamp || '-')}</span>
          <span>latency: ${escapeHtml(item.latency_ms)}ms</span>
          <span>tokens: ${escapeHtml(item.total_tokens)}</span>
          <span>code: ${escapeHtml(item.status_code)}</span>
        </div>
        <p class="snippet"><span>Request:</span> ${escapeHtml(item.prompt_preview)}</p>
        <p class="snippet"><span>Response:</span> ${escapeHtml(item.response_preview)}</p>
        ${item.error ? `<p class="snippet"><span>Error:</span> ${escapeHtml(item.error)}</p>` : ''}
      </article>
    `;
  }).join('');
}

async function refresh() {
  try {
    const limit = encodeURIComponent(historyLimitEl.value || '50');
    const [status, metrics, history] = await Promise.all([
      fetchJson('/dashboard/api/status'),
      fetchJson('/dashboard/api/metrics'),
      fetchJson(`/dashboard/api/history?limit=${limit}`),
    ]);
    renderStatus(status, metrics);
    renderHistory(history);
  } catch (err) {
    statusEls.lastUpdated.textContent = `Ошибка обновления: ${String(err)}`;
  }
}

historyLimitEl.addEventListener('change', () => {
  refresh();
});

refresh();
setInterval(refresh, 3000);
