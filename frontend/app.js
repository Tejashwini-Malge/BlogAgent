/* ─── Blog Agent · Frontend ──────────────────────────────────────────────── */

const AGENTS = ['researcher', 'writer', 'editor'];

const AGENT_META = {
  researcher: { icon: '🔍', heading: 'Researcher is working', sub: 'Building a structured research brief' },
  writer:     { icon: '✏️', heading: 'Writer is drafting',    sub: 'Turning the brief into a blog post'  },
  editor:     { icon: '📝', heading: 'Editor is polishing',   sub: 'Reviewing grammar, flow and hooks'   },
};

const MILESTONES = { researcher: 10, writer: 48, editor: 82, done: 100 };

/* ── DOM ────────────────────────────────────────────────────────────────── */
const $ = id => document.getElementById(id);
const topicInput      = $('topicInput');
const generateBtn     = $('generateBtn');
const btnLabel        = $('btnLabel');
const formHint        = $('formHint');
const styleToggleBtn  = $('styleToggleBtn');
const stylePanel      = $('stylePanel');
const styleNotes      = $('styleNotes');
const progressRow     = $('progressRow');
const progressFill    = $('progressFill');
const progressLabel   = $('progressLabel');
const logsSection     = $('logsSection');
const logsPanel       = $('logsPanel');
const logsLive        = $('logsLive');
const clearBtn        = $('clearBtn');
const collapseBtn     = $('collapseBtn');
const resultSection   = $('resultSection');
const proseContent    = $('proseContent');
const rawTextarea     = $('rawTextarea');
const wordCount       = $('wordCount');
const copyBtn         = $('copyBtn');
const copyLabel       = $('copyLabel');
const tabPreview      = $('tabPreview');
const tabRaw          = $('tabRaw');
const tabIndicator    = $('tabIndicator');
const panelPreview    = $('panelPreview');
const panelRaw        = $('panelRaw');
const pipelineDetail  = $('pipelineDetail');
const pdetailIcon     = $('pdetailIcon');
const pdetailHeading  = $('pdetailHeading');
const pdetailSub      = $('pdetailSub');
const pdetailTimer    = $('pdetailTimer');
const toast           = $('toast');
const toastIcon       = $('toastIcon');
const toastTitle      = $('toastTitle');
const toastMsg        = $('toastMsg');
const toastClose      = $('toastClose');

/* ── State ──────────────────────────────────────────────────────────────── */
let currentResult = '';
let activeEs      = null;
let isRunning     = false;   /* FIX: explicit run flag — guards Enter-key re-entry */
let activeTab     = 'preview';
let progressPct   = 0;
let progressCrawl = null;
let agentTimers   = {};
let agentSeconds  = {};
let toastTimer    = null;

/* ── Style Settings ─────────────────────────────────────────────────────── */
const STYLE_DEFAULTS = { tone: 'professional', length: 'medium', audience: 'general', notes: '' };
let agentStyle = { ...STYLE_DEFAULTS };

function loadStyle() {
  try {
    const saved = JSON.parse(localStorage.getItem('blogAgentStyle') || '{}');
    agentStyle = { ...STYLE_DEFAULTS, ...saved };
  } catch {}
  syncStyleUI();
}

function saveStyle() {
  try { localStorage.setItem('blogAgentStyle', JSON.stringify(agentStyle)); } catch {}
}

function syncStyleUI() {
  document.querySelectorAll('.style-pills[data-pref]').forEach(group => {
    const pref = group.dataset.pref;
    group.querySelectorAll('.style-pill').forEach(pill => {
      pill.classList.toggle('is-active', pill.dataset.value === agentStyle[pref]);
    });
  });
  if (styleNotes) styleNotes.value = agentStyle.notes || '';
}

/* Pill group click handler */
document.querySelectorAll('.style-pills[data-pref]').forEach(group => {
  group.addEventListener('click', e => {
    const pill = e.target.closest('.style-pill');
    if (!pill) return;
    const pref = group.dataset.pref;
    agentStyle[pref] = pill.dataset.value;
    group.querySelectorAll('.style-pill').forEach(p => p.classList.toggle('is-active', p === pill));
    saveStyle();
  });
});

if (styleNotes) {
  styleNotes.addEventListener('input', () => {
    agentStyle.notes = styleNotes.value;
    saveStyle();
  });
}

/* Style panel toggle */
if (styleToggleBtn && stylePanel) {
  styleToggleBtn.addEventListener('click', () => {
    const isOpen = stylePanel.classList.toggle('is-open');
    styleToggleBtn.classList.toggle('is-open', isOpen);
    const labelSpan = styleToggleBtn.querySelector('span');
    if (labelSpan) labelSpan.textContent = isOpen ? 'Done' : 'Style';
  });
}

/* ── Progress ───────────────────────────────────────────────────────────── */
function setProgress(pct, label) {
  progressPct = Math.max(progressPct, pct);
  progressFill.style.width = progressPct + '%';
  if (label) progressLabel.textContent = label;
}

function startCrawl(ceiling) {  /* FIX: dead `floor` param removed */
  stopCrawl();
  progressCrawl = setInterval(() => {
    if (progressPct < ceiling - 3) {
      progressPct = Math.min(progressPct + .35, ceiling - 3);
      progressFill.style.width = progressPct + '%';
    }
  }, 1600);
}

function stopCrawl() {
  clearInterval(progressCrawl);
  progressCrawl = null;
}

/* ── Agent timers ───────────────────────────────────────────────────────── */
function startTimer(agent) {
  agentSeconds[agent] = 0;
  updateTimerDOM(agent);
  agentTimers[agent] = setInterval(() => {
    agentSeconds[agent]++;
    updateTimerDOM(agent);
    pdetailTimer.textContent = formatTime(agentSeconds[agent]);
  }, 1000);
}

function stopTimer(agent) {
  clearInterval(agentTimers[agent]);
  delete agentTimers[agent];
  updateTimerDOM(agent);
}

function updateTimerDOM(agent) {
  const el = document.querySelector(`#pstep-${agent} .pipeline-step__timer`);
  if (el) el.textContent = formatTime(agentSeconds[agent] || 0);
}

function formatTime(s) {
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

function stopAllTimers() { AGENTS.forEach(stopTimer); }

/* ── Pipeline ───────────────────────────────────────────────────────────── */
function setStepState(agent, state) {
  const step = $(`pstep-${agent}`);
  if (!step) return;
  step.classList.remove('is-active', 'is-done');
  if (state !== 'idle') step.classList.add(`is-${state}`);

  if (state === 'active') {
    const meta = AGENT_META[agent];
    pdetailIcon.textContent    = meta.icon;
    pdetailHeading.textContent = meta.heading;
    pdetailSub.textContent     = meta.sub;
    pdetailTimer.textContent   = '0:00';
    pipelineDetail.classList.add('is-open');
    startTimer(agent);
  }
  if (state === 'done') stopTimer(agent);
}

function fillEdge(idx) {
  const fill = $(`pedge-fill-${idx}`);
  if (fill) fill.classList.add('is-filled');
}

function resetPipeline() {
  AGENTS.forEach(a => {
    const step = $(`pstep-${a}`);
    if (step) step.classList.remove('is-active', 'is-done');
    let timerEl = document.querySelector(`#pstep-${a} .pipeline-step__timer`);
    if (!timerEl) {
      timerEl = document.createElement('span');
      timerEl.className = 'pipeline-step__timer';
      $(`pstep-${a}`)?.querySelector('.pipeline-step__label')?.appendChild(timerEl);
    }
    timerEl.textContent = '';
  });
  [1, 2].forEach(i => {
    const fill = $(`pedge-fill-${i}`);
    if (fill) fill.classList.remove('is-filled');
  });
  pipelineDetail.classList.remove('is-open');
  agentSeconds = {};
}

/* ── Tab switcher ───────────────────────────────────────────────────────── */
function positionIndicator() {
  /* FIX: bail out while the result section is hidden — getBoundingClientRect
     returns all zeros on display:none elements, which used to leave the tab
     indicator at width 0 until the next window resize. */
  if (!resultSection.classList.contains('is-visible')) return;
  const activeBtn = activeTab === 'preview' ? tabPreview : tabRaw;
  const tabs = activeBtn.closest('.tabs');
  if (!tabs) return;
  const tRect = tabs.getBoundingClientRect();
  const bRect = activeBtn.getBoundingClientRect();
  tabIndicator.style.left  = (bRect.left - tRect.left - 3) + 'px';
  tabIndicator.style.width = bRect.width + 'px';
}

function switchTab(tab) {
  activeTab = tab;
  const isPreview = tab === 'preview';
  tabPreview.classList.toggle('is-active', isPreview);
  tabRaw.classList.toggle('is-active', !isPreview);
  tabPreview.setAttribute('aria-selected', String(isPreview));
  tabRaw.setAttribute('aria-selected', String(!isPreview));
  panelPreview.classList.toggle('is-active', isPreview);
  panelRaw.classList.toggle('is-active', !isPreview);
  copyLabel.textContent = isPreview ? 'Copy' : 'Copy Markdown';
  requestAnimationFrame(positionIndicator);
}

tabPreview.addEventListener('click', () => switchTab('preview'));
tabRaw.addEventListener('click',     () => switchTab('raw'));

window.addEventListener('load', () => {
  AGENTS.forEach(a => {
    const labelEl = document.querySelector(`#pstep-${a} .pipeline-step__label`);
    if (labelEl && !labelEl.querySelector('.pipeline-step__timer')) {
      const t = document.createElement('span');
      t.className = 'pipeline-step__timer';
      labelEl.appendChild(t);
    }
  });
  loadStyle();
});
window.addEventListener('resize', positionIndicator);

/* ── Log ────────────────────────────────────────────────────────────────── */
function appendLog(agent, message) {
  const LABELS = { researcher: 'RES', writer: 'WRT', editor: 'EDT', system: 'SYS' };
  const line = document.createElement('div');
  line.className = 'log-line';
  line.dataset.agent = agent || 'system';

  const tag = document.createElement('span');
  tag.className = 'log-line__tag';
  tag.textContent = LABELS[agent] ?? 'SYS';

  const text = document.createElement('span');
  text.className = 'log-line__text';
  text.textContent = message;

  line.append(tag, text);
  logsPanel.appendChild(line);
  logsPanel.scrollTop = logsPanel.scrollHeight;
}

clearBtn.addEventListener('click', () => { logsPanel.innerHTML = ''; });
collapseBtn.addEventListener('click', () => {
  logsSection.classList.toggle('is-collapsed');
});

/* ── Toast ──────────────────────────────────────────────────────────────── */
function showToast(type, title, message, duration = 6000) {
  clearTimeout(toastTimer);
  toast.classList.remove('is-visible', 'is-hiding', 'toast--error', 'toast--success');
  toastIcon.textContent  = type === 'success' ? '✅' : '⚠️';
  toastTitle.textContent = title;
  toastMsg.textContent   = message;
  toast.classList.add(`toast--${type}`, 'is-visible');
  if (duration > 0) toastTimer = setTimeout(hideToast, duration);
}
function hideToast() {
  toast.classList.add('is-hiding');
  toast.addEventListener('animationend', () => toast.classList.remove('is-visible', 'is-hiding'), { once: true });
}
toastClose.addEventListener('click', hideToast);

/* ── Input validation ───────────────────────────────────────────────────── */
let hintTimer = null;
function showHint(msg) {
  topicInput.classList.add('is-invalid');
  formHint.textContent = msg;
  clearTimeout(hintTimer);
  hintTimer = setTimeout(clearHint, 3000);
}
function clearHint() {
  topicInput.classList.remove('is-invalid');
  formHint.textContent = '';
}
topicInput.addEventListener('input', () => {
  if (topicInput.classList.contains('is-invalid')) clearHint();
});

/* ── Copy ───────────────────────────────────────────────────────────────── */
copyBtn.addEventListener('click', () => {
  if (!currentResult) return;
  const text = activeTab === 'raw' ? currentResult : (proseContent.innerText || currentResult);
  navigator.clipboard.writeText(text).then(() => {
    const orig = copyLabel.textContent;
    copyLabel.textContent = 'Copied!';
    setTimeout(() => (copyLabel.textContent = orig), 2000);
  });
});

/* ── Reset ──────────────────────────────────────────────────────────────── */
function resetUI() {
  stopAllTimers(); stopCrawl();
  resetPipeline();
  logsPanel.innerHTML = '';
  progressPct = 0;
  progressFill.style.width = '0%';
  progressLabel.textContent = '';
  progressRow.classList.remove('is-visible');
  logsSection.classList.remove('is-visible', 'is-collapsed');
  resultSection.classList.remove('is-visible');
  proseContent.innerHTML = '';
  rawTextarea.value = '';
  wordCount.textContent = '';
  currentResult = '';
  switchTab('preview');
  logsLive.style.display = '';
}

/* FIX: shared cleanup so every exit path (done / error / connection loss)
   restores the button and run-state identically — no half-stuck states. */
function endRun(label) {
  isRunning = false;
  stopCrawl(); stopAllTimers();
  activeEs?.close();
  activeEs = null;
  generateBtn.disabled = false;
  generateBtn.classList.remove('is-loading');
  btnLabel.textContent = label;
}

/* ── Helpers ────────────────────────────────────────────────────────────── */
/* FIX: the old non-marked fallback injected raw LLM output into innerHTML
   unescaped — any literal <tag> in the post broke the page (or worse). */
function escapeHtml(s) {
  return s.replace(/[&<>"']/g, c => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}

/* ── SSE ────────────────────────────────────────────────────────────────── */
function handleEvent(raw) {
  let e; try { e = JSON.parse(raw); } catch { return; }

  switch (e.type) {

    case 'start':
      appendLog('system', `Starting crew · topic: "${e.topic}"`);
      progressRow.classList.add('is-visible');
      logsSection.classList.add('is-visible');
      setProgress(8, 'Assembling crew…');
      setStepState('researcher', 'active');
      setProgress(MILESTONES.researcher, 'Researcher working…');
      startCrawl(MILESTONES.writer);
      break;

    case 'agent_active': {
      if (!e.agent) break;
      const ORDER = { researcher: 0, writer: 1, editor: 2 };
      const idx = ORDER[e.agent] ?? 0;
      stopCrawl();
      AGENTS.slice(0, idx).forEach((a, i) => { setStepState(a, 'done'); fillEdge(i + 1); });
      setStepState(e.agent, 'active');
      const next = AGENTS[idx + 1];
      setProgress(MILESTONES[e.agent], `${e.agent.charAt(0).toUpperCase() + e.agent.slice(1)} working…`);
      startCrawl(next ? MILESTONES[next] : MILESTONES.done);
      break;
    }

    case 'log':
      if (e.message?.trim()) appendLog(e.agent || 'system', e.message.trim());
      break;

    case 'final': {
      currentResult = e.content || '';
      stopCrawl(); stopAllTimers();
      AGENTS.forEach((a, i) => { setStepState(a, 'done'); if (i < 2) fillEdge(i + 1); });
      pipelineDetail.classList.remove('is-open');
      setProgress(100, 'Complete');
      logsLive.style.display = 'none';

      const html = window.marked
        ? marked.parse(currentResult)
        : `<pre>${escapeHtml(currentResult)}</pre>`;  /* FIX: escaped fallback */
      proseContent.innerHTML = html;
      rawTextarea.value = currentResult;
      const wc = currentResult.trim().split(/\s+/).filter(Boolean).length;
      wordCount.textContent = `~${wc} words`;

      resultSection.classList.add('is-visible');
      /* FIX: position the tab indicator now that the section is actually
         rendered — previously it was measured while display:none and showed
         up with zero width. */
      requestAnimationFrame(positionIndicator);
      setTimeout(() => resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' }), 80);
      appendLog('system', `Saved → ${e.saved_to ?? 'output/'}`);
      showToast('success', 'Blog post ready', `${wc} words · saved to ${e.saved_to ?? 'output/'}`, 5000);
      break;
    }

    case 'error':
      appendLog('system', `Error: ${e.message}`);
      showToast('error', 'Generation failed', e.message || 'Something went wrong.');
      endRun('Try again');
      break;

    case 'done':
      /* Critical: close the EventSource so it doesn't auto-reconnect
         and silently restart the whole crew run. */
      if (isRunning) endRun('Generate again');
      break;
  }
}

/* ── Generate ───────────────────────────────────────────────────────────── */
function generate() {
  /* FIX: re-entry guard — the Enter key called generate() directly and
     bypassed the disabled button, killing the stream and launching a second
     crew run server-side while the first kept burning API credits. */
  if (isRunning) return;

  const topic = topicInput.value.trim();
  if (!topic) { showHint('Please enter a topic first.'); topicInput.focus(); return; }

  activeEs?.close();
  activeEs = null;
  resetUI();
  isRunning = true;
  generateBtn.disabled = true;
  generateBtn.classList.add('is-loading');
  btnLabel.textContent = 'Running…';

  const params = new URLSearchParams({
    topic,
    tone:     agentStyle.tone,
    length:   agentStyle.length,
    audience: agentStyle.audience,
    notes:    agentStyle.notes || '',
  });

  activeEs = new EventSource(`/api/generate?${params}`);
  activeEs.onmessage = ev => handleEvent(ev.data);
  activeEs.onerror   = () => {
    /* FIX: ignore the error EventSource fires after WE close the stream
       (post-done/error cleanup) — it used to flash a bogus
       "Connection lost" toast on perfectly successful runs. */
    if (!isRunning) return;
    showToast('error', 'Connection lost', 'The stream was interrupted. Please try again.');
    endRun('Generate');
  };
}

generateBtn.addEventListener('click', generate);
topicInput.addEventListener('keydown', e => { if (e.key === 'Enter') generate(); });