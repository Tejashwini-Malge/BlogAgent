/* ─── Blog Agent · Frontend v7 — 90s Typewriter Edition ──────────────────── */

const AGENTS = ['researcher', 'writer', 'editor'];

const AGENT_META = {
  researcher: { icon: '🔍', heading: 'Researcher is working',  sub: 'Building a structured research brief',   phase: 'researching', btnLabel: 'Researching…' },
  writer:     { icon: '✏️',  heading: 'Writer is drafting',     sub: 'Turning the brief into a blog post',    phase: 'writing',     btnLabel: 'Writing…'     },
  editor:     { icon: '📝', heading: 'Editor is polishing',    sub: 'Reviewing grammar, flow and hooks',     phase: 'editing',     btnLabel: 'Editing…'     },
};

const MILESTONES = { researcher: 10, writer: 48, editor: 82, done: 100 };

/* ── Topic suggestion pool ──────────────────────────────────────────────── */
const TOPIC_POOL = [
  '⚡ Why async/await beats callbacks in 2025',
  '🌿 The hidden cost of premature optimization',
  '⚙ Building a second brain with AI tools',
  '💬 How to write a developer blog that actually gets read',
  '⚡ The future of remote work after AI automation',
  '🌿 Why your morning routine is sabotaging your creativity',
  '⚙ From idea to production: deploying a FastAPI app in 30 minutes',
  '💬 The case for boring technology in startups',
  '⚡ How AI agents are changing knowledge work',
  '🌿 Deep work in a world of constant notifications',
  '⚙ What I learned shipping 10 side projects',
  '💬 Why most developer documentation fails (and how to fix it)',
  '⚡ The rise of ambient computing',
  '🌿 How to build a personal brand as an engineer',
  '⚙ The art of the minimal viable product',
  '💬 Open source sustainability: who pays for free software?',
  '⚡ LLMs as reasoning engines, not just text generators',
  '🌿 Why slow is smooth and smooth is fast',
  '⚙ The underrated power of plain-text systems',
  '💬 Building products that people actually want to use',
];

/* ── DOM ────────────────────────────────────────────────────────────────── */
const $ = id => document.getElementById(id);
const topicInput      = $('topicInput');
const generateBtn     = $('generateBtn');
const btnLabel        = $('btnLabel');
const btnSpinner      = $('btnSpinner');
const btnArrow        = $('btnArrow');
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
const chipsRow        = $('chipsRow');
const historyBtn      = $('historyBtn');
const historyBadge    = $('historyBadge');
const historyDrawer   = $('historyDrawer');
const drawerOverlay   = $('drawerOverlay');
const drawerClose     = $('drawerClose');
const drawerBody      = $('drawerBody');
const exportWrap      = $('exportWrap');
const exportToggle    = $('exportToggle');
const exportMenu      = $('exportMenu');
const vignettePulse   = $('vignettePulse');
const soundBtn        = $('soundBtn');
const metricsPanel    = $('metricsPanel');
const metricsPanelBody= $('metricsPanelBody');
const metricsBadge    = $('metricsBadge');

/* ── State ──────────────────────────────────────────────────────────────── */
let currentResult = '';
let currentTopic  = '';
let activeEs      = null;
let isRunning     = false;
let activeTab     = 'preview';
let progressPct   = 0;
let progressCrawl = null;
let agentTimers   = {};
let agentSeconds  = {};
let toastTimer    = null;
let drawerOpen    = false;
let activeAgent   = null;
let metricsData   = {}; // { agentName: { iters: [{...},...], skipped: false, skipReason: '' } }

/* ── Metric labels (mirrors src/metrics.py METRIC_LABELS) ──────────────── */
const METRIC_LABELS_JS = {
  word_count:       'Words',
  section_count:    'Sections',
  bullet_count:     'Bullets',
  has_caveats:      'Has caveats',
  h2_count:         'H2 headers',
  avg_sentence_len: 'Avg sent len',
  hook_score:       'Hook strength',
  transition_count: 'Transitions',
  passive_count:    'Passive voice',
};

/* ── Metrics panel ──────────────────────────────────────────────────────── */
function resetMetricsPanel() {
  metricsData = {};
  metricsPanel?.classList.remove('is-visible');
  if (metricsPanelBody) metricsPanelBody.innerHTML = '';
  if (metricsBadge)     metricsBadge.textContent   = '';
}

function renderAgentMetrics(agent) {
  const d = metricsData[agent];
  if (!d || !d.iters.length) return;
  metricsPanel?.classList.add('is-visible');

  let card = $(`mcard-${agent}`);
  if (!card) {
    card = document.createElement('div');
    card.id        = `mcard-${agent}`;
    card.className = 'metrics-card';
    metricsPanelBody?.appendChild(card);
  }

  const baseline    = d.iters[0];
  const latest      = d.iters[d.iters.length - 1];
  const hasRevision = d.iters.length > 1;
  const status      = d.skipped ? 'skipped' : hasRevision ? 'improved' : 'running';
  const statusLabel = d.skipped ? 'no change' : hasRevision ? `+${d.iters.length - 1} rev` : 'measuring…';

  const keys       = Object.keys(baseline);
  const headerAfter = hasRevision ? '<th>After</th>' : '';
  const rows = keys.map(k => {
    const b    = baseline[k];
    const a    = latest[k];
    const diff = typeof b === 'number' ? +(a - b).toFixed(1) : 0;
    let delta  = '';
    if (hasRevision) {
      if (diff > 0)      delta = `<span class="metric-delta metric-delta--up">▲${diff}</span>`;
      else if (diff < 0) delta = `<span class="metric-delta metric-delta--down">▼${Math.abs(diff)}</span>`;
      else               delta = `<span class="metric-delta metric-delta--same">—</span>`;
    }
    const afterTd = hasRevision ? `<td>${a}${delta}</td>` : '';
    return `<tr><td>${METRIC_LABELS_JS[k] || k}</td><td>${b}</td>${afterTd}</tr>`;
  }).join('');

  card.innerHTML = `
    <div class="metrics-card__head">
      <span class="metrics-card__agent metrics-card__agent--${agent}">${agent}</span>
      <span class="metrics-card__status metrics-card__status--${status}">${statusLabel}</span>
    </div>
    <table class="metrics-card__table">
      <thead><tr><th>Metric</th><th>Before</th>${headerAfter}</tr></thead>
      <tbody>${rows}</tbody>
    </table>
    ${d.skipped ? `<div class="metrics-card__skip">⊘ ${d.skipReason}</div>` : ''}
  `;

  const totalRevisions = Object.values(metricsData)
    .reduce((n, x) => n + Math.max(x.iters.length - 1, 0), 0);
  if (metricsBadge)
    metricsBadge.textContent = totalRevisions > 0
      ? `${totalRevisions} revision${totalRevisions !== 1 ? 's' : ''}`
      : '';
}

/* ── Typewriter Sounds ──────────────────────────────────────────────────── */
let _audioCtx   = null;
let _soundOn    = true;

function getAudioCtx() {
  if (!_audioCtx) {
    _audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
  if (_audioCtx.state === 'suspended') _audioCtx.resume();
  return _audioCtx;
}

/* Warm up AudioContext on first user interaction */
document.addEventListener('pointerdown', () => { try { getAudioCtx(); } catch(e) {} }, { once: true });

function playKeyClick(vol = 0.12) {
  if (!_soundOn) return;
  try {
    const ctx = getAudioCtx();
    const dur = 0.038;
    const buf = ctx.createBuffer(1, Math.ceil(ctx.sampleRate * dur), ctx.sampleRate);
    const d   = buf.getChannelData(0);
    for (let i = 0; i < d.length; i++) {
      d[i] = (Math.random() * 2 - 1) * Math.exp(-i / (ctx.sampleRate * 0.006));
    }
    const src = ctx.createBufferSource();
    src.buffer = buf;
    const lpf = ctx.createBiquadFilter();
    lpf.type = 'lowpass'; lpf.frequency.value = 2800;
    const g = ctx.createGain();
    g.gain.value = vol;
    src.connect(lpf); lpf.connect(g); g.connect(ctx.destination);
    src.start();
  } catch(e) {}
}

function playCarriageReturn() {
  if (!_soundOn) return;
  try {
    const ctx = getAudioCtx();
    const dur = 0.28;
    const buf = ctx.createBuffer(1, Math.ceil(ctx.sampleRate * dur), ctx.sampleRate);
    const d   = buf.getChannelData(0);
    for (let i = 0; i < d.length; i++) {
      const t = i / ctx.sampleRate;
      d[i] = ((Math.random() * 2 - 1) * 0.45 + Math.sin(t * 2 * Math.PI * 90) * 0.12)
             * Math.exp(-t / 0.09);
    }
    const src = ctx.createBufferSource();
    src.buffer = buf;
    const g = ctx.createGain();
    g.gain.value = 0.2;
    src.connect(g); g.connect(ctx.destination);
    src.start();
  } catch(e) {}
}

function playBell() {
  if (!_soundOn) return;
  try {
    const ctx = getAudioCtx();
    const now = ctx.currentTime;
    [880, 1108].forEach((freq, i) => {
      const osc = ctx.createOscillator();
      const g   = ctx.createGain();
      osc.type = 'sine';
      osc.frequency.value = freq;
      g.gain.setValueAtTime(i === 0 ? 0.22 : 0.09, now);
      g.gain.exponentialRampToValueAtTime(0.001, now + 2.0);
      osc.connect(g); g.connect(ctx.destination);
      osc.start(now); osc.stop(now + 2.0);
    });
  } catch(e) {}
}

/* Sound toggle */
if (soundBtn) {
  soundBtn.addEventListener('click', () => {
    _soundOn = !_soundOn;
    soundBtn.classList.toggle('is-muted', !_soundOn);
    soundBtn.title = _soundOn ? 'Mute sounds' : 'Unmute sounds';
    if (_soundOn) playKeyClick(0.1);
  });
}

/* ── Typewriter Text Reveal ─────────────────────────────────────────────── */
let _twTimer = null;

function clearTypewriterTimer() {
  if (_twTimer) { clearTimeout(_twTimer); _twTimer = null; }
}

function typewriterReveal(container, htmlContent, onComplete) {
  clearTypewriterTimer();
  container.innerHTML = htmlContent;

  /* Collect all non-empty text nodes */
  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT, null, false);
  const nodes  = [];
  let node;
  while ((node = walker.nextNode())) {
    if (node.textContent.trim()) nodes.push(node);
  }
  if (!nodes.length) { onComplete?.(); return; }

  const originals = nodes.map(n => n.textContent);
  nodes.forEach(n => { n.textContent = ''; });

  let ni = 0, ci = 0, clickAcc = 0;
  const CHARS = 5;   // characters revealed per tick
  const TICK  = 22;  // ms per tick

  function tick() {
    if (ni >= nodes.length) {
      playBell();
      onComplete?.();
      return;
    }
    for (let c = 0; c < CHARS; c++) {
      if (ni >= nodes.length) break;
      const orig = originals[ni];
      ci++;
      nodes[ni].textContent = orig.slice(0, ci);
      if (ci >= orig.length) { ni++; ci = 0; }
    }
    clickAcc += CHARS;
    if (clickAcc >= 8) { playKeyClick(0.07); clickAcc = 0; }
    _twTimer = setTimeout(tick, TICK);
  }
  tick();
}

/* ── Personalization Modal ──────────────────────────────────────────────── */
function initPersonalizeModal() {
  /* Show once per browser session */
  if (sessionStorage.getItem('tw_welcomed')) return;

  const overlay = $('personOverlay');
  if (!overlay) return;

  overlay.classList.add('is-open');

  $('personYes')?.addEventListener('click', () => {
    dismissPersonModal(overlay);
    /* Open style panel after modal fades */
    setTimeout(() => {
      stylePanel?.classList.add('is-open');
      styleToggleBtn?.classList.add('is-open');
      const lbl = styleToggleBtn?.querySelector('span');
      if (lbl) lbl.textContent = 'Done';
    }, 380);
    playKeyClick(0.15);
  });

  $('personSkip')?.addEventListener('click', () => {
    dismissPersonModal(overlay);
    setTimeout(() => topicInput?.focus(), 380);
    playKeyClick(0.1);
  });

  overlay.addEventListener('keydown', e => {
    if (e.key === 'Escape') {
      dismissPersonModal(overlay);
      playKeyClick(0.08);
    }
  });
}

function dismissPersonModal(overlay) {
  sessionStorage.setItem('tw_welcomed', '1');
  overlay.classList.add('is-closing');
  overlay.classList.remove('is-open');
  setTimeout(() => { overlay.style.display = 'none'; }, 380);
}

/* ── Style Settings ─────────────────────────────────────────────────────── */
const STYLE_DEFAULTS = { tone: 'professional', length: 'medium', audience: 'general', notes: '', critique: 'off' };
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

document.querySelectorAll('.style-pills[data-pref]').forEach(group => {
  group.addEventListener('click', e => {
    const pill = e.target.closest('.style-pill');
    if (!pill) return;
    playKeyClick(0.1);
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

if (styleToggleBtn && stylePanel) {
  styleToggleBtn.addEventListener('click', () => {
    const isOpen = stylePanel.classList.toggle('is-open');
    styleToggleBtn.classList.toggle('is-open', isOpen);
    const labelSpan = styleToggleBtn.querySelector('span');
    if (labelSpan) labelSpan.textContent = isOpen ? 'Done' : 'Style';
    playKeyClick(0.1);
  });
}

/* ── Topic Chips ────────────────────────────────────────────────────────── */
function initChips() {
  if (!chipsRow) return;
  const shuffled = [...TOPIC_POOL].sort(() => Math.random() - .5).slice(0, 6);

  shuffled.forEach((text, i) => {
    const btn = document.createElement('button');
    btn.className = 'chip';
    btn.type = 'button';
    btn.textContent = text;
    btn.style.animationDelay = `${i * 55}ms`;
    btn.addEventListener('click', () => {
      topicInput.value = text.replace(/^[⚡🌿⚙💬]\s*/, '');
      topicInput.focus();
      clearHint();
      playKeyClick(0.12);
    });
    chipsRow.appendChild(btn);
  });

  const surprise = document.createElement('button');
  surprise.className = 'chip chip--surprise';
  surprise.type = 'button';
  surprise.textContent = '🎲 Surprise me';
  surprise.style.animationDelay = `${shuffled.length * 55}ms`;
  surprise.addEventListener('click', () => {
    const pick = TOPIC_POOL[Math.floor(Math.random() * TOPIC_POOL.length)];
    topicInput.value = pick.replace(/^[⚡🌿⚙💬]\s*/, '');
    topicInput.focus();
    clearHint();
    playKeyClick(0.14);
  });
  chipsRow.appendChild(surprise);
}

/* ── Typewriter click on every key press in topic input ─────────────────── */
if (topicInput) {
  topicInput.addEventListener('keydown', e => {
    if (e.key.length === 1 || e.key === 'Backspace') playKeyClick(0.08);
  });
}

/* ── Page phase ─────────────────────────────────────────────────────────── */
function setPagePhase(phase) {
  document.body.classList.remove('phase--writing', 'phase--editing');
  if (phase === 'writing') document.body.classList.add('phase--writing');
  if (phase === 'editing') document.body.classList.add('phase--editing');
}

/* ── Particle burst on agent handoff ────────────────────────────────────── */
function fireParticles(agentId) {
  const node = $(`pnode-${agentId}`);
  if (!node) return;
  const rect = node.getBoundingClientRect();
  const cx = rect.left + rect.width / 2;
  const cy = rect.top  + rect.height / 2;
  const colors = ['#5DADA8', '#A4DAD6', '#22D473', '#D97026', '#F5A355'];
  const count  = 12;

  for (let i = 0; i < count; i++) {
    const angle = (i / count) * 2 * Math.PI;
    const dist  = 28 + Math.random() * 22;
    const bx    = Math.round(Math.cos(angle) * dist);
    const by    = Math.round(Math.sin(angle) * dist);
    const p     = document.createElement('span');
    p.className = 'particle';
    p.style.cssText = `left:${cx-2}px;top:${cy-2}px;background:${colors[i % colors.length]};--bx:${bx}px;--by:${by}px;`;
    document.body.appendChild(p);
    p.addEventListener('animationend', () => p.remove(), { once: true });
  }

  if (vignettePulse) {
    vignettePulse.style.display = 'block';
    vignettePulse.style.animation = 'none';
    requestAnimationFrame(() => {
      vignettePulse.style.animation = '';
      vignettePulse.addEventListener('animationend', () => {
        vignettePulse.style.display = 'none';
      }, { once: true });
    });
  }
}

/* ── Progress ───────────────────────────────────────────────────────────── */
function setProgress(pct, label) {
  progressPct = Math.max(progressPct, pct);
  progressFill.style.width = progressPct + '%';
  if (label) progressLabel.textContent = label;
}

function startCrawl(ceiling) {
  stopCrawl();
  progressCrawl = setInterval(() => {
    if (progressPct < ceiling - 3) {
      progressPct = Math.min(progressPct + .35, ceiling - 3);
      progressFill.style.width = progressPct + '%';
    }
  }, 1600);
}

function stopCrawl() { clearInterval(progressCrawl); progressCrawl = null; }

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
    activeAgent = agent;
    const meta = AGENT_META[agent];
    pdetailIcon.textContent    = meta.icon;
    pdetailHeading.textContent = meta.heading;
    pdetailSub.textContent     = meta.sub;
    pdetailTimer.textContent   = '0:00';
    pipelineDetail.classList.add('is-open');
    startTimer(agent);
    setPagePhase(meta.phase);

    generateBtn.classList.remove('phase--researching', 'phase--writing', 'phase--editing');
    generateBtn.classList.add(`phase--${meta.phase}`);
    btnLabel.textContent = meta.btnLabel;

    /* Carriage return sound on each agent handoff */
    playCarriageReturn();
  }

  if (state === 'done') {
    if (activeAgent === agent) activeAgent = null;
    stopTimer(agent);
    fireParticles(agent);
  }
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
  activeAgent = null;
  setPagePhase(null);
}

/* ── Tab switcher ───────────────────────────────────────────────────────── */
function positionIndicator() {
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
  requestAnimationFrame(positionIndicator);
  playKeyClick(0.1);
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
  initChips();
  fetchHistoryCount();
  initPersonalizeModal();

  const strip = $('schedStrip');
  if (strip) strip.style.display = 'inline-flex';
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

clearBtn.addEventListener('click', () => { logsPanel.innerHTML = ''; playKeyClick(0.1); });
collapseBtn.addEventListener('click', () => {
  logsSection.classList.toggle('is-collapsed');
  playKeyClick(0.1);
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

/* ── Export / Copy ──────────────────────────────────────────────────────── */
function downloadBlob(content, filename, mime) {
  const blob = new Blob([content], { type: mime });
  const a    = document.createElement('a');
  a.href     = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

function safeTopic() {
  return (currentTopic || 'blog-post').replace(/[^a-z0-9]+/gi, '-').toLowerCase().slice(0, 60);
}

copyBtn.addEventListener('click', e => {
  if (e.target === exportToggle || exportToggle.contains(e.target)) return;
  if (!currentResult) return;
  const text = activeTab === 'raw' ? currentResult : (proseContent.innerText || currentResult);
  navigator.clipboard.writeText(text).then(() => {
    const orig = copyLabel.textContent;
    copyLabel.textContent = 'Copied!';
    playKeyClick(0.15);
    setTimeout(() => (copyLabel.textContent = orig), 2000);
  });
});

if (exportToggle) {
  exportToggle.addEventListener('click', e => {
    e.stopPropagation();
    exportMenu.classList.toggle('is-open');
    playKeyClick(0.1);
  });
}

document.addEventListener('click', e => {
  if (exportMenu && exportMenu.classList.contains('is-open')) {
    if (!exportWrap?.contains(e.target)) exportMenu.classList.remove('is-open');
  }
});
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') exportMenu?.classList.remove('is-open');
});

$('exportCopyMd')?.addEventListener('click', () => {
  if (!currentResult) return;
  navigator.clipboard.writeText(currentResult).then(() => {
    showToast('success', 'Copied!', 'Markdown copied to clipboard.', 2500);
  });
  exportMenu.classList.remove('is-open');
  playKeyClick(0.12);
});

$('exportCopyHtml')?.addEventListener('click', () => {
  if (!proseContent.innerHTML) return;
  navigator.clipboard.writeText(proseContent.innerHTML).then(() => {
    showToast('success', 'Copied!', 'HTML copied to clipboard.', 2500);
  });
  exportMenu.classList.remove('is-open');
  playKeyClick(0.12);
});

$('exportDownloadMd')?.addEventListener('click', () => {
  if (!currentResult) return;
  downloadBlob(currentResult, `${safeTopic()}.md`, 'text/markdown');
  exportMenu.classList.remove('is-open');
  playKeyClick(0.12);
});

$('exportDownloadTxt')?.addEventListener('click', () => {
  if (!currentResult) return;
  const txt = proseContent.innerText || currentResult;
  downloadBlob(txt, `${safeTopic()}.txt`, 'text/plain');
  exportMenu.classList.remove('is-open');
  playKeyClick(0.12);
});

/* ── History Drawer ─────────────────────────────────────────────────────── */
function openDrawer() {
  drawerOpen = true;
  drawerOverlay.classList.add('is-open');
  historyDrawer.classList.add('is-open');
  document.body.style.overflow = 'hidden';
  fetchHistory();
  playKeyClick(0.12);
}

function closeDrawer() {
  drawerOpen = false;
  historyDrawer.classList.remove('is-open');
  drawerOverlay.classList.remove('is-open');
  document.body.style.overflow = '';
  playKeyClick(0.1);
}

historyBtn?.addEventListener('click', openDrawer);
drawerClose?.addEventListener('click', closeDrawer);
drawerOverlay?.addEventListener('click', closeDrawer);

document.addEventListener('keydown', e => {
  if (e.key === 'Escape' && drawerOpen) closeDrawer();
});

function fetchHistoryCount() {
  fetch('/api/posts').then(r => r.ok ? r.json() : []).then(posts => {
    if (historyBadge) historyBadge.textContent = posts.length || '0';
  }).catch(() => {});
}

function fetchHistory() {
  if (!drawerBody) return;
  drawerBody.innerHTML = '<div class="drawer-empty">Loading…</div>';

  fetch('/api/posts')
    .then(r => r.ok ? r.json() : [])
    .then(posts => {
      if (historyBadge) historyBadge.textContent = posts.length || '0';
      if (!posts.length) {
        drawerBody.innerHTML = '<div class="drawer-empty">No posts yet.<br>Generate your first blog post above.</div>';
        return;
      }
      drawerBody.innerHTML = '';
      [...posts].reverse().forEach(post => { drawerBody.appendChild(buildHistoryCard(post)); });
    })
    .catch(() => {
      drawerBody.innerHTML = '<div class="drawer-empty">Could not load posts.</div>';
    });
}

function buildHistoryCard(post) {
  const card    = document.createElement('div');
  card.className = 'hcard';

  const status  = post.status || 'pending';
  const topic   = post.topic  || 'Untitled';
  const created = (post.created_at || '').slice(0, 10);
  const wc      = post.content
    ? post.content.trim().split(/\s+/).filter(Boolean).length : 0;

  card.innerHTML = `
    <div class="hcard__top">
      <span class="hcard__title">${escapeHtml(topic)}</span>
      <span class="hcard__badge hcard__badge--${status}">${status}</span>
    </div>
    <div class="hcard__meta">
      <span>${created}</span>
      ${wc ? `<span>~${wc} words</span>` : ''}
    </div>
  `;

  card.addEventListener('click', () => {
    if (!post.content) return;
    clearTypewriterTimer();
    currentResult = post.content;
    currentTopic  = post.topic || '';
    const html = window.marked ? marked.parse(currentResult) : `<pre>${escapeHtml(currentResult)}</pre>`;
    proseContent.innerHTML = html;
    rawTextarea.value = currentResult;
    const words = currentResult.trim().split(/\s+/).filter(Boolean).length;
    wordCount.textContent = `~${words} words`;
    resultSection.classList.add('is-visible');
    requestAnimationFrame(positionIndicator);
    closeDrawer();
    setTimeout(() => resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' }), 80);
    playKeyClick(0.15);
  });

  return card;
}

/* ── Reset ──────────────────────────────────────────────────────────────── */
function resetUI() {
  clearTypewriterTimer();
  stopAllTimers(); stopCrawl();
  resetPipeline();
  logsPanel.innerHTML = '';
  resetMetricsPanel();
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

  generateBtn.classList.remove('phase--researching', 'phase--writing', 'phase--editing', 'phase--done');
}

function endRun(label) {
  isRunning = false;
  clearTypewriterTimer();
  stopCrawl(); stopAllTimers();
  activeEs?.close();
  activeEs = null;
  generateBtn.disabled = false;
  generateBtn.classList.remove('is-loading', 'phase--researching', 'phase--writing', 'phase--editing', 'phase--done');
  btnLabel.textContent = label;
  setPagePhase(null);
  fetchHistoryCount();
}

/* ── Helpers ─────────────────────────────────────────────────────────────── */
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => (
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
      const idx   = ORDER[e.agent] ?? 0;
      stopCrawl();
      AGENTS.slice(0, idx).forEach((a, i) => { setStepState(a, 'done'); fillEdge(i + 1); });
      setStepState(e.agent, 'active');
      const next = AGENTS[idx + 1];
      setProgress(MILESTONES[e.agent], `${e.agent.charAt(0).toUpperCase() + e.agent.slice(1)} working…`);
      startCrawl(next ? MILESTONES[next] : MILESTONES.done);
      break;
    }

    case 'log': {
      const msg = e.message?.trim();
      if (msg) {
        appendLog(e.agent || 'system', msg);
        if (e.agent && e.agent === activeAgent) {
          const truncated = msg.length > 72 ? msg.slice(0, 70) + '…' : msg;
          pdetailSub.textContent = truncated;
          pdetailSub.classList.remove('detail-flash');
          void pdetailSub.offsetWidth; // force reflow to restart animation
          pdetailSub.classList.add('detail-flash');
        }
      }
      break;
    }

    case 'final': {
      currentResult = e.content || '';
      currentTopic  = topicInput.value.trim();
      stopCrawl(); stopAllTimers();
      AGENTS.forEach((a, i) => { setStepState(a, 'done'); if (i < 2) fillEdge(i + 1); });
      pipelineDetail.classList.remove('is-open');
      setProgress(100, 'Complete — manuscript ready');
      logsLive.style.display = 'none';
      setPagePhase(null);

      generateBtn.classList.remove('phase--researching', 'phase--writing', 'phase--editing');
      generateBtn.classList.add('phase--done');
      btnLabel.textContent = '✓ Done';
      setTimeout(() => {
        generateBtn.classList.remove('phase--done');
        generateBtn.disabled = false;
        btnLabel.textContent = 'Generate again';
      }, 1800);

      /* Show result card first, then typewriter-reveal the content */
      rawTextarea.value = currentResult;
      const wc   = currentResult.trim().split(/\s+/).filter(Boolean).length;
      wordCount.textContent = `~${wc} words`;
      resultSection.classList.add('is-visible');
      requestAnimationFrame(positionIndicator);
      setTimeout(() => resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' }), 80);

      const html = window.marked ? marked.parse(currentResult) : `<pre>${escapeHtml(currentResult)}</pre>`;

      /* Small delay so scroll settles before the typing starts */
      setTimeout(() => {
        typewriterReveal(proseContent, html, () => {
          appendLog('system', `Saved → ${e.saved_to ?? 'output/'}`);
          showToast('success', 'Manuscript ready', `${wc} words · saved to ${e.saved_to ?? 'output/'}`, 5000);
          fetchHistoryCount();
        });
      }, 300);
      break;
    }

    case 'metrics': {
      if (!metricsData[e.agent])
        metricsData[e.agent] = { iters: [], skipped: false, skipReason: '' };
      metricsData[e.agent].iters[e.iteration] = e.metrics;
      renderAgentMetrics(e.agent);
      break;
    }

    case 'critique_start': {
      if (e.agent === activeAgent) {
        pdetailSub.textContent = `Self-review round ${e.iteration}…`;
        pdetailSub.classList.remove('detail-flash');
        void pdetailSub.offsetWidth;
        pdetailSub.classList.add('detail-flash');
      }
      appendLog(e.agent, `Self-reviewing (round ${e.iteration})…`);
      playKeyClick(0.06);
      break;
    }

    case 'metrics_skip': {
      if (metricsData[e.agent]) {
        metricsData[e.agent].skipped    = true;
        metricsData[e.agent].skipReason = e.reason;
      }
      appendLog(e.agent, `Self-critic: ${e.reason}`);
      renderAgentMetrics(e.agent);
      break;
    }

    case 'error':
      appendLog('system', `Error: ${e.message}`);
      showToast('error', 'Generation failed', e.message || 'Something went wrong.');
      endRun('Try again');
      break;

    case 'done':
      if (isRunning) {
        isRunning = false;
        stopCrawl(); stopAllTimers();
        activeEs?.close();
        activeEs = null;
        generateBtn.disabled = false;
        generateBtn.classList.remove('is-loading');
      }
      break;
  }
}

/* ── Generate ───────────────────────────────────────────────────────────── */
function generate() {
  if (isRunning) return;

  const topic = topicInput.value.trim();
  if (!topic) { showHint('Please enter a topic first.'); topicInput.focus(); return; }

  activeEs?.close();
  activeEs = null;
  resetUI();
  isRunning    = true;
  currentTopic = topic;
  generateBtn.disabled = true;
  generateBtn.classList.add('is-loading', 'phase--researching');
  btnLabel.textContent = 'Researching…';

  /* Big key press sound on generate */
  playKeyClick(0.2);

  const params = new URLSearchParams({
    topic,
    tone:     agentStyle.tone,
    length:   agentStyle.length,
    audience: agentStyle.audience,
    notes:    agentStyle.notes || '',
    critique: agentStyle.critique === 'on' ? 'true' : 'false',
  });

  activeEs = new EventSource(`/api/generate?${params}`);
  activeEs.onmessage = ev => handleEvent(ev.data);
  activeEs.onerror   = () => {
    if (!isRunning) return;
    showToast('error', 'Connection lost', 'The stream was interrupted. Please try again.');
    endRun('Generate');
  };
}

generateBtn.addEventListener('click', generate);
topicInput.addEventListener('keydown', e => { if (e.key === 'Enter') generate(); });
