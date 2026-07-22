// AnimateView.js — Phase C (v2) : anime le DATAFLOW d'un token sur le pipeline.
//
// Support visuel = le pipeline (onglet « Flux pipeline »), pas le graphe de
// hiérarchie. Deux modes :
//   • Démo (défaut, jouable immédiatement, pas de Run) : timeline synthétique du
//     pipeline complet basée sur le modèle de timing du design. Montre la pleine
//     traversée embed → rms → QKV → rope → attn → FFN (×5 couches) → lm_head →
//     argmax sur 17 tokens. C'est pédagogique (le VCD tb_node, path non-FLASH_BOOT
//     rtl=0, reste bloqué en embed/qkv — voir [[sim-path-vs-silicon-path]]).
//   • VCD réel (bouton Run tb_node) : lit u_seq.st + u_seq.phase du VCD et mappe
//     à l'étape pipeline (FSM gen_seq.v). Couvre le début réel (embed→rms→qkv).
//
// Au cycle courant on colore le nœud `step:*` + les vecteurs incidents du SVG.

import svgPanZoom from 'svg-pan-zoom';
import { runSim, simStatus, simTimeline, getPipeline, getPipelineSvg } from '../api.js';
import { cssEscape } from '../graph/GraphView.js';
import { NLAYERS, NTOKENS, buildDemo } from '../pipeline/timeline.js';

const TB = 'tb_node';
const TB_PREFIX = 'tb_node.u_dut.';
const SIG_ST = `${TB_PREFIX}u_seq.st`;
const SIG_PHASE = `${TB_PREFIX}u_seq.phase`;

const TIER_NAME = { flash: 'Flash', sdram: 'SDRAM', bram: 'BRAM' };
const TIER_ORDER = ['flash', 'sdram', 'bram'];

// FSM gen_seq.v : phase[3:0] + st[4:0] -> étape pipeline (mode VCD réel).
const PHASE_STEP = {
  13: 'embed', 0: 'rms_att', 1: 'qkv', 2: 'qkv', 3: 'qkv', 4: 'attn',
  5: 'wo', 6: 'rms_ffn', 7: 'ffn_up', 8: 'ffn_up', 9: 'silu',
  10: 'ffn_down', 11: 'rms_final', 12: 'lm_head',
};
function stepOf(st, phase, lp) {
  switch (st) {
    case 8: case 9: return 'rope';
    case 10: return 'kvwrite';
    case 11: return 'attn';
    case 12: case 14: case 15: case 16: case 17: case 18:
      return (lp.val === 5) ? 'resid1' : (lp.val === 10) ? 'resid2' : 'resid1';
    case 19: case 20: case 22: case 23: return 'argmax';
    case 21: return 'lm_head';
    case 6: case 13: return 'token';
    case 0: case 1: return null;
  }
  const s = PHASE_STEP[phase];
  if (s !== undefined) { lp.val = phase; return s; }
  return null;
}

// --- Démo synthétique : modèle de timing du design (durées relatives) ---
// Factored dans ../pipeline/timeline.js (partagé avec la vue Cluster pour
// l'animation synchronisée du trafic ss_link). NLAYERS/NTOKENS/LAYER/ONE_TOKEN
// + buildDemo() importés en tête de fichier.

export class AnimateView {
  constructor(root) {
    root.innerHTML = `
      <div class="pt-layout">
        <div class="pt-toolbar">
          <select id="an-mode">
            <option value="demo">Démo (modèle)</option>
            <option value="vcd">VCD réel (tb_node)</option>
          </select>
          <button id="an-run" class="pt-btn" disabled>▶ Run tb_node</button>
          <span id="an-status" class="pt-muted">— Animate dataflow : play ▶</span>
          <span class="pt-sep">|</span>
          <button id="an-play" class="pt-btn">▶ play</button>
          <button id="an-stepb" class="pt-btn">◀ cycle</button>
          <button id="an-stepf" class="pt-btn">cycle ▶</button>
          <label>cyc/s <input id="an-speed" type="number" value="240" min="1" max="4000" style="width:54px"></label>
          <input id="an-seek" type="range" min="0" max="0" value="0" style="flex:1;min-width:160px">
          <span id="an-cyc" class="pt-muted">0 / 0</span>
        </div>
        <div id="an-memband" class="pt-memband"></div>
        <div class="pt-main">
          <div id="an-graph" class="pt-graph"></div>
          <div class="pt-sidebar" id="an-sidebar">
            <div class="pt-side-sec"><b>Étape courante</b><div id="an-now">—</div></div>
            <div class="pt-side-sec"><b>Mémoires actives</b><div id="an-mem">—</div></div>
            <div class="pt-side-sec"><b>Position</b><div id="an-pos">—</div></div>
            <div class="pt-side-sec"><b>Vecteurs en transit</b><div id="an-vec">—</div></div>
            <div class="pt-side-sec"><b>Légende</b>
              <div class="pt-mod-desc">
                Démo : le token traverse le pipeline (5 couches puis lm_head→argmax),
                puis boucle pour le token suivant (×17). Survolez une étape du
                pipeline pour sa description. VCD réel : lit la FSM <code>u_seq.st/phase</code>
                du testbench (path sim — ne dépasse pas qkv).
              </div>
            </div>
          </div>
        </div>
      </div>`;
    this.el = root;
    this.runBtn = root.querySelector('#an-run');
    this.modeSel = root.querySelector('#an-mode');
    this.statusEl = root.querySelector('#an-status');
    this.playBtn = root.querySelector('#an-play');
    this.stepbBtn = root.querySelector('#an-stepb');
    this.stepfBtn = root.querySelector('#an-stepf');
    this.speedInp = root.querySelector('#an-speed');
    this.seek = root.querySelector('#an-seek');
    this.cycEl = root.querySelector('#an-cyc');
    this.graphHost = root.querySelector('#an-graph');
    this.memBand = root.querySelector('#an-memband');
    this.nowEl = root.querySelector('#an-now');
    this.memEl = root.querySelector('#an-mem');
    this.posEl = root.querySelector('#an-pos');
    this.vecEl = root.querySelector('#an-vec');

    this.mode = 'demo';
    this.cycles = [];
    this.stepAtCycle = [];
    this.tokAtCycle = [];
    this.layAtCycle = [];
    this.iCycle = 0;
    this.playing = false;
    this._loadPipeline();
    this._bind();
  }

  async _loadPipeline() {
    const [svg, data] = await Promise.all([getPipelineSvg(), getPipeline()]);
    this.data = data;
    this._buildMemBand();
    this.graphHost.innerHTML = svg;
    const svgEl = this.graphHost.querySelector('svg');
    if (!svgEl) return;
    svgEl.setAttribute('width', '100%');
    svgEl.setAttribute('height', '100%');
    if (this.panZoom) this.panZoom.destroy();
    this.panZoom = svgPanZoom(svgEl, {
      controlIconsEnabled: true, fit: true, center: true, minZoom: 0.2, maxZoom: 8,
    });
    // la démo est jouable dès que le SVG est là.
    this._startDemo();
  }

  // Construit la bande mémoire : 3 colonnes (flash/SDRAM/BRAM) avec les régions
  // en cellules. Statique — seules les classes d'activité bougent par cycle.
  _buildMemBand() {
    const regions = this.data.mem_regions || {};
    const byTier = {};
    for (const id in regions) {
      const r = regions[id];
      (byTier[r.tier] = byTier[r.tier] || []).push(id);
    }
    const cols = TIER_ORDER.map((tier) => {
      const cells = (byTier[tier] || []).map((id) =>
        `<div class="pt-mem-cell" id="anmem-${id}" data-tier="${tier}">${id}</div>`).join('');
      return `<div class="pt-mem-col" data-tier="${tier}">` +
        `<div class="pt-mem-col-title">${TIER_NAME[tier]}</div>` +
        `<div class="pt-mem-cells">${cells}</div></div>`;
    }).join('');
    this.memBand.innerHTML = cols;
  }

  // Allume les cellules mémoire de l'étape courante + remplit la sidebar.
  _renderMem(stepId) {
    const step = this.data.steps[stepId] || {};
    const mem = step.mem || [];
    const regions = this.data.mem_regions || {};
    if (!mem.length) {
      this.memEl.innerHTML = '<span class="pt-muted">— (aucune accès mémoire)</span>';
      return;
    }
    for (const a of mem) {
      const cell = this.memBand.querySelector(`#anmem-${cssEscape(a.region)}`);
      if (cell) {
        cell.classList.add('pt-mem-active');
        cell.classList.add(a.op === 'write' ? 'pt-mem-write' : 'pt-mem-read');
      }
    }
    const chips = mem.map((a) => {
      const reg = regions[a.region] || { tier: '?' };
      const op = a.op === 'write' ? 'W' : 'R';
      const name = TIER_NAME[reg.tier] || reg.tier;
      return `<div style="margin-bottom:3px"><span class="pt-mem-chip" data-tier="${reg.tier}">${name} · ${op}</span><code>${a.region}</code></div>`;
    });
    this.memEl.innerHTML = `<div class="pt-mod-desc">${chips.join('')}</div>`;
  }

  _startDemo() {
    const d = buildDemo();
    this.stepAtCycle = d.stepAt;
    this.tokAtCycle = d.tokAt;
    this.layAtCycle = d.layAt;
    this.cycles = d.stepAt.map((_, i) => i);
    this.iCycle = 0;
    this.seek.max = Math.max(0, this.cycles.length - 1);
    this.seek.value = 0;
    this.statusEl.textContent = `démo : ${this.cycles.length} ticks, ${NTOKENS} tokens, ${NLAYERS} couches`;
    this._render();
  }

  _bind() {
    this.runBtn.onclick = () => this._run();
    this.modeSel.onchange = () => {
      this.mode = this.modeSel.value;
      this.runBtn.disabled = (this.mode !== 'vcd');
      if (this.mode === 'demo') this._startDemo();
    };
    this.playBtn.onclick = () => this._togglePlay();
    this.stepfBtn.onclick = () => this._step(1);
    this.stepbBtn.onclick = () => this._step(-1);
    this.seek.oninput = () => { this.iCycle = +this.seek.value; this._render(); };
    this.speedInp.onchange = () => { if (this.playing) this._startTick(); };
  }

  async _run() {
    if (this.mode !== 'vcd') return;
    this.runBtn.disabled = true;
    this.statusEl.textContent = `lancement ${TB}…`;
    const job = await runSim(TB, 'iverilog');
    this.jobId = job.job_id;
    this._poll();
  }

  async _poll() {
    const job = await simStatus(this.jobId);
    if (job.status === 'running') {
      this.statusEl.textContent = `running…`;
      setTimeout(() => this._poll(), 300);
      return;
    }
    if (job.status !== 'done') {
      this.statusEl.textContent = `échec : ${job.error || 'voir log'}`;
      this.runBtn.disabled = false;
      return;
    }
    this.statusEl.textContent = `VCD prêt — lecture de u_seq.st/phase…`;
    await this._fetchTimeline();
  }

  async _fetchTimeline() {
    this.tl = await simTimeline(this.jobId, [SIG_ST, SIG_PHASE], 0, null, null);
    this.cycles = this.tl.cycles || [];
    const byName = new Map(this.tl.signals.filter((s) => s.found).map((s) => [s.name, s]));
    const stSig = byName.get(SIG_ST);
    const phSig = byName.get(SIG_PHASE);
    if (!stSig || !phSig) {
      this.statusEl.textContent = 'signaux u_seq.st/phase introuvables';
      this.runBtn.disabled = false;
      return;
    }
    const lp = { val: -1 };
    this.stepAtCycle = this.cycles.map((t) => {
      const st = this._valAt(stSig, t);
      const ph = this._valAt(phSig, t);
      if (st === null || ph === null) return null;
      return stepOf(st, ph, lp);
    });
    this.tokAtCycle = this.cycles.map(() => -1);
    this.layAtCycle = this.cycles.map(() => -1);
    this.iCycle = 0;
    this.seek.max = Math.max(0, this.cycles.length - 1);
    this.seek.value = 0;
    this.runBtn.disabled = false;
    const n = this.stepAtCycle.filter((x) => x).length;
    this.statusEl.textContent = `VCD : ${this.cycles.length} cycles, ${n} actifs (path sim, bloqué après qkv)`;
    this._render();
  }

  _valAt(sig, t) {
    const tr = sig.transitions;
    if (!tr || !tr.length) return null;
    let lo = 0, hi = tr.length - 1, res = 0;
    while (lo <= hi) {
      const m = (lo + hi) >> 1;
      if (tr[m].t <= t) { res = m; lo = m + 1; } else hi = m - 1;
    }
    const v = tr[res].v;
    if (typeof v !== 'string' || /[xz]/i.test(v)) return null;
    const n = parseInt(v, 2);
    return Number.isNaN(n) ? null : n;
  }

  _togglePlay() {
    this.playing = !this.playing;
    this.playBtn.textContent = this.playing ? '⏸ pause' : '▶ play';
    if (this.playing) this._startTick(); else this._stopTick();
  }

  _startTick() {
    this._stopTick();
    const cps = Math.max(1, +this.speedInp.value || 240);
    this._tick = setInterval(() => {
      this.iCycle++;
      if (this.iCycle >= this.cycles.length) {
        this.iCycle = (this.mode === 'demo') ? 0 : this.cycles.length - 1;  // démo boucle
        if (this.mode !== 'demo') { this._togglePlay(); return; }
      }
      this.seek.value = this.iCycle;
      this._render();
    }, 1000 / cps);
  }

  _stopTick() { if (this._tick) { clearInterval(this._tick); this._tick = null; } }

  _step(d) {
    this.iCycle = Math.max(0, Math.min(this.cycles.length - 1, this.iCycle + d));
    this.seek.value = this.iCycle;
    this._render();
  }

  _render() {
    this.cycEl.textContent = `${this.iCycle} / ${Math.max(0, this.cycles.length - 1)}`;
    if (!this.panZoom || !this.data) return;
    this.graphHost.querySelectorAll('[id^="step:"]').forEach((n) => n.classList.remove('pt-active'));
    this.graphHost.querySelectorAll('[id^="vec:"]').forEach((e) => e.classList.remove('pt-active-edge'));
    this.memBand.querySelectorAll('.pt-mem-cell').forEach((c) =>
      c.classList.remove('pt-mem-active', 'pt-mem-read', 'pt-mem-write'));
    const stepId = this.stepAtCycle[this.iCycle];
    if (!stepId) {
      this.nowEl.innerHTML = '<span class="pt-muted">— (IDLE / non démarré)</span>';
      this.memEl.innerHTML = '<span class="pt-muted">—</span>';
      this.posEl.innerHTML = '—';
      this.vecEl.innerHTML = '—';
      return;
    }
    this._renderMem(stepId);
    const node = this.graphHost.querySelector(`#${cssEscape('step:' + stepId)}`);
    if (node) node.classList.add('pt-active');
    const incs = this.data.vecs.filter((v) => v.from === stepId || v.to === stepId);
    for (const v of incs) {
      const el = this.graphHost.querySelector(`#${cssEscape(v.id)}`);
      if (el) el.classList.add('pt-active-edge');
    }
    const step = this.data.steps[stepId] || {};
    this.nowEl.innerHTML = `<code>${(step.label || stepId).replace(/\n/g, ' ')}</code>` +
      `<div class="pt-mod-desc">${step.desc || ''}</div>`;
    // position (token / couche) — démo seulement
    const tok = this.tokAtCycle[this.iCycle];
    const lay = this.layAtCycle[this.iCycle];
    this.posEl.innerHTML = tok >= 0
      ? `token <b>${tok + 1}/${NTOKENS}</b>` + (lay >= 0 ? ` · couche <b>${lay + 1}/${NLAYERS}</b>` : '')
      : (this.mode === 'demo'
        ? '<span class="pt-muted">boot (power-up, 1×)</span>'
        : '<span class="pt-muted">— (VCD réel)</span>');
    this.vecEl.innerHTML = incs.length
      ? incs.map((v) => `<code>${v.label}</code> <span class="pt-muted">(${v.from}→${v.to})</span>`).join('<br>')
      : '—';
  }

  get ready() { return !!this.panZoom; }
}