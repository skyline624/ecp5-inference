// TimelineView.js — Phase B : timeline VCD (waveform) + lecture cycle-par-cycle.
//
// Toolbar : choix TB + Run, puis play/pause/step±1 cycle, seek, vitesse.
// Waveform : une lane par signal sélectionné (escalier pour 1-bit, valeur pour
// bus). Curseur vertical au cycle courant. Les fronts montants de clk = cycles.
// Sidebar : liste des signaux VCD (checkboxes) + valeurs au curseur.
//
// Le VCD vient de /api/sim/{job_id}/timeline (POST batch). On garde les
// transitions côté client et on calcule la valeur au curseur par bissection.

import { listTbs, runSim, simStatus, simSignals, simTimeline } from '../api.js';

const LANE_H = 30;
const LABEL_W = 120;

export class TimelineView {
  constructor(root) {
    root.innerHTML = `
      <div class="pt-layout">
        <div class="pt-toolbar">
          <label>TB <select id="tl-tb"></select></label>
          <button id="tl-run" class="pt-btn">▶ Run</button>
          <span id="tl-status" class="pt-muted">—</span>
          <span class="pt-sep">|</span>
          <button id="tl-play" class="pt-btn">▶ play</button>
          <button id="tl-stepb" class="pt-btn">◀ cycle</button>
          <button id="tl-stepf" class="pt-btn">cycle ▶</button>
          <label>cyc/s <input id="tl-speed" type="number" value="40" min="1" max="500" style="width:54px"></label>
          <input id="tl-seek" type="range" min="0" max="0" value="0" style="flex:1;min-width:160px">
          <span id="tl-cyc" class="pt-muted">0 / 0</span>
          <span class="pt-hint">Phase B — timeline VCD</span>
        </div>
        <div class="pt-main">
          <div id="tl-wave" class="tl-wave"></div>
          <div class="pt-sidebar" id="tl-sidebar">
            <div class="pt-side-sec"><b>Signaux VCD</b><div id="tl-siglist" class="tl-siglist"></div></div>
            <div class="pt-side-sec"><b>Valeurs au curseur</b><div id="tl-vals"></div></div>
          </div>
        </div>
      </div>`;
    this.el = root;
    this.tbSel = root.querySelector('#tl-tb');
    this.runBtn = root.querySelector('#tl-run');
    this.statusEl = root.querySelector('#tl-status');
    this.playBtn = root.querySelector('#tl-play');
    this.stepbBtn = root.querySelector('#tl-stepb');
    this.stepfBtn = root.querySelector('#tl-stepf');
    this.speedInp = root.querySelector('#tl-speed');
    this.seek = root.querySelector('#tl-seek');
    this.cycEl = root.querySelector('#tl-cyc');
    this.waveEl = root.querySelector('#tl-wave');
    this.siglistEl = root.querySelector('#tl-siglist');
    this.valsEl = root.querySelector('#tl-vals');
    this.cycles = [];
    this.tl = null;        // timeline data
    this.selected = new Set();
    this.iCycle = 0;
    this.playing = false;
    this._bind();
    this._loadTbs();
  }

  async _loadTbs() {
    const t = await listTbs();
    this.tbs = t.tbs;
    this.tbSel.innerHTML = this.tbs.map((x) =>
      `<option value="${x.name}">${x.name} — ${x.desc}</option>`).join('');
    // default : le 1er TB iverilog (compile en icarus strict)
    const iv = this.tbs.find((x) => x.mode === 'iverilog') || this.tbs[0];
    this.tbSel.value = iv.name;
  }

  _bind() {
    this.runBtn.onclick = () => this._run();
    this.playBtn.onclick = () => this._togglePlay();
    this.stepfBtn.onclick = () => this._step(1);
    this.stepbBtn.onclick = () => this._step(-1);
    this.seek.oninput = () => { this.iCycle = +this.seek.value; this._render(); };
    this.speedInp.onchange = () => { if (this.playing) this._startTick(); };
  }

  async _run() {
    const tb = this.tbSel.value;
    this.statusEl.textContent = `lancement ${tb}…`;
    const job = await runSim(tb);
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
      return;
    }
    this.statusEl.textContent = `VCD prêt (${job.test})`;
    await this._loadSignals();
  }

  async _loadSignals() {
    const sig = await simSignals(this.jobId);
    // scopes -> {scope: [{ref,name,width}]}
    this.scopes = sig.scopes;
    // scope le plus shallow (le TB lui-même, pas l'intérieur du DUT)
    const scopes = Object.keys(this.scopes).sort((a, b) => a.split('.').length - b.split('.').length);
    const topScope = scopes[0] || '';
    this.allRefs = this.scopes[topScope] || [];
    // default sélection : tout sauf clk et vars de boucle (i, j)
    this.selected = new Set();
    for (const s of this.allRefs) {
      const base = s.name.split('.').pop();
      if (base === 'clk' || base === 'i' || base === 'j') continue;
      this.selected.add(s.ref);
    }
    this._renderSigList();
    await this._fetchTimeline();
  }

  _renderSigList() {
    this.siglistEl.innerHTML = this.allRefs.map((s) =>
      `<label class="tl-sig"><input type="checkbox" data-ref="${s.ref}" ${this.selected.has(s.ref) ? 'checked' : ''}>
       <code>${s.name}</code> <span class="pt-muted">[${s.width}]</span></label>`).join('');
    this.siglistEl.querySelectorAll('input').forEach((cb) => {
      cb.onchange = () => {
        if (cb.checked) this.selected.add(cb.dataset.ref);
        else this.selected.delete(cb.dataset.ref);
        this._fetchTimeline();
      };
    });
  }

  async _fetchTimeline() {
    if (!this.selected.size) { this.tl = null; this._render(); return; }
    const sigs = [...this.selected];
    this.tl = await simTimeline(this.jobId, sigs, 0, null, null);
    this.cycles = this.tl.cycles;
    this.iCycle = Math.min(this.iCycle, this.cycles.length);
    this.seek.max = Math.max(0, this.cycles.length - 1);
    this.seek.value = this.iCycle;
    this._render();
  }

  _togglePlay() {
    this.playing = !this.playing;
    this.playBtn.textContent = this.playing ? '⏸ pause' : '▶ play';
    if (this.playing) this._startTick();
    else this._stopTick();
  }

  _startTick() {
    this._stopTick();
    const cps = Math.max(1, +this.speedInp.value || 40);
    this._tick = setInterval(() => {
      this.iCycle++;
      if (this.iCycle >= this.cycles.length) { this.iCycle = this.cycles.length - 1; this._togglePlay(); return; }
      this.seek.value = this.iCycle;
      this._render();
    }, 1000 / cps);
  }

  _stopTick() {
    if (this._tick) { clearInterval(this._tick); this._tick = null; }
  }

  _step(d) {
    this.iCycle = Math.max(0, Math.min(this.cycles.length - 1, this.iCycle + d));
    this.seek.value = this.iCycle;
    this._render();
  }

  _tAt() {
    // instant (ps) au cycle courant = front montant correspondant (0 si avant le 1er)
    if (!this.cycles.length) return 0;
    if (this.iCycle <= 0) return 0;
    return this.cycles[Math.min(this.iCycle, this.cycles.length) - 1];
  }

  _render() {
    this.cycEl.textContent = `${this.iCycle} / ${Math.max(0, this.cycles.length - 1)}`;
    if (!this.tl) { this.waveEl.innerHTML = '<div class="pt-muted" style="padding:12px">Lancez une simulation.</div>'; this.valsEl.innerHTML = '—'; return; }
    const endT = this.tl.end_time || 1;
    const width = Math.max(600, this.waveEl.clientWidth || 800);
    const lanes = this.tl.signals.filter((s) => s.found);
    const H = Math.max(120, lanes.length * LANE_H + 20);
    const x = (t) => LABEL_W + (t / endT) * (width - LABEL_W - 10);

    let svg = `<svg width="100%" viewBox="0 0 ${width} ${H}" preserveAspectRatio="none" style="height:${H}px">`;
    // marqueurs de cycle (lignes légères)
    for (const t of this.cycles)
      svg += `<line x1="${x(t)}" y1="0" x2="${x(t)}" y2="${H}" stroke="#45475a" stroke-width="0.5" opacity="0.6"/>`;
    // lanes
    let y = 10;
    for (const s of lanes) {
      svg += this._renderLane(s, x, y, endT);
      y += LANE_H;
    }
    // curseur
    const ct = this._tAt();
    svg += `<line x1="${x(ct)}" y1="0" x2="${x(ct)}" y2="${H}" stroke="#f38ba8" stroke-width="1.5"/>`;
    svg += `</svg>`;
    this.waveEl.innerHTML = svg;
    this._renderVals(ct);
  }

  _renderLane(s, x, y, endT) {
    const trans = s.transitions;
    const hi = y + 6, lo = y + LANE_H - 8;
    let out = `<text x="6" y="${y + 18}" fill="#cdd6f4" font-size="11" font-family="monospace">${s.name.split('.').pop()}</text>`;
    out += `<line x1="${LABEL_W}" y1="${y + LANE_H / 2}" x2="${x(endT)}" y2="${y + LANE_H / 2}" stroke="#45475a" stroke-width="0.5"/>`;
    if (!trans.length) return out;
    const isBit = s.width === 1;
    let prevVal = trans[0].v;
    let prevT = 0;
    const seg = (t0, t1, val) => {
      const xv0 = x(t0), xv1 = x(t1);
      if (isBit) {
        const yv = val === '1' ? hi : lo;
        out += `<line x1="${xv0}" y1="${yv}" x2="${xv1}" y2="${yv}" stroke="#89b4fa" stroke-width="1.4"/>`;
        if (prevVal !== val && prevT !== t0)
          out += `<line x1="${xv0}" y1="${val === '1' ? lo : hi}" x2="${xv0}" y2="${yv}" stroke="#89b4fa" stroke-width="1.4"/>`;
      } else {
        const yv = y + 10;
        out += `<line x1="${xv0}" y1="${yv}" x2="${xv1}" y2="${yv}" stroke="#89b4fa" stroke-width="1.4"/>`;
        if (xv1 - xv0 > 18)
          out += `<text x="${xv0 + 3}" y="${y + 8}" fill="#f9e2af" font-size="10" font-family="monospace">${fmt(val, s.width)}</text>`;
      }
    };
    for (let i = 0; i < trans.length; i++) {
      const t = trans[i].t, v = trans[i].v;
      seg(prevT, t, prevVal);
      prevVal = v; prevT = t;
    }
    seg(prevT, endT, prevVal);
    return out;
  }

  _renderVals(ct) {
    if (!this.tl) { this.valsEl.innerHTML = '—'; return; }
    let html = '';
    for (const s of this.tl.signals) {
      if (!s.found) continue;
      const v = valAt(s.transitions, ct);
      html += `<div class="tl-val"><code>${s.name.split('.').pop()}</code> <b style="color:#f9e2af">${fmt(v, s.width)}</b></div>`;
    }
    this.valsEl.innerHTML = html;
  }
}

// valeur d'un signal (transitions [{t,v}]) à l'instant t (dernière <= t)
function valAt(trans, t) {
  if (!trans.length) return 'x';
  if (t < trans[0].t) return trans[0].v;
  let lo = 0, hi = trans.length - 1, res = trans[0].v;
  while (lo <= hi) {
    const m = (lo + hi) >> 1;
    if (trans[m].t <= t) { res = trans[m].v; lo = m + 1; }
    else hi = m - 1;
  }
  return res;
}

// formate une string binaire VCD ('1110','x','1') pour affichage
function fmt(v, width) {
  if (v == null) return 'x';
  if (v === 'x' || v.includes('x') || v === 'z' || v.includes('z')) return v;
  if (width === 1) return v;
  if (width <= 4) return v;
  try {
    const n = parseInt(v, 2);
    return '0x' + n.toString(16).toUpperCase().padStart(Math.ceil(width / 4), '0');
  } catch {
    return v;
  }
}