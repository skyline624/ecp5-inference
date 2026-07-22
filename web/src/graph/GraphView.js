// GraphView.js — Phase A : rend le graphe de modules RTL (SVG) + pan/zoom + sidebar.
// Le SVG vient de /api/graph_svg/{top} (Graphviz), les ids `mod:*`/`net:*` sont
// déjà injectés -> on les sélectionne par querySelector. Prépare la Phase C
// (overlay) en exposant les nœuds/edges par leur hier.

import svgPanZoom from 'svg-pan-zoom';
import { getGraph, getGraphSvg, getTops } from '../api.js';

export class GraphView {
  constructor(root) {
    root.innerHTML = `
      <div class="pt-layout">
        <div class="pt-toolbar">
          <select id="pt-top"></select>
          <span class="pt-title" id="pt-title">FPGA Packet Tracer</span>
          <span class="pt-hint">Phase A — graphe RTL. Survolez un module pour surligner ses buses.</span>
        </div>
        <div class="pt-main">
          <div id="pt-graph" class="pt-graph"></div>
          <div class="pt-sidebar" id="pt-sidebar">
            <div class="pt-side-sec"><b>Module</b><div id="pt-mod-info">—</div></div>
            <div class="pt-side-sec"><b>Buses (edges sélectionnés)</b><div id="pt-bus-info">—</div></div>
            <div class="pt-side-sec"><b>Signaux internes notables</b>
              <div class="pt-int" id="pt-internals">
                <code>state[8:0]</code>, <code>op_sel[3:0]</code>,
                <code>wcache_filling</code>, <code>booting</code>,
                <code>ffn_hit_q</code>, <code>run_mode</code>
                <br><span class="pt-muted">(FSM interne au node_top.v — panneau Phase C)</span>
              </div>
            </div>
          </div>
        </div>
      </div>`;
    this.graphEl = root.querySelector('#pt-graph');
    this.sideEl = root.querySelector('#pt-sidebar');
    this.modInfo = root.querySelector('#pt-mod-info');
    this.busInfo = root.querySelector('#pt-bus-info');
    this.topSel = root.querySelector('#pt-top');
    this.titleEl = root.querySelector('#pt-title');
    this._bindTopSelect();
  }

  async _bindTopSelect() {
    const t = await getTops();
    this.topSel.innerHTML = t.tops
      .map((x) => `<option value="${x.name}" ${x.available ? '' : 'disabled'}>${x.name}${x.available ? '' : ' (n/a)'}</option>`)
      .join('');
    this.topSel.onchange = () => this.load(this.topSel.value);
    this.load(t.tops.find((x) => x.available).name);
  }

  async load(top) {
    this.top = top;
    this.titleEl.textContent = `FPGA Packet Tracer — ${top}`;
    const svg = await getGraphSvg(top);
    this.graphEl.innerHTML = svg;
    const svgEl = this.graphEl.querySelector('svg');
    if (!svgEl) return;
    svgEl.setAttribute('width', '100%');
    svgEl.setAttribute('height', '100%');
    // svg-pan-zoom a besoin d'un viewBox ; Graphviz en met un par défaut.
    if (this.panZoom) this.panZoom.destroy();
    this.panZoom = svgPanZoom(svgEl, {
      controlIconsEnabled: true,
      fit: true,
      center: true,
      minZoom: 0.2,
      maxZoom: 8,
    });
    this.graph = await getGraph(top);
    this._indexGraph();
    this._attachHover();
    this._applyPhases();
    this._renderLegend();
    this.modInfo.textContent = '— (survolez un module)';
  }

  _indexGraph() {
    // map id -> {hier, module, buses_in, buses_out, phase, desc}
    const byId = {};
    for (const [hier, m] of Object.entries(this.graph.modules)) {
      byId[m.id] = {
        hier, module: m.module, buses_in: [], buses_out: [],
        phase: m.phase || '', desc: m.desc || '',
      };
    }
    for (const e of this.graph.edges) {
      if (byId[this.graph.modules[e.from].id]) byId[this.graph.modules[e.from].id].buses_out.push(...e.buses);
      if (byId[this.graph.modules[e.to].id]) byId[this.graph.modules[e.to].id].buses_in.push(...e.buses);
    }
    this.byId = byId;
  }

  _attachHover() {
    const nodes = this.graphEl.querySelectorAll('[id^="mod:"]');
    nodes.forEach((n) => {
      n.style.cursor = 'pointer';
      n.addEventListener('mouseenter', () => this._highlight(n));
      n.addEventListener('mouseleave', () => this._unhighlight());
      n.addEventListener('click', () => this._highlight(n, true));
    });
  }

  // Phase C : colore chaque nœud module selon sa phase fonctionnelle + ajoute un
  // tooltip natif (<title>) avec la description pédagogique.
  _applyPhases() {
    const nodes = this.graphEl.querySelectorAll('[id^="mod:"]');
    nodes.forEach((n) => {
      const info = this.byId[n.getAttribute('id')];
      if (!info) return;
      if (info.phase) n.classList.add(`pt-phase-${info.phase}`);
      // tooltip natif : remplace le <title> Graphviz (label brut) par la desc.
      let t = n.querySelector('title');
      if (!t) { t = document.createElementNS('http://www.w3.org/2000/svg', 'title'); n.appendChild(t); }
      t.textContent = info.desc
        ? `${info.hier.replace(`${this.top}.`, '')} : ${info.module}\n${info.desc}`
        : `${info.hier} : ${info.module}`;
    });
  }

  // Phase C : légende overlay des phases (couleurs + ordre du flux d'un token).
  _renderLegend() {
    const phases = [
      ['seq', 'Séquenceur (FSM)'],
      ['io', 'FIFOs / ss_link'],
      ['mem', 'Mémoire (SDRAM/cache)'],
      ['norm', 'RMSNorm'],
      ['attn', 'Attention (rope/attn/soft)'],
      ['act', 'Activation (silu)'],
      ['mac', 'Matmul (fm_mac)'],
      ['clk', 'Clocking (PLL)'],
      ['boot', 'Boot flash'],
    ];
    const items = phases.map(([k, lbl]) =>
      `<span class="pt-leg-item"><span class="pt-leg-swatch pt-phase-${k}"></span>${lbl}</span>`).join('');
    const flow = 'Flux d\'un token : <b>embed</b> → rms → <b>QKV</b> → rope → <b>attn</b> → out → rms → <b>FFN</b> (w1·silu·w2) → rms → <b>lm_head</b> → <b>argmax</b> (u_seq) → token suivant';
    const comms = '<b>Communications runtime</b> (flèches pointillées entre modules frères) : '
      + '<span style="color:#89b4fa">━ ━ cmd</span> '
      + '<span style="color:#a6e3a1">━ ━ resp</span> '
      + '<span style="color:#f38ba8">⋯ ss_link</span> '
      + '<span style="color:#fab387">⋯ boot</span>';
    const html = `<div class="pt-legend">
      <div class="pt-leg-title">Modules par rôle</div>
      <div class="pt-leg-items">${items}</div>
      <div class="pt-leg-flow">${flow}</div>
      <div class="pt-leg-flow">${comms}</div>
    </div>`;
    // recrée la légende à chaque load (le SVG est régénéré, mais pas le conteneur).
    this.graphEl.querySelectorAll('.pt-legend').forEach((e) => e.remove());
    this.graphEl.insertAdjacentHTML('beforeend', html);
  }

  _highlight(node, sticky = false) {
    const id = node.getAttribute('id');
    const info = this.byId[id];
    if (!info) return;
    // surligne le nœud
    this.graphEl.querySelectorAll('[id^="mod:"]').forEach((n) => n.classList.remove('pt-hl'));
    node.classList.add('pt-hl');
    // surligne les edges incidents (contenance + communications runtime)
    const hier = info.hier;
    this.graphEl.querySelectorAll('[id^="net:"]').forEach((e) => e.classList.remove('pt-edge-hl'));
    this.graphEl.querySelectorAll('[id^="comm:"]').forEach((e) => e.classList.remove('pt-edge-hl'));
    for (const e of this.graph.edges) {
      if (e.from === hier || e.to === hier) {
        const el = this.graphEl.querySelector(`#${cssEscape(e.id)}`);
        if (el) el.classList.add('pt-edge-hl');
      }
    }
    // comm_edges : communications logiques (frères) — cmd/resp/ss_link/boot
    const comms = this.graph.comm_edges || [];
    const commHit = comms.filter((e) => e.from === hier || e.to === hier);
    for (const e of commHit) {
      const el = this.graphEl.querySelector(`#${cssEscape(e.id)}`);
      if (el) el.classList.add('pt-edge-hl');
    }
    const inB = [...new Set(info.buses_in)];
    const outB = [...new Set(info.buses_out)];
    const shortHier = info.hier.startsWith(`${this.top}.`) ? info.hier.slice(this.top.length + 1) : info.hier;
    this.modInfo.innerHTML = `<code>${shortHier}</code><br>: <b>${info.module}</b>` +
      (info.desc ? `<div class="pt-mod-desc">${info.desc}</div>` : '');
    this.busInfo.innerHTML =
      `<div><b>← in</b>: ${inB.length ? inB.map((b) => `<code>${b}</code>`).join(', ') : '—'}</div>` +
      `<div><b>→ out</b>: ${outB.length ? outB.map((b) => `<code>${b}</code>`).join(', ') : '—'}</div>`;
  }

  _unhighlight() {
    // garde le dernier highlight au click (pas de auto-unhighlight au leave)
  }

  // --- API publique Phase C : coloration pilotée de l'extérieur (AnimateView) ---
  // Colorer un nœud module ou edge par son id CSS (`mod:...` / `net:...`).
  highlightById(id, cls = 'pt-active') {
    const el = this.graphEl.querySelector(`#${cssEscape(id)}`);
    if (el) el.classList.add(cls);
    return el;
  }

  // Retirer toutes les occurrences d'une classe de coloration sur le graphe.
  clearHighlights(cls = 'pt-active') {
    this.graphEl.querySelectorAll(`.${cls}`).forEach((el) => el.classList.remove(cls));
  }

  // L'instance svg-pan-zoom est-elle prête (graphe chargé) ?
  get ready() {
    return !!this.panZoom;
  }
}

// escape un id CSS pour querySelector. Les ids graphe contiennent ':' (prefix
// `mod:`/`net:`) qui est un méta-caractère CSS (pseudo-classe) -> doit être
// échappé en `\:`. On privilégie CSS.escape (natif) qui gère tous les cas.
export function cssEscape(s) {
  if (typeof CSS !== 'undefined' && CSS.escape) return CSS.escape(s);
  return s.replace(/[^a-zA-Z0-9_-]/g, (c) => `\\${c}`);
}