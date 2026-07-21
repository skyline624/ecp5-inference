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
    this.modInfo.textContent = '— (survolez un module)';
  }

  _indexGraph() {
    // map id -> {hier, module, buses_in, buses_out}
    const byId = {};
    for (const [hier, m] of Object.entries(this.graph.modules)) {
      byId[m.id] = { hier, module: m.module, buses_in: [], buses_out: [] };
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

  _highlight(node, sticky = false) {
    const id = node.getAttribute('id');
    const info = this.byId[id];
    if (!info) return;
    // surligne le nœud
    this.graphEl.querySelectorAll('[id^="mod:"]').forEach((n) => n.classList.remove('pt-hl'));
    node.classList.add('pt-hl');
    // surligne les edges incidents
    const hier = info.hier;
    this.graphEl.querySelectorAll('[id^="net:"]').forEach((e) => e.classList.remove('pt-edge-hl'));
    for (const e of this.graph.edges) {
      if (e.from === hier || e.to === hier) {
        const el = this.graphEl.querySelector(`#${cssEscape(e.id)}`);
        if (el) el.classList.add('pt-edge-hl');
      }
    }
    const inB = [...new Set(info.buses_in)];
    const outB = [...new Set(info.buses_out)];
    this.modInfo.innerHTML = `<code>${info.hier}</code><br>: <b>${info.module}</b>`;
    this.busInfo.innerHTML =
      `<div><b>← in</b>: ${inB.length ? inB.map((b) => `<code>${b}</code>`).join(', ') : '—'}</div>` +
      `<div><b>→ out</b>: ${outB.length ? outB.map((b) => `<code>${b}</code>`).join(', ') : '—'}</div>`;
  }

  _unhighlight() {
    // garde le dernier highlight au click (pas de auto-unhighlight au leave)
  }
}

// escape un id CSS pour querySelector (les ids ici sont déjà safe : pas de . >)
function cssEscape(s) {
  return s.replace(/[^a-zA-Z0-9_:-]/g, (c) => `\\${c}`);
}