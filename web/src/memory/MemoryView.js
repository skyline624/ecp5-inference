// MemoryView.js — vue "Mémoires" (architecture triple mémoire du design).
//
// Montre les 3 tiers mémoire du plan §2.2 : flash = poids (RO), SDRAM = KV-cache
// (R/W, grandit), BRAM = calculs internes. Chaque tier est un cluster Graphviz
// contenant ses régions/blocs (tok_emb/layers/KV, wcache_ffn/vfile/xbuf...). Les
// flèches = flux de données entre mémoires (boot, cache FFN, KV read/write).
// Le SVG vient de /api/memory_svg (Graphviz), ids `mem:*`/`flow:*` injectés.
// Au survol d'un tier/région/flux, la description s'affiche dans la sidebar.

import svgPanZoom from 'svg-pan-zoom';
import { getMemory, getMemorySvg } from '../api.js';
import { cssEscape } from '../graph/GraphView.js';

const PHASE_LABEL = {
  flash: 'Flash — poids (RO)',
  sdram: 'SDRAM — KV-cache + miroir (R/W)',
  bram: 'BRAM — calculs internes (R/W)',
};

export class MemoryView {
  constructor(root) {
    root.innerHTML = `
      <div class="pt-layout">
        <div class="pt-toolbar">
          <span class="pt-title">Triple mémoire — flash / SDRAM / BRAM</span>
          <span class="pt-hint">Survolez un tier, une région ou un flux pour son rôle. Lecture gauche → droite : flash → SDRAM → BRAM.</span>
        </div>
        <div class="pt-main">
          <div id="pm-graph" class="pt-graph"></div>
          <div class="pt-sidebar" id="pm-sidebar">
            <div class="pt-side-sec"><b>Élément</b><div id="pm-node-info">— (survolez un tier ou une région)</div></div>
            <div class="pt-side-sec"><b>Flux survolé</b><div id="pm-flow-info">—</div></div>
            <div class="pt-side-sec"><b>Lecture</b>
              <div class="pt-mod-desc">
                Le design sépare 3 mémoires par motif d'accès (plan §2.2) : la
                <b>flash</b> SPI (RO) stocke les poids et les copie vers la
                <b>SDRAM</b> au boot ; la <b>SDRAM</b> (R/W, volatile) héberge le
                miroir des poids <i>et</i> le KV-cache qui grandit avec les tokens
                (étape 1) ; la <b>BRAM</b> on-chip (1 cycle) regroupe les
                scratchpads de calcul (cache FFN, register-file, activations,
                shifts de re-align, ROMs RoPE).
              </div>
            </div>
          </div>
        </div>
      </div>`;
    this.graphEl = root.querySelector('#pm-graph');
    this.nodeInfo = root.querySelector('#pm-node-info');
    this.flowInfo = root.querySelector('#pm-flow-info');
    this.load();
  }

  async load() {
    const [svg, data] = await Promise.all([getMemorySvg(), getMemory()]);
    this.data = data;
    this.graphEl.innerHTML = svg;
    const svgEl = this.graphEl.querySelector('svg');
    if (!svgEl) return;
    svgEl.setAttribute('width', '100%');
    svgEl.setAttribute('height', '100%');
    if (this.panZoom) this.panZoom.destroy();
    this.panZoom = svgPanZoom(svgEl, {
      controlIconsEnabled: true, fit: true, center: true, minZoom: 0.2, maxZoom: 8,
    });
    this._attachHover();
    this._renderLegend();
  }

  // lookup d'un nœud par id (tier ou région)
  _nodeById(id) {
    return this.data.tiers[id] || this.data.regions[id] || null;
  }

  _attachHover() {
    this.graphEl.querySelectorAll('[id^="mem:"]').forEach((n) => {
      n.style.cursor = 'pointer';
      n.addEventListener('mouseenter', () => this._highlightNode(n));
      n.addEventListener('mouseleave', () => this._unhighlight());
      n.addEventListener('click', () => this._highlightNode(n, true));
    });
    this.graphEl.querySelectorAll('[id^="flow:"]').forEach((e) => {
      e.style.cursor = 'help';
      e.addEventListener('mouseenter', () => this._highlightFlow(e));
      e.addEventListener('mouseleave', () => this._unhighlight());
    });
  }

  _highlightNode(node, sticky = false) {
    const id = node.getAttribute('id');
    const n = this._nodeById(id.slice(4));   // enlève "mem:"
    if (!n) return;
    this.graphEl.querySelectorAll('[id^="mem:"]').forEach((x) => x.classList.remove('pt-hl'));
    node.classList.add('pt-hl');
    const tier = n.tier || n.id;   // les régions ont "tier", les tiers sont leur propre tier
    this.nodeInfo.innerHTML =
      `<code>${n.label.replace(/\n/g, ' ')}</code>` +
      `<div class="pt-mod-desc"><b>${PHASE_LABEL[n.phase] || n.phase}</b>` +
      (n.tier ? ` <span class="pt-muted">(${n.tier})</span>` : '') +
      `<br>${n.desc}</div>`;
    // surligne les flux incidents
    this.graphEl.querySelectorAll('[id^="flow:"]').forEach((e) => e.classList.remove('pt-edge-hl'));
    for (const f of this.data.flows) {
      if (f.from === n.id || f.to === n.id) {
        const el = this.graphEl.querySelector(`#${cssEscape(f.id)}`);
        if (el) el.classList.add('pt-edge-hl');
      }
    }
  }

  _highlightFlow(edge) {
    const id = edge.getAttribute('id');
    const f = this.data.flows.find((x) => x.id === id);
    if (!f) return;
    const fromN = this._nodeById(f.from);
    const toN = this._nodeById(f.to);
    const lbl = (x) => x ? x.label.replace(/\n/g, ' ') : f.from;
    this.flowInfo.innerHTML = `<code>${f.label.replace(/\n/g, ' ')}</code>` +
      `<div class="pt-mod-desc"><b>${lbl(fromN)}</b> → <b>${lbl(toN)}</b></div>`;
  }

  _unhighlight() { /* garde le dernier highlight au click */ }

  // Légende overlay des 3 tiers.
  _renderLegend() {
    const phases = [
      ['flash', 'Flash (poids, RO)'],
      ['sdram', 'SDRAM (R/W, KV-cache)'],
      ['bram', 'BRAM (calculs, 1 cyc)'],
    ];
    const items = phases.map(([k, lbl]) =>
      `<span class="pt-leg-item"><span class="pt-leg-swatch pt-phase-${k}"></span>${lbl}</span>`).join('');
    const flow = 'Flux : <b>flash</b> →boot→ <b>SDRAM</b> (poids) ; SDRAM <b>layers</b> →cache→ <b>wcache_ffn</b> ; SDRAM <b>KV</b> ↔ <b>BRAM</b> (read PH_MM / write KVWR)';
    const html = `<div class="pt-legend">
      <div class="pt-leg-title">Mémoires</div>
      <div class="pt-leg-items">${items}</div>
      <div class="pt-leg-flow">${flow}</div>
    </div>`;
    this.graphEl.querySelectorAll('.pt-legend').forEach((e) => e.remove());
    this.graphEl.insertAdjacentHTML('beforeend', html);
  }

  get ready() { return !!this.panZoom; }
}