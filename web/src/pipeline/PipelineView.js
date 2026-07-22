// PipelineView.js — vue "Flux pipeline" (dataflow d'un token).
//
// Contrairement au graphe d'instanciation (GraphView, qui montre la hiérarchie
// module/contenance), cette vue montre le FLUX de données d'un token à travers
// le transformer : embed → rms → QKV → rope → attention → FFN → lm_head →
// argmax, avec les vecteurs (XB, XN, Q/K/V, ATT, logits...) sur les arêtes.
// Le SVG vient de /api/pipeline_svg (Graphviz), ids `step:*`/`vec:*` injectés.
// Au survol d'une étape, la description pédagogique s'affiche dans la sidebar.

import svgPanZoom from 'svg-pan-zoom';
import { getPipeline, getPipelineSvg } from '../api.js';
import { cssEscape } from '../graph/GraphView.js';

const PHASE_LABEL = {
  io: 'Prompt / IO',
  mem: 'Mémoire',
  norm: 'RMSNorm',
  mac: 'Matmul',
  attn: 'Attention',
  act: 'Activation',
  seq: 'Séquenceur',
};

const TIER_NAME = { flash: 'Flash', sdram: 'SDRAM', bram: 'BRAM' };

export class PipelineView {
  constructor(root) {
    root.innerHTML = `
      <div class="pt-layout">
        <div class="pt-toolbar">
          <span class="pt-title">Flux pipeline — dataflow d'un token</span>
          <span class="pt-hint">Survolez une étape pour son rôle. Les flèches = vecteurs de données.</span>
        </div>
        <div class="pt-main">
          <div id="pp-graph" class="pt-graph"></div>
          <div class="pt-sidebar" id="pp-sidebar">
            <div class="pt-side-sec"><b>Étape</b><div id="pp-step-info">— (survolez une étape)</div></div>
            <div class="pt-side-sec"><b>Mémoires accédées</b><div id="pp-mem-info">—</div></div>
            <div class="pt-side-sec"><b>Vecteur survolé</b><div id="pp-vec-info">—</div></div>
            <div class="pt-side-sec"><b>Lecture</b>
              <div class="pt-mod-desc">
                Un token entre par le <b>prompt</b>, est plongé en vecteur
                (<code>embed</code>), puis traverse <b>5 couches</b> (cluster
                rouge) : RMSNorm → QKV → RoPE → écriture KV cache → attention →
                projection WO → résiduel → RMSNorm → FFN (up·SiLU·down) →
                résiduel. Après la 5ᵉ couche, RMSNorm final → <b>lm_head</b> →
                <b>argmax</b> donne le token émis, qui reboucle vers embed
                (token suivant, jusqu'à 17 tokens).
              </div>
            </div>
          </div>
        </div>
      </div>`;
    this.graphEl = root.querySelector('#pp-graph');
    this.sideEl = root.querySelector('#pp-sidebar');
    this.stepInfo = root.querySelector('#pp-step-info');
    this.memInfo = root.querySelector('#pp-mem-info');
    this.vecInfo = root.querySelector('#pp-vec-info');
    this.load();
  }

  async load() {
    const [svg, data] = await Promise.all([getPipelineSvg(), getPipeline()]);
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

  _attachHover() {
    this.graphEl.querySelectorAll('[id^="step:"]').forEach((n) => {
      n.style.cursor = 'pointer';
      n.addEventListener('mouseenter', () => this._highlightStep(n));
      n.addEventListener('mouseleave', () => this._unhighlight());
      n.addEventListener('click', () => this._highlightStep(n, true));
    });
    this.graphEl.querySelectorAll('[id^="vec:"]').forEach((e) => {
      e.style.cursor = 'help';
      e.addEventListener('mouseenter', () => this._highlightVec(e));
      e.addEventListener('mouseleave', () => this._unhighlight());
    });
  }

  _highlightStep(node, sticky = false) {
    const id = node.getAttribute('id');
    const step = this.data.steps[id.slice(5)];
    if (!step) return;
    this.graphEl.querySelectorAll('[id^="step:"]').forEach((n) => n.classList.remove('pt-hl'));
    node.classList.add('pt-hl');
    this.stepInfo.innerHTML =
      `<code>${step.label.replace(/\n/g, ' ')}</code>` +
      `<div class="pt-mod-desc"><b>Phase :</b> ${PHASE_LABEL[step.phase] || step.phase}<br>${step.desc}</div>`;
    this._renderMem(step);
    // surligne les arêtes incidentes
    this.graphEl.querySelectorAll('[id^="vec:"]').forEach((e) => e.classList.remove('pt-edge-hl'));
    for (const v of this.data.vecs) {
      if (v.from === step.id || v.to === step.id) {
        const el = this.graphEl.querySelector(`#${cssEscape(v.id)}`);
        if (el) el.classList.add('pt-edge-hl');
      }
    }
  }

  _highlightVec(edge) {
    const id = edge.getAttribute('id');
    const v = this.data.vecs.find((x) => x.id === id);
    if (!v) return;
    this.vecInfo.innerHTML = `<code>${v.label}</code><div class="pt-mod-desc">` +
      `<b>${v.from}</b> → <b>${v.to}</b></div>`;
  }

  // Liste les accès mémoire de l'étape (chips tier + région + op R/W).
  _renderMem(step) {
    const mem = step.mem || [];
    if (!mem.length) {
      this.memInfo.innerHTML = '<span class="pt-muted">— (aucune accès mémoire)</span>';
      return;
    }
    const regions = this.data.mem_regions || {};
    const rows = mem.map((a) => {
      const reg = regions[a.region] || { label: a.region, tier: '?' };
      const op = a.op === 'write' ? 'W' : 'R';
      const name = TIER_NAME[reg.tier] || reg.tier;
      return `<div style="margin-bottom:3px"><span class="pt-mem-chip" data-tier="${reg.tier}">${name} · ${op}</span>` +
        `<code>${reg.label}</code></div>`;
    });
    this.memInfo.innerHTML = `<div class="pt-mod-desc">${rows.join('')}</div>`;
  }

  _unhighlight() { /* garde le dernier highlight au click */ }

  // Légende overlay des phases + tiers mémoire + ordre du flux (rappel textuel).
  _renderLegend() {
    const phases = [
      ['io', 'Prompt / IO'], ['mem', 'Mémoire'], ['norm', 'RMSNorm'],
      ['mac', 'Matmul'], ['attn', 'Attention'], ['act', 'Activation'], ['seq', 'Séquenceur'],
    ];
    const items = phases.map(([k, lbl]) =>
      `<span class="pt-leg-item"><span class="pt-leg-swatch pt-phase-${k}"></span>${lbl}</span>`).join('');
    const tiers = [
      ['flash', 'Flash (poids)'], ['sdram', 'SDRAM (KV + poids)'], ['bram', 'BRAM (calculs)'],
    ];
    const tierItems = tiers.map(([k, lbl]) =>
      `<span class="pt-leg-item"><span class="pt-leg-swatch pt-phase-${k}"></span>${lbl}</span>`).join('');
    const flow = 'Flux : <b>embed</b> → rms → <b>QKV</b> → rope → kv-write → <b>attn</b> → WO → résiduel → rms → <b>FFN</b> → résiduel <i>(×5 couches)</i> → rms → <b>lm_head</b> → <b>argmax</b> → token (boucle)';
    const html = `<div class="pt-legend">
      <div class="pt-leg-title">Étapes par phase</div>
      <div class="pt-leg-items">${items}</div>
      <div class="pt-leg-title" style="margin-top:4px">Mémoires (pastilles sur les nœuds)</div>
      <div class="pt-leg-items">${tierItems}</div>
      <div class="pt-leg-flow">${flow}</div>
    </div>`;
    this.graphEl.querySelectorAll('.pt-legend').forEach((e) => e.remove());
    this.graphEl.insertAdjacentHTML('beforeend', html);
  }

  get ready() { return !!this.panZoom; }
}