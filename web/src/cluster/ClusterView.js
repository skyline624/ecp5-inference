// ClusterView.js — vue "Cluster" (Phase D) : 2× i9 tensor-parallel head-parallel
// + ss_link anneau, avec trafic (all-gather / all-reduce) animé.
//
// Montre l'architecture multi-puce du design (plan §2.1-2.4) : 2 cartes i9
// côte à côte (A heads 0..3 / B heads 4..7), chaque carte = séquenceur TP +
// nœud TP + 3 mémoires (flash shard / SDRAM miroir+KV local / BRAM scratch),
// ss_link anneau au milieu. Invariants : KV 100% local, poids shardés
// préchargés (ne traversent jamais le lien), le lien ne porte que des
// activations O(D)/token.
//
// Animation : reprend le timing du pipeline Animate (timeline.js partagé) et,
// aux étapes qui utilisent le lien (ffn_up=all-gather, wo/ffn_down=all-reduce),
// fait couler les 2 ss_link edges (CSS .pt-link-active) + allume les nœuds TP.
// Sync cross-tab : publie le cycle sur window.ptState (l'onglet Animate reprend
// au même cycle). Le SVG vient de /api/cluster_svg (ids node:*/flow:*/sslink:*).

import svgPanZoom from 'svg-pan-zoom';
import { getCluster, getClusterSvg } from '../api.js';
import { cssEscape } from '../graph/GraphView.js';
import { NTOKENS, NLAYERS, buildDemo } from '../pipeline/timeline.js';

const TIER_NAME = { flash: 'Flash', sdram: 'SDRAM', bram: 'BRAM' };
const KIND_LABEL = {
  seq: 'Séquenceur TP', node: 'Nœud TP (compute)',
  flash: 'Mémoire Flash', sdram: 'Mémoire SDRAM', bram: 'Mémoire BRAM',
};

export class ClusterView {
  constructor(root) {
    root.innerHTML = `
      <div class="pt-layout">
        <div class="pt-toolbar">
          <span class="pt-title">Cluster 2× i9 — tensor-parallel head-parallel</span>
          <button id="pc-play" class="pt-btn">▶ play</button>
          <button id="pc-stepb" class="pt-btn">◀ cycle</button>
          <button id="pc-stepf" class="pt-btn">cycle ▶</button>
          <label>cyc/s <input id="pc-speed" type="number" value="180" min="1" max="4000" style="width:54px"></label>
          <input id="pc-seek" type="range" min="0" max="0" value="0" style="flex:1;min-width:160px">
          <span id="pc-cyc" class="pt-muted">0 / 0</span>
        </div>
        <div class="pt-main">
          <div id="pc-graph" class="pt-graph"></div>
          <div class="pt-sidebar" id="pc-sidebar">
            <div class="pt-side-sec"><b>Étape courante</b><div id="pc-now">—</div></div>
            <div class="pt-side-sec"><b>Trafic ss_link</b><div id="pc-link">—</div></div>
            <div class="pt-side-sec"><b>Élément survolé</b><div id="pc-info">— (survolez un nœud ou un lien)</div></div>
            <div class="pt-side-sec"><b>Lecture</b>
              <div class="pt-mod-desc">
                <b>Head-parallel</b> : les 8 heads sont splittées entre les 2
                cartes (A : 0..3, B : 4..7). Le <b>KV-cache suit la frontière des
                heads → 100% local</b> à chaque carte (en SDRAM), il ne traverse
                jamais le lien. Les <b>poids sont shardés et préchargés en
                backdoor</b> (flash→SDRAM au boot) : ils ne traversent jamais le
                lien non plus. Le ss_link ne porte que des <b>activations O(D)
                par token</b> (constant, ne grandit pas avec le contexte) :
                <b>all-gather</b> (W1/W3 row-parallel, ffn_up) et <b>all-reduce</b>
                (W2/WO column-parallel, ffn_down + wo — partiels int32 sommés
                localement + requantés une fois). Hors ces étapes, chaque carte
                calcule localement (lien idle).
              </div>
            </div>
          </div>
        </div>
      </div>`;
    this.graphEl = root.querySelector('#pc-graph');
    this.nowEl = root.querySelector('#pc-now');
    this.linkEl = root.querySelector('#pc-link');
    this.infoEl = root.querySelector('#pc-info');
    this.playBtn = root.querySelector('#pc-play');
    this.stepbBtn = root.querySelector('#pc-stepb');
    this.stepfBtn = root.querySelector('#pc-stepf');
    this.speedInp = root.querySelector('#pc-speed');
    this.seek = root.querySelector('#pc-seek');
    this.cycEl = root.querySelector('#pc-cyc');
    this.iCycle = 0;
    this.playing = false;
    this.load();
    this._bind();
  }

  async load() {
    const [svg, data] = await Promise.all([getClusterSvg(), getCluster()]);
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
    this._buildDemoTimeline();
    this._attachHover();
    this._renderLegend();
    this._render();
  }

  _buildDemoTimeline() {
    const d = buildDemo();
    this.stepAt = d.stepAt;
    this.tokAt = d.tokAt;
    this.layAt = d.layAt;
    this.cycles = this.stepAt.map((_, i) => i);
    // sync cross-tab : reprend le cycle publié par l'onglet Animate si dispo.
    if (window.ptState && Number.isInteger(window.ptState.iCycle)
        && window.ptState.iCycle < this.cycles.length) {
      this.iCycle = window.ptState.iCycle;
    } else {
      this.iCycle = 0;
    }
    this.seek.max = Math.max(0, this.cycles.length - 1);
    this.seek.value = this.iCycle;
  }

  _bind() {
    this.playBtn.onclick = () => this._togglePlay();
    this.stepfBtn.onclick = () => this._step(1);
    this.stepbBtn.onclick = () => this._step(-1);
    this.seek.oninput = () => { this.iCycle = +this.seek.value; this._render(); };
    this.speedInp.onchange = () => { if (this.playing) this._startTick(); };
  }

  _togglePlay() {
    this.playing = !this.playing;
    this.playBtn.textContent = this.playing ? '⏸ pause' : '▶ play';
    if (this.playing) this._startTick(); else this._stopTick();
  }

  _startTick() {
    this._stopTick();
    const cps = Math.max(1, +this.speedInp.value || 180);
    this._tick = setInterval(() => {
      this.iCycle = (this.iCycle + 1) % this.cycles.length;   // boucle (demo)
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
    // publie le cycle pour sync cross-tab avec l'onglet Animate.
    if (window.ptState) window.ptState.setCycle(this.iCycle, this.cycles);
    this.graphEl.querySelectorAll('[id^="node:"]').forEach((n) => n.classList.remove('pt-active'));
    this.graphEl.querySelectorAll('[id^="sslink:"]').forEach((e) => e.classList.remove('pt-link-active'));
    const step = this.stepAt[this.iCycle];
    const act = (this.data.link_activity || {})[step];
    const stepLbl = step ? step.replace(/_/g, ' ') : '— (idle)';
    // position (token/couche) — comme Animate.
    const tok = this.tokAt[this.iCycle];
    const lay = this.layAt[this.iCycle];
    const pos = tok >= 0
      ? `token <b>${tok + 1}/${NTOKENS}</b>` + (lay >= 0 ? ` · couche <b>${lay + 1}/${NLAYERS}</b>` : '')
      : '<span class="pt-muted">boot (power-up, 1×)</span>';
    this.nowEl.innerHTML = `<code>${stepLbl}</code><div class="pt-mod-desc">${pos}</div>`;
    if (act) {
      // trafic lien actif : allume les 2 sslink + les 2 nœuds TP (compute lockstep).
      for (const sl of this.data.sslinks) {
        const el = this.graphEl.querySelector(`#${cssEscape(sl.id)}`);
        if (el) el.classList.add('pt-link-active');
      }
      for (const id of ['a_node', 'b_node']) {
        const el = this.graphEl.querySelector(`#${cssEscape('node:' + id)}`);
        if (el) el.classList.add('pt-active');
      }
      this.linkEl.innerHTML =
        `<span class="pt-mem-chip" data-tier="sdram" style="background:#f38ba8">${act.pattern}</span>` +
        `<div class="pt-mod-desc">${act.desc}</div>`;
    } else {
      this.linkEl.innerHTML = '<span class="pt-muted">lien idle — calcul local (pas de trafic inter-carte)</span>';
    }
  }

  _attachHover() {
    this.graphEl.querySelectorAll('[id^="node:"]').forEach((n) => {
      n.style.cursor = 'pointer';
      n.addEventListener('mouseenter', () => this._highlightNode(n));
      n.addEventListener('mouseleave', () => this._unhighlight());
      n.addEventListener('click', () => this._highlightNode(n, true));
    });
    this.graphEl.querySelectorAll('[id^="sslink:"]').forEach((e) => {
      e.style.cursor = 'help';
      e.addEventListener('mouseenter', () => this._highlightSslink(e));
      e.addEventListener('mouseleave', () => this._unhighlight());
    });
    this.graphEl.querySelectorAll('[id^="flow:"]').forEach((e) => {
      e.style.cursor = 'help';
      e.addEventListener('mouseenter', () => this._highlightFlow(e));
      e.addEventListener('mouseleave', () => this._unhighlight());
    });
  }

  _highlightNode(node, sticky = false) {
    const id = node.getAttribute('id');
    const n = this.data.nodes[id.slice(5)];   // enlève "node:"
    if (!n) return;
    const card = this.data.cards[n.card] || {};
    this.infoEl.innerHTML =
      `<code>${n.label.replace(/\n/g, ' ')}</code>` +
      `<div class="pt-mod-desc"><b>${KIND_LABEL[n.kind] || n.kind}</b> · ` +
      `<span class="pt-muted">${card.label || n.card} (${card.heads || ''})</span><br>${n.desc}</div>`;
  }

  _highlightSslink(edge) {
    const id = edge.getAttribute('id');
    const sl = (this.data.sslinks || []).find((x) => x.id === id);
    if (!sl) return;
    this.infoEl.innerHTML =
      `<code>ss_link ${sl.label}</code><div class="pt-mod-desc">` +
      `Lien source-synchrone parallèle (DW=8, 11 pins = clk + 8 data + valid + ` +
      `full), anneau unidirectionnel. CDC async_fifo (Gray + 2FF) entre clk_tx ` +
      `forwarded et clk_rx locale. <b>${sl.from}</b> → <b>${sl.to}</b>. Porte ` +
      `uniquement des activations O(D)/token (all-gather / all-reduce).</div>`;
  }

  _highlightFlow(edge) {
    const id = edge.getAttribute('id');
    const f = (this.data.card_edges || []).find((x) => x.id === id);
    if (!f) return;
    this.infoEl.innerHTML = `<code>${f.label}</code><div class="pt-mod-desc">` +
      `<b>${f.from}</b> → <b>${f.to}</b></div>`;
  }

  _unhighlight() { /* garde le dernier highlight au click */ }

  _renderLegend() {
    const items = [
      ['seq', 'Séquenceur TP'], ['node', 'Nœud TP (compute)'],
      ['flash', 'Flash (shard)'], ['sdram', 'SDRAM (shard+KV local)'], ['bram', 'BRAM (scratch)'],
    ].map(([k, lbl]) =>
      `<span class="pt-leg-item"><span class="pt-leg-swatch pt-phase-${k}"></span>${lbl}</span>`).join('');
    const flow = '<b>ss_link</b> (rose) : anneau A↔B, activations O(D)/token — <b>all-gather</b> (ffn_up) + <b>all-reduce</b> (wo, ffn_down). Hors ces étapes : lien idle.';
    const html = `<div class="pt-legend">
      <div class="pt-leg-title">Cluster 2× i9</div>
      <div class="pt-leg-items">${items}</div>
      <div class="pt-leg-flow">${flow}</div>
    </div>`;
    this.graphEl.querySelectorAll('.pt-legend').forEach((e) => e.remove());
    this.graphEl.insertAdjacentHTML('beforeend', html);
  }

  get ready() { return !!this.panZoom; }
}