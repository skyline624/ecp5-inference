// FlowView.js — vue "Parcours" : le trajet complet d'un token à travers le
// modèle, avec la métaphore camions/autoroute vivante.
//
// Self-contained (pas de backend). Montre :
//  - le parcours du modèle (token in -> embedding -> NL couches [rmsnorm,
//    attention Q/K/V/RoPE/KV-cache/scores/V-mix/O, rmsnorm, FFN W1/silu/W2]
//    -> rmsnorm final -> lm_head -> token out) ;
//  - le KV-cache comme un "parking" qui grandit avec T (tokens de contexte) ;
//  - l'autoroute animée : W voies, dim camions qui défilent par rangées de W,
//    cycles = dim/W, capacité de transport = W camions/cycle ;
//  - sliders dim / W / T (contexte) / NL (couches) — tout se met à jour live.
//
// Pédagogie : dim = nb de camions à faire passer (longueur du convoi),
// W = nb de voies (largeur de l'autoroute, = béton/LUT), cycles = dim/W,
// KV-cache = parking de T véhicules garés (grandit avec le contexte).

const DIMS = [64, 128, 256, 512];
const WS = [8, 4, 2, 1];
const NL_DEFAULT = 5;        // stories260K

// Camion = petit colis. Couleur selon le "type" de vecteur qui circule.
const C_X = '#89b4fa';    // x (activations)
const C_KV = '#f9e2af';   // K / V (KV cache)
const C_LOGIT = '#a6e3a1'; // logits
const C_W = '#cba6c7';    // poids (camions "poids" sur l'autoroute de la matmul)

// Parcours du modèle (une entrée = une étape affichée). cost = fonction
// (dim,W,T) -> cycles de compute (relatif ; le vrai goulot est la SDRAM).
const STEPS = (dim, W, T) => [
  { k: 'in',   label: 'Token entrant',         sub: '1 id',                camions: 0,  cost: 0 },
  { k: 'emb',  label: 'Embedding',             sub: 'tok_emb[id] -> x',    camions: dim, cost: 0 },
  { k: 'rms1', label: 'RMSNorm',               sub: 'normalise x',         camions: dim, cost: 2 * dim / W },
  { k: 'q',    label: 'Q proj (matvec)',        sub: 'x . Wq',              camions: dim, cost: dim * dim / W },
  { k: 'k',    label: 'K proj (matvec)',        sub: 'x . Wk -> K (park)',  camions: dim, cost: dim * dim / W, park: 'K' },
  { k: 'v',    label: 'V proj (matvec)',        sub: 'x . Wv -> V (park)',  camions: dim, cost: dim * dim / W, park: 'V' },
  { k: 'rope', label: 'RoPE',                  sub: 'rotation Q,K',        camions: dim, cost: dim / W },
  { k: 'att',  label: 'Attention Q·K',         sub: `scores vs ${T} caches`,camions: T,   cost: T * dim / W },
  { k: 'soft', label: 'Softmax',               sub: 'scores -> probas',    camions: T,   cost: T },
  { k: 'vmix', label: 'V mix',                 sub: 'probas . V cache',     camions: dim, cost: T * dim / W },
  { k: 'o',    label: 'O proj (matvec)',       sub: 'attn . Wo',           camions: dim, cost: dim * dim / W },
  { k: 'rms2', label: 'RMSNorm',               sub: 'normalise',          camions: dim, cost: 2 * dim / W },
  { k: 'w1',   label: 'FFN W1 (matvec)',        sub: 'x . W1 (dim->4dim)',   camions: 4 * dim, cost: 4 * dim * dim / W },
  { k: 'silu', label: 'SiLU',                  sub: 'activation',         camions: 4 * dim, cost: 4 * dim / W },
  { k: 'w2',   label: 'FFN W2 (matvec)',        sub: 'h . W2 (4dim->dim)',   camions: dim, cost: 4 * dim * dim / W },
  { k: 'rmsF', label: 'RMSNorm final',         sub: 'normalise',           camions: dim, cost: 2 * dim / W },
  { k: 'lm',   label: 'lm_head (matvec)',      sub: 'x . Wout -> logits',  camions: 0,  cost: 0, vocab: true },
  { k: 'out',  label: 'Token sortant',          sub: 'argmax(logits)',     camions: 0,  cost: 0 },
];

export class FlowView {
  constructor(root) {
    root.innerHTML = `
      <div class="pt-layout">
        <div class="pt-toolbar">
          <span class="pt-title">Parcours du modèle — camions & autoroutes</span>
          <label>dim <select id="fl-dim">${DIMS.map(d=>`<option ${d===128?'selected':''}>${d}</option>`).join('')}</select></label>
          <label>W (voies) <select id="fl-w">${WS.map(w=>`<option ${w===8?'selected':''}>${w}</option>`).join('')}</select></label>
          <label>T (contexte/KV) <input id="fl-t" type="number" value="8" min="1" max="64" style="width:52px"></label>
          <label>NL (couches) <input id="fl-nl" type="number" value="${NL_DEFAULT}" min="1" max="12" style="width:52px"></label>
          <button id="fl-play" class="pt-btn">▶ autoroute</button>
          <span id="fl-sum" class="pt-muted"></span>
        </div>
        <div class="pt-main">
          <div id="fl-canvas" class="pt-graph" style="overflow:auto;padding:14px;background:#181825"></div>
          <div class="pt-sidebar" id="fl-sidebar">
            <div class="pt-side-sec"><b>Survol</b><div id="fl-info">Survolez une étape.</div></div>
            <div class="pt-side-sec"><b>KV-cache (parking)</b><div id="fl-park"></div></div>
            <div class="pt-side-sec"><b>Lecture</b>
              <div class="pt-mod-desc">
                <b>dim</b> = nb de camions à faire passer (longueur du convoi).<br>
                <b>W</b> = nb de voies de l'autoroute (= béton/LUT).<br>
                <b>cycles</b> = dim ÷ W (plus de voies = plus rapide).<br>
                <b>KV-cache</b> = parking de <b>T</b> véhicules garés (K,V des
                tokens précédents). Grandit avec le contexte ; relu à chaque
                attention pour ne pas recalculer le passé.
              </div>
            </div>
          </div>
        </div>
      </div>`;
    this.canvas = root.querySelector('#fl-canvas');
    this.infoEl = root.querySelector('#fl-info');
    this.parkEl = root.querySelector('#fl-park');
    this.sumEl = root.querySelector('#fl-sum');
    this.dimSel = root.querySelector('#fl-dim');
    this.wSel = root.querySelector('#fl-w');
    this.tInp = root.querySelector('#fl-t');
    this.nlInp = root.querySelector('#fl-nl');
    this.playBtn = root.querySelector('#fl-play');
    this.dimSel.onchange = this.wSel.onchange = () => { this.render(); this._resetRoad(); };
    this.tInp.onchange = this.nlInp.onchange = () => this.render();
    this.playBtn.onclick = () => this._toggleRoad();
    this.render();
    this._startRoad();
  }

  _st() {
    return {
      dim: +this.dimSel.value,
      W: +this.wSel.value,
      T: Math.max(1, +this.tInp.value || 1),
      NL: Math.max(1, +this.nlInp.value || 1),
    };
  }

  render() {
    const { dim, W, T, NL } = this._st();
    const base = STEPS(dim, W, T);
    // étapes d'une couche (de rms1 à w2), répétées NL fois, puis rmsF/lm/out.
    const layerSteps = base.filter(s => !['in','emb','rmsF','lm','out'].includes(s.k));
    const tail = base.filter(s => ['rmsF','lm','out'].includes(s.k));
    const head = base.filter(s => ['in','emb'].includes(s.k));
    const steps = [...head];
    for (let l = 0; l < NL; l++) steps.push({ ...layerSteps[0], layer: l+1 }, ...layerSteps.slice(1));
    steps.push(...tail);

    // ---- Parcours vertical (boîtes empilées) ----
    const boxW = 250, boxH = 38, gap = 7, x = 20;
    const rows = steps.map((s, i) => {
      const y = 20 + i * (boxH + gap);
      const cyc = s.vocab ? `${(dim * 512 / W)|0} cyc` : s.cost ? `${Math.round(s.cost)} cyc` : '—';
      const layerTag = s.layer != null ? `<tspan fill="#6c7086" font-size="11"> · couche ${s.layer}/${NL}</tspan>` : '';
      const camions = s.camions
        ? `<tspan fill="#89b4fa" font-size="12">▸ ${s.camions} camions</tspan>`
        : (s.vocab ? `<tspan fill="#a6e3a1" font-size="12">▸ ${512} logits</tspan>` : '');
      const pk = s.park ? `<tspan fill="#f9e2af" font-size="11">  ⏭ ${s.park}→park</tspan>` : '';
      return `<g class="fl-box" data-k="${s.k}" data-i="${i}" style="cursor:help">
        <rect x="${x}" y="${y}" width="${boxW}" height="${boxH}" rx="6"
              fill="rgba(30,30,46,0.9)" stroke="#45475a" stroke-width="1.4"/>
        <text x="${x+10}" y="${y+16}" fill="#cdd6f4" font-size="15" font-weight="600">${s.label}${layerTag}</text>
        <text x="${x+10}" y="${y+31}" fill="#8b8da6" font-size="12">${s.sub}</text>
        <text x="${x+boxW-10}" y="${y+16}" text-anchor="end" fill="#f9e2af" font-size="13">${cyc}</text>
        <text x="${x+boxW-10}" y="${y+31}" text-anchor="end">${camions}${pk}</text>
      </g>`;
    }).join('');

    // flèche verticale de progression (gauche des boîtes)
    const arrowX = x - 12;
    const arrow = `<line x1="${arrowX}" y1="30" x2="${arrowX}" y2="${20 + steps.length*(boxH+gap) - gap - 10}"
        stroke="#45475a" stroke-width="1.2" marker-end="url(#fl-arr)"/>
      <defs><marker id="fl-arr" markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto">
        <path d="M0 0 L8 4 L0 8 z" fill="#45475a"/></marker></defs>`;

    // ---- Autoroute animée (sous le parcours) ----
    const roadY = 30 + steps.length * (boxH + gap) + 10;
    const roadSvg = this._roadSvg(dim, W, roadY);

    const totalCyc = steps.reduce((s, st) => s + (st.vocab ? dim*512/W : st.cost || 0), 0);
    const svg = `<svg width="100%" height="${roadY + 200}" viewBox="0 0 760 ${roadY + 200}" preserveAspectRatio="xMinYMin meet" font-family="Helvetica Neue,Arial,sans-serif">
      ${arrow}${rows}
      ${roadSvg}
    </svg>`;
    this.canvas.innerHTML = svg;
    this.canvas.querySelectorAll('.fl-box').forEach(el => {
      el.addEventListener('mouseenter', () => this._hover(+el.dataset.i));
    });

    // ---- Sidebar : KV-cache parking + total ----
    this._renderPark(dim, T);
    this.sumEl.innerHTML = `<b>compute</b> ≈ ${Math.round(totalCyc).toLocaleString()} cyc/tok ·
      <b>autoroute</b> ${W} voies × ${Math.ceil(dim/W)} rangées = ${dim} camions / convoi`;

    this._steps = steps;
  }

  // ---- KV-cache : parking de T véhicules (K + V) ----
  _renderPark(dim, T) {
    const cells = [];
    for (let i = 0; i < T; i++) {
      const fill = i === T - 1 ? 'rgba(249,226,175,0.5)' : 'rgba(249,226,175,0.18)';
      cells.push(`<span class="pt-mem-cell pt-mem-active" data-tier="bram"
        style="border-color:#f9e2af;background:${fill}">t${i}${i===T-1?' ★':''}</span>`);
    }
    this.parkEl.innerHTML =
      `<div class="pt-mem-cells" style="margin-bottom:6px">${cells.join('')}</div>
       <div class="pt-mod-desc">T = ${T} tokens garés. Chaque entrée = K + V de
       dimension ${this._st().dim}. À chaque nouveau token, l'attention lit les
       ${T} K du parking pour scorer, sans recalculer le passé. Le parking
       <b>grandit avec le contexte</b> (mais pas avec W ni le béton).</div>`;
  }

  // ---- Autoroute animée : W voies, dim camions par rangées de W ----
  _roadSvg(dim, W, y0) {
    const W2 = Math.min(W, 8);                 // max 8 voies affichées
    const laneH = 22, laneW = 680, laneX = 40;
    const lanes = [];
    for (let i = 0; i < W2; i++) {
      const ly = y0 + 30 + i * laneH;
      lanes.push(`<line x1="${laneX}" y1="${ly+laneH/2}" x2="${laneX+laneW}" y2="${ly+laneH/2}"
        stroke="#45475a" stroke-width="1" stroke-dasharray="6 4"/>`);
    }
    // bornes entrée/sortie
    const top = y0 + 30, bot = y0 + 30 + W2 * laneH;
    const gate = `<line x1="${laneX}" y1="${top-6}" x2="${laneX}" y2="${bot+6}" stroke="#89b4fa" stroke-width="2.5"/>
      <text x="${laneX-4}" y="${top-10}" fill="#89b4fa" font-size="13" text-anchor="middle">in</text>
      <line x1="${laneX+laneW}" y1="${top-6}" x2="${laneX+laneW}" y2="${bot+6}" stroke="#a6e3a1" stroke-width="2.5"/>
      <text x="${laneX+laneW+4}" y="${top-10}" fill="#a6e3a1" font-size="13" text-anchor="middle">out</text>`;
    const title = `<text x="${laneX}" y="${y0+18}" fill="#cdd6f4" font-size="16" font-weight="600">
      Autoroute : ${W} voies · convoi = ${dim} camions en ${Math.ceil(dim/W)} rangées de ${W}</text>`;
    const cap = `<text x="${laneX+laneW}" y="${y0+18}" text-anchor="end" fill="#f9e2af" font-size="14">
      capacité = ${W} camions / cycle</text>`;
    // conteneur des camions (rempli par l'animation)
    return `<g id="fl-road">
      ${title}${cap}
      ${lanes}${gate}
      <g id="fl-trucks"></g>
      <text id="fl-roadcount" x="${laneX+laneW/2}" y="${bot+22}" text-anchor="middle" fill="#8b8da6" font-size="13">—</text>
    </g>`;
  }

  _toggleRoad() {
    this._playing = !this._playing;
    this.playBtn.textContent = this._playing ? '⏸ autoroute' : '▶ autoroute';
  }
  _startRoad() {
    this._playing = true;
    this._t = 0; this._rang = 0;
    const loop = () => {
      this._raf = requestAnimationFrame(loop);
      if (!this._playing) return;
      this._tickRoad();
    };
    loop();
  }
  _resetRoad() { this._t = 0; this._rang = 0; }

  _tickRoad() {
    const { dim, W } = this._st();
    const W2 = Math.min(W, 8);
    const trucksG = this.canvas.querySelector('#fl-trucks');
    const countEl = this.canvas.querySelector('#fl-roadcount');
    if (!trucksG) return;
    const laneX = 40, laneW = 680, laneH = 22, y0 = (this._steps?.length||0)*45 + 40;
    this._t += 1;
    // un "cycle" tous les ~8 frames : on pousse une nouvelle rangée de W camions.
    if (this._t % 8 === 0) this._rang = (this._rang + 1);
    const rows = Math.ceil(dim / W);
    const rang = this._rang % (rows + 2);   // +2 pour vider l'autoroute
    // dessine les rangées actuellement sur l'autoroute (les 4 dernières)
    let svg = '';
    for (let r = 0; r < 4; r++) {
      const rr = rang - r;
      if (rr < 0 || rr >= rows) continue;
      const prog = (this._t % 8) / 8 + r;     // position 0..~4 (avance vers out)
      const x = laneX + Math.min(prog / 4, 1) * laneW - 14;
      const baseCamion = rr * W;
      for (let w = 0; w < W2; w++) {
        const idx = baseCamion + w;
        if (idx >= dim) break;
        const ly = y0 + 30 + w * laneH + laneH/2 - 10;
        const val = (idx * 7 + 3) & 0xff;      // cosmétique
        svg += `<g><rect x="${x}" y="${ly}" width="26" height="20" rx="4"
          fill="${C_X}" opacity="${0.95 - r*0.18}" stroke="#1e1e2e" stroke-width="0.6"/>
          <text x="${x+13}" y="${ly+14}" text-anchor="middle" fill="#1e1e2e" font-size="11" font-weight="700">${val}</text></g>`;
      }
    }
    trucksG.innerHTML = svg;
    if (countEl) countEl.textContent =
      `rangée ${Math.min(rang+1, rows)} / ${rows}  ·  camions passés ${Math.min(rang*W, dim)} / ${dim}`;
  }

  _hover(i) {
    const s = this._steps?.[i];
    if (!s) return;
    const { dim, W, T } = this._st();
    const detail = s.vocab
      ? `Matvec final x·Wout : produit ${dim} activations × ${512} mots de vocabulaire = ${Math.round(dim*512/W)} cycles.`
      : s.cost
        ? `${s.camions ? s.camions + ' camions à faire passer. ' : ''}Compute ≈ ${Math.round(s.cost)} cycles (${Math.ceil((s.camions||1)/W)} rangées de ${W}).`
        : (s.camions ? `${s.camions} camions.` : 'Lookup, hors datapath.');
    this.infoEl.innerHTML = `<code>${s.label}</code>
      <div class="pt-mod-desc">${s.sub}<br>${detail}</div>`;
  }

  onShow() { this.render(); }
  get ready() { return true; }
}