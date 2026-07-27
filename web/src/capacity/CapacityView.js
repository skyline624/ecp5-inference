// CapacityView.js — vue "Capacité" : pourquoi dim -> LUT, et jusqu'où on peut
// pousser dim sur une seule ECP5-45.
//
// Self-contained (pas de backend) : données = mesures synthèse yosys
// `synth_ecp5 -noflatten` dim128 (commit e21d94b), calibrées pour retomber sur
// 32196 LUT4 = 77% des 41664 LUT de l'ECP5-45F (synthèse ; le P&R nextpnr
// consomme un peu plus). Le modèle décompose chaque bloc
// selon ce qui scale avec W·D (registres datapath), D (opérateurs vectoriels),
// W (lanes matmul), ou fixe (FSM, périphériques), et applique un facteur
// d'optimisation d'aplatissement (0.869) mesuré à dim128.
//
// 3 leviers actionnables (slider W, toggle BRAM) montrent comment regagner
// assez de LUT pour dim256/512 single-FPGA — validés ensuite par sim Verilator
// (oracle bit-exact). Voir docs/CAPACITE_FPGA.md et memory regression-postfusion.

// ---- Modèle LUT calibré dim128 (W=8, D=128) ----
// Décomposition par module (définitions, -noflatten) :
//   top (node_top local) = 10995  -> scale W·D  (registres xbuf/wbuf W*D bits)
//   gen_seq              = 9319   -> 80% fixe (FSM) + 20% scale D (compteurs)
//   vec_alu2 (matmul)    = 5467   -> scale W (lanes)
//   attention_head_op    = 4633   -> scale D
//   softmax+rmsnorm+silu+rope = 3074 -> scale D (datapath vectoriel + LUTs)
//   periph (flash_boot/sdram/uart/wrapper) = 3500 -> fixe
// Somme = 37043 ; aplati réel = 32196 -> opt = 0.869.
const OPT = 32196 / 37043;
const TOTAL_LUT = 41664;       // ECP5-45F
const REF_D = 128, REF_W = 8;

const BLOCS = [
  { key: 'top',   label: 'top (node_top) — registres xbuf/wbuf', color: '#89b4fa',
    base: 10995, kind: 'WD', desc: 'Datapath du nœud : registres d\'entrée xbuf/wbuf larges de W·D bits. Double avec dim ET avec W. C\'est le gros consommateur.' },
  { key: 'gseq',  label: 'gen_seq — FSM + lm_head', color: '#f9e2af',
    base: 9319, kind: 'mix', fix: 0.8, desc: 'Séquenceur (FSM) ~fixe + lm_head/argmax + compteurs qui scale un peu avec D. Le contrôleur ne grandit presque pas avec dim.' },
  { key: 'vec',   label: 'vec_alu2 — matmul W lanes', color: '#a6e3a1',
    base: 5467, kind: 'W', desc: 'Datapath du matvec : W lanes parallèles. Scale avec W (le parallélisme), pas avec dim.' },
  { key: 'attn',  label: 'attention_head_op', color: '#f38ba8',
    base: 4633, kind: 'D', desc: 'Q·K, softmax, V mix (1 instance partagée entre les heads). Datapath vectoriel large de D. Scale avec dim.' },
  { key: 'ops',   label: 'softmax+rmsnorm+silu+rope', color: '#fab387',
    base: 3074, kind: 'D', desc: 'Opérateurs vectoriels + LUTs (exp/inv/rsqrt/silu) indexées sur D. Scale avec dim. Levier BRAM : migrer les LUTs/ROM en BRAM sous-utilisées (10/208).' },
  { key: 'perf',  label: 'Périphériques (flash/sdram/uart)', color: '#6c7086',
    base: 3500, kind: 'fix', desc: 'Boot flash, contrôleur SDRAM, UART, wrapper. ~constant quelle que soit dim.' },
];

function lutFor(bloc, D, W, bramLevier) {
  const r = bloc.base;
  switch (bloc.kind) {
    case 'WD':  return r * (W * D) / (REF_W * REF_D);
    case 'D': {
      // Levier BRAM : migre les LUTs/ROM (exp/inv/rsqrt/silu + RoPE) vers BRAM.
      // Ces tables vivent surtout dans ops (softmax+rmsnorm+silu, ~35% de leur
      // LUT) et un peu dans attn (exp/inv + RoPE, ~12%). Le reste est du datapath
      // de calcul (additionneurs/compareteurs) qui ne part PAS en BRAM.
      const f = bramLevier ? (bloc.key === 'ops' ? 0.65 : bloc.key === 'attn' ? 0.88 : 1) : 1;
      return r * D / REF_D * f;
    }
    case 'W':   return r * W / REF_W;
    case 'mix': return r * (bloc.fix + (1 - bloc.fix) * D / REF_D);
    case 'fix': return r;
  }
  return r;
}

function totalLut(D, W, bramLevier) {
  return OPT * BLOCS.reduce((s, b) => s + lutFor(b, D, W, bramLevier), 0);
}

const DIMS = [64, 128, 256, 512];
const WS = [8, 4, 2, 1];

export class CapacityView {
  constructor(root) {
    root.innerHTML = `
      <div class="pt-layout">
        <div class="pt-toolbar">
          <span class="pt-title">Capacité LUT vs dim — ECP5-45F (41 664 LUT)</span>
          <label>dim <select id="cap-dim">${DIMS.map(d=>`<option ${d===128?'selected':''}>${d}</option>`).join('')}</select></label>
          <label>W (lanes) <select id="cap-w">${WS.map(w=>`<option ${w===8?'selected':''}>${w}</option>`).join('')}</select></label>
          <label><input id="cap-bram" type="checkbox"> Migrer LUTs/ROM → BRAM</label>
          <button id="cap-anim" class="pt-btn">▶ sweep dim</button>
          <span id="cap-verdict" class="pt-muted"></span>
        </div>
        <div class="pt-main">
          <div id="cap-canvas" class="pt-graph" style="overflow:auto;padding:14px"></div>
          <div class="pt-sidebar" id="cap-sidebar">
            <div class="pt-side-sec"><b>Survol</b><div id="cap-info">Survolez un bloc.</div></div>
            <div class="pt-side-sec"><b>Pourquoi dim → LUT ?</b>
              <div class="pt-mod-desc">
                Le datapath fait un matvec <code>y = W·x</code>, <code>x ∈ ℝ^dim</code>,
                avec <code>W</code> éléments de x traités en parallèle par cycle.
                La <b>largeur physique</b> des bus/registres est <code>W·dim</code> <b>bits</b>.
                <ul style="margin:6px 0 0 14px;padding:0">
                  <li><b>xbuf/wbuf</b> (W·D bits) : ×2 si dim×2 → ×2 bascules/LUT.</li>
                  <li><b>Opérateurs</b> (largeur D) : ×2 si dim×2.</li>
                  <li><b>FSM</b> (gen_seq) : ~fixe — le contrôleur ne change pas.</li>
                </ul>
                ~80% du design est <b>linéaire en dim</b>. dim128→256 double les
                blocs qui scalent (~25k LUT) → ~50k &gt; 41 664 = déborde.
              </div>
            </div>
            <div class="pt-side-sec"><b>Leviers</b>
              <div class="pt-mod-desc">
                <b>W↓</b> : vec_alu2 + registres W·D. W=4 ≈ −3-4k LUT, coût ~2× cycles.<br>
                <b>BRAM</b> : LUTs/ROM (exp/inv/rsqrt/silu) → BRAM (10/208 utilisé). ~−40% sur ops/attn, 0 cycle.<br>
                <b>Buffers → BRAM</b> : xbuf/wbuf/KV en BRAM au lieu de LUT-RAM.
              </div>
            </div>
          </div>
        </div>
      </div>`;
    this.canvas = root.querySelector('#cap-canvas');
    this.infoEl = root.querySelector('#cap-info');
    this.verdictEl = root.querySelector('#cap-verdict');
    this.dimSel = root.querySelector('#cap-dim');
    this.wSel = root.querySelector('#cap-w');
    this.bramCb = root.querySelector('#cap-bram');
    this.animBtn = root.querySelector('#cap-anim');
    this.dimSel.onchange = this.wSel.onchange = this.bramCb.onclick = () => this.render();
    this.animBtn.onclick = () => this._toggleSweep();
    this.render();
  }

  _state() {
    return {
      D: +this.dimSel.value,
      W: +this.wSel.value,
      bram: this.bramCb.checked,
    };
  }

  render() {
    const { D, W, bram } = this._state();
    const parts = BLOCS.map(b => ({ ...b, lut: lutFor(b, D, W, bram) * OPT }));
    const total = parts.reduce((s, p) => s + p.lut, 0);
    const pct = (total / TOTAL_LUT) * 100;
    const maxBar = Math.max(TOTAL_LUT, total, 50000);
    const busBits = W * D;

    // ---- Datapath animé : un tube de largeur W·D ----
    const tubeW = Math.min(560, 80 + busBits / 8);   // px, proportionnel à W·D
    const tubeH = 36;
    const cx = 300, cy = 70;
    const tube = `
      <g>
        <text x="${cx}" y="22" text-anchor="middle" fill="#cdd6f4" font-size="13" font-weight="600">Datapath — largeur du bus = W·D = ${W}×${D} = ${busBits} bits</text>
        <rect x="${cx - tubeW/2}" y="${cy - tubeH/2}" width="${tubeW}" height="${tubeH}" rx="6"
              fill="rgba(137,180,250,0.12)" stroke="#89b4fa" stroke-width="1.5"/>
        <text x="${cx - tubeW/2 - 6}" y="${cy+4}" text-anchor="end" fill="#8b8da6" font-size="11">x</text>
        <text x="${cx + tubeW/2 + 6}" y="${cy+4}" text-anchor="start" fill="#8b8da6" font-size="11">y</text>
        <text x="${cx}" y="${cy+5}" text-anchor="middle" fill="#f9e2af" font-size="11">xbuf / wbuf  [W·D bits]</text>
        <text x="${cx}" y="${cy + tubeH/2 + 16}" text-anchor="middle" fill="#6c7086" font-size="10">
          dim×2 ⇒ bus ×2 ⇒ ~×2 bascules/ports ⇒ ~×2 LUT (registres + routage)
        </text>
      </g>`;

    // ---- Barres empilées LUT par module ----
    const barX = 40, barW = 520, barH = 30, barY = 170;
    let acc = 0;
    const segs = parts.map(p => {
      const w = (p.lut / maxBar) * barW;
      const seg = `<rect class="cap-seg" data-key="${p.key}" x="${barX + acc}" y="${barY}" width="${w}" height="${barH}" fill="${p.color}" opacity="0.85"><title>${p.label}: ${Math.round(p.lut)} LUT</title></rect>`;
      acc += w;
      return seg;
    }).join('');
    // ligne de capacité ECP5-45
    const capX = barX + (TOTAL_LUT / maxBar) * barW;
    const capLine = `<line x1="${capX}" y1="${barY - 8}" x2="${capX}" y2="${barY + barH + 8}" stroke="#f38ba8" stroke-width="2.5" stroke-dasharray="5 3"/>
      <text x="${capX}" y="${barY - 12}" text-anchor="middle" fill="#f38ba8" font-size="11" font-weight="600">ECP5-45 : 41 664 LUT</text>`;
    // ligne du total courant
    const totX = barX + (total / maxBar) * barW;
    const totLine = total > maxBar * 0.98 ? '' :
      `<line x1="${totX}" y1="${barY + barH}" x2="${totX}" y2="${barY + barH + 6}" stroke="#cdd6f4" stroke-width="1.5"/>`;

    // légendes blocs (sous la barre)
    const legY = barY + barH + 28;
    const legend = parts.map((p, i) => {
      const lx = barX + (i % 3) * 200;
      const ly = legY + Math.floor(i / 3) * 22;
      return `<g class="cap-seg" data-key="${p.key}" style="cursor:help">
        <rect x="${lx}" y="${ly}" width="12" height="12" fill="${p.color}" opacity="0.85"/>
        <text x="${lx + 18}" y="${ly + 11}" fill="#cdd6f4" font-size="11">${p.label} — ${Math.round(p.lut)}</text>
      </g>`;
    }).join('');

    // jauge % circulaire (droite)
    const gauge = this._gauge(pct, 430, 130);

    const svg = `<svg width="100%" height="100%" viewBox="0 0 840 420" preserveAspectRatio="xMidYMid meet" font-family="Helvetica Neue,Arial,sans-serif">
      ${tube}
      ${segs}${totLine}${capLine}
      <text x="${barX}" y="${barY - 8}" fill="#8b8da6" font-size="11">LUT4 par bloc (empilé)</text>
      <text x="${barX + barW}" y="${barY + barH + 18}" text-anchor="end" fill="#cdd6f4" font-size="12" font-weight="600">
        total = ${Math.round(total)} LUT (${pct.toFixed(1)}%)
      </text>
      ${legend}
      ${gauge}
    </svg>`;
    this.canvas.innerHTML = svg;
    this.canvas.querySelectorAll('.cap-seg').forEach(el => {
      el.addEventListener('mouseenter', () => this._hover(el.dataset.key));
    });

    // verdict
    const marge = TOTAL_LUT - total;
    const ok = marge >= 0;
    const profile = DIMS.find(d => d === D) ? `dim${D}` : '';
    const winfo = W !== 8 ? ` · W=${W}` : '';
    const braminfo = bram ? ' · BRAM' : '';
    this.verdictEl.innerHTML = ok
      ? `<span style="color:#a6e3a1">✓ ${profile}${winfo}${braminfo} tiendrait : ${Math.round(marge)} LUT libres (${(marge/TOTAL_LUT*100).toFixed(0)}%)</span>`
      : `<span style="color:#f38ba8">✗ ${profile}${winfo}${braminfo} déborde de ${Math.round(-marge)} LUT (${pct.toFixed(0)}%)</span>`;
  }

  _gauge(pct, x, y) {
    const r = 38, cx = x, cy = y;
    const a = Math.min(pct, 100) / 100 * Math.PI * 2;
    const x2 = cx + r * Math.sin(a), y2 = cy - r * Math.cos(a);
    const large = a > Math.PI ? 1 : 0;
    const col = pct > 100 ? '#f38ba8' : pct > 87 ? '#f9e2af' : '#a6e3a1';
    return `<g>
      <circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="#45475a" stroke-width="8"/>
      <path d="M ${cx} ${cy - r} A ${r} ${r} 0 ${large} 1 ${x2} ${y2}" fill="none" stroke="${col}" stroke-width="8"/>
      <text x="${cx}" y="${cy - 2}" text-anchor="middle" fill="#cdd6f4" font-size="20" font-weight="700">${pct.toFixed(0)}%</text>
      <text x="${cx}" y="${cy + 16}" text-anchor="middle" fill="#8b8da6" font-size="10">LUT used</text>
    </g>`;
  }

  _hover(key) {
    const b = BLOCS.find(x => x.key === key);
    if (!b) return;
    const { D, W, bram } = this._state();
    const lut = lutFor(b, D, W, bram) * OPT;
    this.infoEl.innerHTML = `<code style="color:${b.color}">${b.label}</code>
      <div class="pt-mod-desc"><b>${Math.round(lut)} LUT</b> à dim=${D}, W=${W}${bram?' +BRAM':''}.<br>${b.desc}</div>`;
  }

  _toggleSweep() {
    if (this._sweep) { clearInterval(this._sweep); this._sweep = null; this.animBtn.textContent = '▶ sweep dim'; return; }
    this.animBtn.textContent = '⏸ sweep';
    let i = 0;
    this._sweep = setInterval(() => {
      this.dimSel.value = DIMS[i % DIMS.length];
      this.render();
      i++;
    }, 1100);
  }

  onShow() { this.render(); }
  get ready() { return true; }
}