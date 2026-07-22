// main.js — point d'entrée. Routeur de vues (Phase A GraphView, B TimelineView,
// C AnimateView, + Flux pipeline PipelineView).
//
// Initialisation paresseuse : une vue n'est construite qu'au premier affichage de
// son onglet. Sinon svg-pan-zoom s'initialise sur un SVG dans un conteneur
// `display:none` (hauteur 0) -> matrice non inversible -> "Failed to execute
// 'inverse' on 'SVGMatrix'" à chaque zoom/destroy.
//
// État partagé `state` (EventTarget) : publie le cycle courant pour synchroniser
// une vue animée avec une autre (ex. play dans Timeline -> AnimateView suit).
import { GraphView } from './graph/GraphView.js';
import { TimelineView } from './timeline/TimelineView.js';
import { AnimateView } from './animate/AnimateView.js';
import { PipelineView } from './pipeline/PipelineView.js';
import { MemoryView } from './memory/MemoryView.js';
import { ClusterView } from './cluster/ClusterView.js';

const app = document.getElementById('app');

const bar = document.createElement('div');
bar.className = 'pt-tabs';
bar.innerHTML = `
  <button class="pt-tab active" data-view="graph">Graphe RTL</button>
  <button class="pt-tab" data-view="pipeline">Flux pipeline</button>
  <button class="pt-tab" data-view="memory">Mémoires</button>
  <button class="pt-tab" data-view="cluster">Cluster</button>
  <button class="pt-tab" data-view="timeline">Timeline VCD</button>
  <button class="pt-tab" data-view="animate">Animate</button>`;
app.before(bar);

// État partagé cycle (Phase C). Les vues écoutent 'cycle' pour se synchroniser.
const state = new EventTarget();
state.iCycle = 0;
state.cycles = [];
state.tl = null;
state.jobId = null;
state.setCycle = function (iCycle, cycles) {
  state.iCycle = iCycle;
  if (cycles) state.cycles = cycles;
  state.dispatchEvent(new CustomEvent('cycle', { detail: { iCycle, cycles } }));
};
window.ptState = state;

// Une div par vue, toutes créées au boot (vides, cachées sauf graph). La vue
// elle-même n'est instanciée qu'au premier switchView (lazy) -> son panZoom
// naît dans un conteneur visible (hauteur non nulle).
const ctors = { graph: GraphView, pipeline: PipelineView, memory: MemoryView, cluster: ClusterView, timeline: TimelineView, animate: AnimateView };
const panels = {};
const views = {};
for (const name of Object.keys(ctors)) {
  const p = document.createElement('div');
  p.style.height = '100%';
  p.style.display = name === 'graph' ? 'block' : 'none';
  app.appendChild(p);
  panels[name] = p;
}

function ensureView(name) {
  if (views[name]) return views[name];
  views[name] = new ctors[name](panels[name]);
  // branche le cycle partagé quand TimelineView est créée.
  if (name === 'timeline') {
    views[name].onCycleChange = (i, cycles) => state.setCycle(i, cycles);
  }
  return views[name];
}

function switchView(name) {
  bar.querySelectorAll('.pt-tab').forEach((b) =>
    b.classList.toggle('active', b.dataset.view === name));
  for (const n of Object.keys(panels)) panels[n].style.display = (n === name) ? 'block' : 'none';
  const v = ensureView(name);
  window.pt = v;
  // Les vues à panZoom peuvent avoir des dimensions stale après être restées
  // cachées : re-fit pour recadrer au conteneur maintenant visible.
  if (v.panZoom) {
    try { v.panZoom.resize(); v.panZoom.fit(); v.panZoom.center(); } catch (_) { /* lazy: pas encore prêt */ }
  }
  if (v.onShow) v.onShow();
}

bar.querySelectorAll('.pt-tab').forEach((b) =>
  b.addEventListener('click', () => switchView(b.dataset.view)));

// instancie la vue par défaut (graph, visible au boot).
ensureView('graph');
switchView('graph');