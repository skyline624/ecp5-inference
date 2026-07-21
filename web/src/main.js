// main.js — point d'entrée. Routeur de vues minimal (Phase A = GraphView,
// Phase B = TimelineView). Un onglet par vue ; l'instance courante est exposée
// sur window.pt pour debug.
import { GraphView } from './graph/GraphView.js';
import { TimelineView } from './timeline/TimelineView.js';

const app = document.getElementById('app');

const bar = document.createElement('div');
bar.className = 'pt-tabs';
bar.innerHTML = `
  <button class="pt-tab active" data-view="graph">Graphe RTL</button>
  <button class="pt-tab" data-view="timeline">Timeline VCD</button>`;
app.before(bar);

const host = document.createElement('div');
host.style.height = '100%';
app.appendChild(host);

let cur = null;
const views = {
  graph: () => new GraphView(host),
  timeline: () => new TimelineView(host),
};

function switchView(name) {
  bar.querySelectorAll('.pt-tab').forEach((b) => b.classList.toggle('active', b.dataset.view === name));
  host.innerHTML = '';
  cur = views[name]();
  window.pt = cur;
}

bar.querySelectorAll('.pt-tab').forEach((b) => b.addEventListener('click', () => switchView(b.dataset.view)));
switchView('graph');