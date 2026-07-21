// main.js — point d'entrée. Routeur de vues minimal (Phase A = GraphView).
import { GraphView } from './graph/GraphView.js';

const app = document.getElementById('app');
const gv = new GraphView(app);
// charge le top par défaut (gen_seq_silicon) via le sélecteur interne
window.pt = gv; // debug