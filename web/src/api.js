// api.js — client HTTP vers le backend FastAPI (proxy /api -> :8000 via vite).

const BASE = '/api';

export async function getTops() {
  const r = await fetch(`${BASE}/tops`);
  return r.json();
}

export async function getGraph(top) {
  const r = await fetch(`${BASE}/graph/${top}`);
  if (!r.ok) throw new Error(`graph ${top}: ${r.status}`);
  return r.json();
}

export async function getGraphSvg(top) {
  const r = await fetch(`${BASE}/graph_svg/${top}`);
  if (!r.ok) throw new Error(`graph_svg ${top}: ${r.status}`);
  return r.text();
}