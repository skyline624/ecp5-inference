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

// --- Dataflow d'un token (vue pipeline) ---
export async function getPipeline() {
  const r = await fetch(`${BASE}/pipeline`);
  if (!r.ok) throw new Error(`pipeline: ${r.status}`);
  return r.json();
}

export async function getPipelineSvg() {
  const r = await fetch(`${BASE}/pipeline_svg`);
  if (!r.ok) throw new Error(`pipeline_svg: ${r.status}`);
  return r.text();
}

// --- Carte mémoire 3 tiers (vue Mémoires) ---
export async function getMemory() {
  const r = await fetch(`${BASE}/memory`);
  if (!r.ok) throw new Error(`memory: ${r.status}`);
  return r.json();
}

export async function getMemorySvg() {
  const r = await fetch(`${BASE}/memory_svg`);
  if (!r.ok) throw new Error(`memory_svg: ${r.status}`);
  return r.text();
}

// --- Phase B : simulation / VCD ---

export async function listTbs() {
  const r = await fetch(`${BASE}/tb`);
  return r.json();
}

export async function runSim(tb, mode = '') {
  const r = await fetch(`${BASE}/sim/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ tb, mode }),
  });
  if (!r.ok) throw new Error(`run ${tb}: ${r.status}`);
  return r.json();
}

export async function simStatus(jobId) {
  const r = await fetch(`${BASE}/sim/${jobId}/status`);
  return r.json();
}

export async function simSignals(jobId) {
  const r = await fetch(`${BASE}/sim/${jobId}/signals`);
  if (!r.ok) throw new Error(`signals: ${r.status}`);
  return r.json();
}

export async function simTimeline(jobId, signals, fromT = 0, toT = null, clkRef = null) {
  const r = await fetch(`${BASE}/sim/${jobId}/timeline`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ signals, from_t: fromT, to_t: toT, clk_ref: clkRef }),
  });
  if (!r.ok) throw new Error(`timeline: ${r.status}`);
  return r.json();
}

export async function simJobs() {
  const r = await fetch(`${BASE}/sim`);
  return r.json();
}