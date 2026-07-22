"""main.py — serveur FastAPI "FPGA Packet Tracer".

Phase A : expose le graphe de modules RTL (GET /api/graph/{top}) + le SVG rendu.
Phase B/C : endpoints simulation/VCD (ajoutés dans sim/vcd).
Le frontend (web/) tourne sous Vite en dev (port 5173) et proxy /api -> 8000.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

from . import config, env
from .rtl import extract, graphviz_svg, pipeline as rtl_pipeline, memory as rtl_memory
from .sim import runner as sim_runner
from .sim import tb_registry
from .vcd import events as vcd_events

# Configure le PATH (oss-cad + dot) une fois au démarrage.
env.setup()

app = FastAPI(title="FPGA Packet Tracer", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # dev : vite 5173 -> 8000
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache du graphe RTL (regénère si les sources changent).
_rtl_graph: Optional[extract.RTLGraph] = None
_graph_cache: dict = {}


def _get_rtl() -> extract.RTLGraph:
    global _rtl_graph
    if _rtl_graph is None:
        _rtl_graph = extract.RTLGraph.build(list(config.RTL_FILES))
    return _rtl_graph


def _get_graph(top: str) -> dict:
    if top not in _graph_cache:
        _graph_cache[top] = extract.build_graph_dict(top, _get_rtl(), max_depth=3)
    return _graph_cache[top]


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "modules": len(_get_rtl().modules)}


@app.get("/api/graph/{top}")
def api_graph(top: str) -> dict:
    if top not in _get_rtl().modules:
        raise HTTPException(404, f"top module '{top}' inconnu")
    return _get_graph(top)


@app.get("/api/graph_svg/{top}")
def api_graph_svg(top: str) -> Response:
    if top not in _get_rtl().modules:
        raise HTTPException(404, f"top module '{top}' inconnu")
    svg = graphviz_svg.graph_svg(_get_graph(top))
    return Response(content=svg, media_type="image/svg+xml")


@app.get("/api/pipeline")
def api_pipeline() -> dict:
    """Dataflow d'un token (étapes + vecteurs). Définition fixe du design."""
    return rtl_pipeline.pipeline_json()


@app.get("/api/pipeline_svg")
def api_pipeline_svg() -> Response:
    """SVG du dataflow (rendu Graphviz). ids `step:*`/`vec:*` pour le frontend."""
    svg = rtl_pipeline.pipeline_svg()
    return Response(content=svg, media_type="image/svg+xml")


@app.get("/api/memory")
def api_memory() -> dict:
    """Carte mémoire 3 tiers (flash/SDRAM/BRAM) + régions + flux. Définition fixe."""
    return rtl_memory.memory_json()


@app.get("/api/memory_svg")
def api_memory_svg() -> Response:
    """SVG de la carte mémoire (rendu Graphviz). ids `mem:*`/`flow:*` pour le frontend."""
    svg = rtl_memory.memory_svg()
    return Response(content=svg, media_type="image/svg+xml")


@app.get("/api/tops")
def api_tops() -> dict:
    """Liste les top modules utilisables (silicium + sim)."""
    g = _get_rtl()
    return {
        "tops": [
            {"name": config.TOP_SILICON, "available": config.TOP_SILICON in g.modules},
            {"name": config.TOP_SIM, "available": config.TOP_SIM in g.modules},
        ]
    }


# --- Phase B : simulation + VCD -----------------------------------------------

class SimRunReq(BaseModel):
    tb: str
    mode: str = ""   # "" = mode par défaut du TB (iverilog pour dump_tb_*, cocotb sinon)


class TimelineReq(BaseModel):
    signals: list[str]
    from_t: int = 0
    to_t: Optional[int] = None
    clk_ref: Optional[str] = None
    max_per_signal: int = 20000


@app.get("/api/tb")
def api_tb() -> dict:
    return {"tbs": tb_registry.list_all()}


@app.post("/api/sim/run")
def api_sim_run(req: SimRunReq) -> dict:
    try:
        job = sim_runner.run_sim(req.tb, req.mode)
    except ValueError as e:
        raise HTTPException(404, str(e))
    return {"job_id": job["job_id"], "status": job["status"], "test": job["test"], "mode": job["mode"]}


@app.get("/api/sim/{job_id}/status")
def api_sim_status(job_id: str) -> dict:
    job = sim_runner.status(job_id)
    if not job:
        raise HTTPException(404, "job inconnu")
    return job


@app.get("/api/sim/{job_id}/signals")
def api_sim_signals(job_id: str) -> dict:
    vcd = sim_runner.get_vcd(job_id)
    if not vcd:
        raise HTTPException(409, "VCD non prêt (job pas fini ou échoué)")
    return vcd_events.list_signals_grouped(vcd)


@app.get("/api/sim/{job_id}/events")
def api_sim_events(
    job_id: str,
    signal: str = Query(...),
    from_t: int = 0,
    to_t: Optional[int] = None,
    max: int = 20000,
) -> dict:
    vcd = sim_runner.get_vcd(job_id)
    if not vcd:
        raise HTTPException(409, "VCD non prêt")
    tl = vcd_events.timeline(vcd, [signal], from_t, to_t, max_per_signal=max)
    return {"signal": tl["signals"][0] if tl["signals"] else None, "cycles": tl["cycles"]}


@app.post("/api/sim/{job_id}/timeline")
def api_sim_timeline(job_id: str, req: TimelineReq) -> dict:
    vcd = sim_runner.get_vcd(job_id)
    if not vcd:
        raise HTTPException(409, "VCD non prêt")
    return vcd_events.timeline(vcd, req.signals, req.from_t, req.to_t, req.clk_ref, req.max_per_signal)


@app.get("/api/sim")
def api_sim_jobs() -> dict:
    return {"jobs": sim_runner.list_jobs()}


# --- Fallback statique : sert web/ (pour test sans vite) ---------------------
@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    idx = config.WEB_DIR / "index.html"
    if idx.exists():
        return HTMLResponse(idx.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>FPGA Packet Tracer</h1><p>Frontend non build (lancez <code>npm run dev</code> dans web/).</p>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.main:app", host="127.0.0.1", port=8000, reload=True)