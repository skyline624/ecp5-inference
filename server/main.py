"""main.py — serveur FastAPI "FPGA Packet Tracer".

Phase A : expose le graphe de modules RTL (GET /api/graph/{top}) + le SVG rendu.
Phase B/C : endpoints simulation/VCD (ajoutés dans sim/vcd).
Le frontend (web/) tourne sous Vite en dev (port 5173) et proxy /api -> 8000.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response

from . import config, env
from .rtl import extract, graphviz_svg

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