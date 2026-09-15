#!/usr/bin/env python3
"""FastAPI + WebSocket server for interactive DeepSDF latent-space exploration.

Loads a trained decoder checkpoint and serves a three.js page where the user
can drag latent-code sliders and see the extracted iso-surface update live.

Usage:
    .venv/bin/python server.py -e <experiment_dir> [--checkpoint latest] [--port 8000]
"""

import argparse
import asyncio
import json
import logging
import os
import struct
import time

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import deep_sdf.workspace
from deep_sdf.differentiable_mesh import extract_differentiable_mesh

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")

app = FastAPI()

STATE = {
    "decoder": None,
    "latent_size": 0,
    "field_type": "sdf",
    "mc_backend": "skimage",
    "latents": None,  # (num_shapes, latent_size) tensor or None
    "shape_names": [],
    "latent_mean": None,
    "latent_std": None,
    "bounds": None,  # extraction domain (xmin,ymin,zmin,xmax,ymax,zmax) or None for [-1,1]^3
}

# GPU is a single stream: serialize mesh extraction.
EXTRACT_LOCK = asyncio.Lock()


def build_decoder_for_specs(specs):
    import networks.deep_sdf_decoder as arch

    decoder = arch.Decoder(specs["CodeLength"], **specs["NetworkSpecs"])
    return decoder


def load_experiment(experiment_dir, checkpoint):
    specs = deep_sdf.workspace.load_experiment_specifications(experiment_dir)

    decoder = build_decoder_for_specs(specs)
    model_file = os.path.join(experiment_dir, "ModelParameters", checkpoint + ".pth")
    if not os.path.isfile(model_file):
        raise RuntimeError('model checkpoint "{}" does not exist'.format(model_file))
    saved = torch.load(model_file, map_location="cpu")
    state = saved["model_state_dict"]
    # tolerate DataParallel "module." prefix
    if any(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    decoder.load_state_dict(state, strict=False)

    if torch.cuda.is_available():
        decoder = decoder.cuda()
        STATE["mc_backend"] = "torch"
    decoder.eval()

    STATE["decoder"] = decoder
    STATE["latent_size"] = specs["CodeLength"]
    STATE["field_type"] = specs.get("FieldType", "sdf")
    if "ExtractBounds" in specs:
        STATE["bounds"] = specs["ExtractBounds"]
        logging.info("extraction bounds from specs: {}".format(STATE["bounds"]))

    latents_file = os.path.join(experiment_dir, "LatentCodes", checkpoint + ".pth")
    if os.path.isfile(latents_file):
        data = torch.load(latents_file, map_location="cpu")
        codes = data["latent_codes"]
        if not isinstance(codes, torch.Tensor):
            codes = codes["weight"]
        STATE["latents"] = codes.float()
        STATE["latent_mean"] = codes.mean(0)
        STATE["latent_std"] = codes.std(0)
        STATE["shape_names"] = [
            "shape_{}".format(i) for i in range(codes.shape[0])
        ]
        logging.info("loaded {} preset latents".format(codes.shape[0]))
    else:
        logging.warning("no latent codes found; preset shapes disabled")

    logging.info(
        "loaded {} (epoch {}), latent_size={}, field_type={}, mc_backend={}".format(
            model_file,
            saved.get("epoch", "?"),
            STATE["latent_size"],
            STATE["field_type"],
            STATE["mc_backend"],
        )
    )


def extract_mesh_sync(z, resolution):
    latent = torch.tensor(z, dtype=torch.float32)
    if torch.cuda.is_available():
        latent = latent.cuda()
    with torch.no_grad():
        verts, faces, normals = extract_differentiable_mesh(
            STATE["decoder"],
            latent,
            resolution=resolution,
            max_batch=2 ** 16,
            field_type=STATE["field_type"],
            mc_backend=STATE["mc_backend"],
            bounds=STATE["bounds"],
        )
    return (
        verts.detach().cpu().numpy().astype(np.float32),
        faces.detach().cpu().numpy().astype(np.int32),
        normals.detach().cpu().numpy().astype(np.float32),
    )


def pack_mesh(verts, faces, normals):
    header = struct.pack("<II", verts.shape[0], faces.shape[0])
    return header + verts.tobytes() + faces.tobytes() + normals.tobytes()


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            z = msg.get("z")
            resolution = int(msg.get("resolution", 63))
            if z is None or len(z) != STATE["latent_size"]:
                await ws.send_text(json.dumps({"error": "bad latent code"}))
                continue
            async with EXTRACT_LOCK:
                t0 = time.perf_counter()
                verts, faces, normals = await asyncio.get_event_loop().run_in_executor(
                    None, extract_mesh_sync, z, resolution
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
            await ws.send_text(
                json.dumps(
                    {
                        "meta": True,
                        "num_verts": int(verts.shape[0]),
                        "num_faces": int(faces.shape[0]),
                        "elapsed_ms": round(elapsed_ms, 1),
                    }
                )
            )
            await ws.send_bytes(pack_mesh(verts, faces, normals))
    except WebSocketDisconnect:
        pass


@app.get("/api/info")
async def info():
    return JSONResponse(
        {
            "latent_size": STATE["latent_size"],
            "field_type": STATE["field_type"],
            "mc_backend": STATE["mc_backend"],
            "shape_names": STATE["shape_names"],
            "latent_mean": STATE["latent_mean"].tolist()
            if STATE["latent_mean"] is not None
            else [0.0] * STATE["latent_size"],
            "latent_std": STATE["latent_std"].tolist()
            if STATE["latent_std"] is not None
            else [0.05] * STATE["latent_size"],
            "bounds": STATE["bounds"],
            "latents": STATE["latents"].tolist()
            if STATE["latents"] is not None
            else [],
        }
    )


@app.get("/")
async def index():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-e", "--experiment", required=True, help="experiment directory")
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--bounds",
        type=float,
        nargs=6,
        metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"),
        default=None,
        help="extraction domain, e.g. the metric data bounding box; overrides specs ExtractBounds",
    )
    args = parser.parse_args()

    load_experiment(args.experiment, args.checkpoint)
    if args.bounds is not None:
        STATE["bounds"] = list(args.bounds)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
