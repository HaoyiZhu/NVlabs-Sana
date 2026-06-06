#!/usr/bin/env python
# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
# Licensed under the Apache License, Version 2.0
# SPDX-License-Identifier: Apache-2.0

"""Realtime interactive SANA-WM demo.

A Gradio-hosted app whose star is a live, game-style camera: pick a scene, press
Start, and the camera glides forward immediately (the first chunks replay from a
per-scene cache so there is zero perceived warm-up). Steer in real time —
**WASD** translate the camera, the **arrow keys** rotate it. Frames are pushed to
a ``<canvas>`` at 16 fps over a WebSocket and the NVFP4-optimised pipeline keeps
generation at ~realtime.

The interactive engine lives in ``app/sana_wm_interactive.py``; this file is the
web layer (FastAPI routes + frontend + a small Gradio status panel).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---- Env knobs (must precede any torch / diffusion import) ----------------
os.environ.setdefault("DISABLE_XFORMERS", "1")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(Path.home() / ".cache" / "sana_wm_torchinductor"))
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
os.environ.setdefault("SANA_WM_PREPARED_MODULE_CACHE", "1")
os.environ.setdefault("SANA_WM_PREPARED_MODULE_CACHE_DIR", str(Path.home() / ".cache" / "sana_wm_prepared_modules"))
if "--cuda_visible_devices" in sys.argv:
    _i = sys.argv.index("--cuda_visible_devices")
    if _i + 1 < len(sys.argv):
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[_i + 1]

# Realtime (GB200 / 5090) optimisation defaults. These are GPU-agnostic; the
# Blackwell-only NVFP4 / fp8-KV knobs are auto-gated after torch import below.
os.environ.setdefault("DPM_TQDM", "True")
os.environ.setdefault("FUSED_GDN_PRECISION", "2")
os.environ.setdefault("SANA_WM_TORCH_COMPILE_DYNAMIC", "0")
# Compile the refiner only. Compiling the causal VAE streaming decoder corrupts
# its cross-chunk cache (chunk >=1 decodes to gray); see sana_wm_interactive.
os.environ.setdefault("SANA_WM_TORCH_COMPILE_TARGETS", "refiner")
os.environ.setdefault("SANA_WM_STAGE1_KV_SAVE_STRIDE", "0")
os.environ.setdefault("SANA_WM_STAGE1_LINEARIZE_FFN", "1")
os.environ.setdefault("SANA_WM_STAGE1_NVFP4_MODE", "self_attn+cross+ffn")
os.environ.setdefault("SANA_WM_STAGE1_NVFP4_TEXT_PAD_MULTIPLE", "8")
os.environ.setdefault("SANA_WM_SDPA_D112_DIRECT", "1")
os.environ.setdefault("SANA_WM_REFINER_ATTN_BACKEND", "_native_flash")
os.environ.setdefault("SANA_WM_REFINER_SELF_ATTN_KERNEL", "flash_attn")
os.environ.setdefault("SANA_WM_REFINER_CROSS_ATTN_KV_CACHE", "1")
os.environ.setdefault("SANA_WM_REFINER_PRECONCAT_PREFIX", "1")
os.environ.setdefault("SANA_WM_REFINER_NO_CLONE_CAPTURED_KV", "1")
os.environ.setdefault("SANA_WM_REFINER_CAPTURE_KV_ONLY_LAST", "1")
os.environ.setdefault("SANA_WM_REFINER_FAST_KV_CAPTURE", "last_predict")
os.environ.setdefault("SANA_WM_REFINER_FAST_KV_CLEAN_INTERVAL", "4")
os.environ.setdefault("SANA_WM_STREAMING_PREDECODE_SINK", "1")
# Interactive demo keeps the VAE decoder resident on GPU for low latency.
os.environ.setdefault("SANA_WM_STREAMING_LAZY_VAE_DECODER", "0")
os.environ.setdefault("SANA_WM_STREAMING_PROMPT_CACHE", "1")

import argparse  # noqa: E402
import asyncio  # noqa: E402
import logging  # noqa: E402
import struct  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402

import gradio as gr  # noqa: E402
import torch  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, status  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response  # noqa: E402

from sana.tools import is_blackwell  # noqa: E402

# NVFP4 / fp8-KV are Blackwell-only (sm_100 B200/GB200, sm_120 5090/GB10). Enable
# them automatically on Blackwell; on Hopper/Ada warn loudly and fall back to bf16
# (these env vars are read at model-build time, so setting them post-torch is fine).
if is_blackwell():
    os.environ.setdefault("SANA_WM_STAGE1_NVFP4", "1")
    os.environ.setdefault("SANA_WM_REFINER_NVFP4", "1")
    os.environ.setdefault("SANA_WM_REFINER_KV_CACHE_DTYPE", "fp8_e4m3fn")
    os.environ.setdefault("SANA_WM_TE_NVFP4_CPU_STAGING", "1")
else:
    os.environ.setdefault("SANA_WM_STAGE1_NVFP4", "0")
    os.environ.setdefault("SANA_WM_REFINER_NVFP4", "0")
    os.environ.setdefault("SANA_WM_REFINER_KV_CACHE_DTYPE", "bf16")
    os.environ.setdefault("SANA_WM_TE_NVFP4_CPU_STAGING", "0")
    print(
        "[realtime-demo] WARNING: non-Blackwell GPU detected (NVFP4/fp8 need "
        "sm_100/sm_120, e.g. B200/GB200/RTX-5090). NVFP4 is OFF and the demo "
        "runs the bf16 path — markedly slower, will NOT reach realtime.",
        flush=True,
    )

from diffusion.utils.logger import get_root_logger  # noqa: E402
from app.sana_wm_interactive import (  # noqa: E402
    DEFAULT_STREAMING_ROOT,
    LOGGER,
    MAX_SECONDS_ACTUAL,
    SCENE_BY_ID,
    SCENES,
    SERVER_QUEUE_HIGHWATER,
    FPS,
    LoadedPipeline,
    Session,
    _persistent_cache_dir,
    load_pipeline,
    prepare_runtime,
    run_session,
)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 7860

LOADED: LoadedPipeline | None = None
SESSION_LOCK = asyncio.Lock()
CURRENT_SESSION: Session | None = None


# ============================================================================
# Frontend (single elegant interactive page)
# ============================================================================

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=1500" />
  <title>SANA-WM · Realtime</title>
  <style>
    :root {
      --bg: #0b0c0f; --panel: #15161b; --panel2: #1c1d24; --line: #2a2b34;
      --fg: #eceef2; --dim: #8a8c96; --green: #a4d65e; --green-mid: #76b900;
      --yellow: #e5c050; --red: #ff5a52; --blue: #4a9eff;
    }
    * { box-sizing: border-box; }
    html, body {
      margin: 0; padding: 0; background:
        radial-gradient(1200px 600px at 70% -10%, #16181f 0%, var(--bg) 60%);
      color: var(--fg); font-family: 'Inter', 'SF Pro Text', system-ui, -apple-system, sans-serif;
      font-size: 13px; line-height: 1.5;
    }
    header {
      padding: 16px 26px; border-bottom: 1px solid var(--line);
      display: flex; align-items: center; gap: 16px;
    }
    header .logo { width: 9px; height: 22px; border-radius: 3px;
      background: linear-gradient(180deg, var(--green) 0%, var(--green-mid) 100%); }
    header h1 { margin: 0; font-size: 17px; font-weight: 650; letter-spacing: 0.01em; }
    header .meta { color: var(--dim); font-size: 12px; }
    main { display: grid; grid-template-columns: minmax(900px, 1fr) 332px; gap: 22px; padding: 22px 26px; }
    .video-card { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 16px;
      box-shadow: 0 18px 48px rgba(0,0,0,0.35); }
    .video-stage { position: relative; width: 100%; max-width: 1280px; aspect-ratio: 1280 / 704; margin: 0 auto; }
    canvas { display: block; width: 100%; height: 100%; background: #000; border-radius: 9px; }
    .preview { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover;
      border-radius: 9px; pointer-events: none; opacity: 0; transition: opacity 0.2s ease-out; }
    .preview.visible { opacity: 1; }
    .overlay { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center;
      justify-content: center; gap: 14px; pointer-events: none; border-radius: 9px;
      background: rgba(8,9,12,0.55); backdrop-filter: blur(2px); opacity: 0; transition: opacity 0.2s ease-out; }
    .overlay.visible { opacity: 1; }
    .spinner { width: 40px; height: 40px; border-radius: 50%; border: 3px solid rgba(255,255,255,0.16);
      border-top-color: var(--green); animation: spin 0.9s linear infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }
    .overlay-text { font-size: 13px; font-weight: 600; text-shadow: 0 1px 3px rgba(0,0,0,0.7); }
    .overlay-sub { color: var(--dim); font-size: 11px; margin-top: -6px; }
    .statusbar { display: flex; gap: 14px; align-items: center; padding: 9px 14px; margin-top: 13px;
      background: var(--panel2); border: 1px solid var(--line); border-radius: 9px; font-size: 12px; color: var(--dim); }
    .dot { width: 9px; height: 9px; border-radius: 50%; background: var(--dim); transition: background 0.2s; }
    .dot.green { background: var(--green); } .dot.yellow { background: var(--yellow); } .dot.red { background: var(--red); }
    .hint { color: var(--dim); font-size: 11.5px; line-height: 1.6; margin-top: 12px; }
    .kbd { display: inline-block; padding: 1px 7px; background: var(--panel2); border: 1px solid var(--line);
      border-radius: 5px; color: var(--fg); font-size: 11px; font-weight: 600; }
    .side { display: flex; flex-direction: column; gap: 16px; }
    .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 15px 17px; }
    .panel h2 { margin: 0 0 12px; font-size: 11px; font-weight: 650; letter-spacing: 0.1em;
      text-transform: uppercase; color: var(--dim); }
    .scenes { display: flex; flex-direction: column; gap: 7px; }
    .scene-btn { display: flex; align-items: center; gap: 11px; background: var(--panel2);
      border: 1px solid var(--line); color: var(--fg); padding: 8px 10px; text-align: left;
      border-radius: 9px; cursor: pointer; font: inherit; transition: border-color 0.15s, background 0.15s; }
    .scene-btn img { width: 56px; height: 31px; object-fit: cover; border-radius: 5px; background: #000; }
    .scene-btn:hover { border-color: var(--green-mid); }
    .scene-btn.selected { border-color: var(--green); background: rgba(164,214,94,0.09); }
    .actions { display: flex; gap: 9px; margin-top: 13px; }
    .btn { flex: 1; padding: 10px 12px; border-radius: 9px; cursor: pointer; font: inherit; font-weight: 600;
      border: 1px solid var(--line); background: var(--panel2); color: var(--fg); transition: all 0.15s; }
    .btn.primary { background: var(--green-mid); border-color: var(--green-mid); color: #0b0c0f; }
    .btn.primary:hover { background: var(--green); border-color: var(--green); }
    .btn:disabled { opacity: 0.4; cursor: not-allowed; }
    .pads { display: flex; gap: 18px; justify-content: space-between; }
    .pad { flex: 1; }
    .pad-label { font-size: 10.5px; color: var(--dim); margin-bottom: 8px; letter-spacing: 0.08em;
      text-transform: uppercase; text-align: center; }
    .keygrid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; }
    .key { background: var(--panel2); border: 1px solid var(--line); border-radius: 8px; padding: 11px 0;
      text-align: center; user-select: none; font-weight: 700; cursor: pointer; color: var(--fg);
      transition: all 0.08s; font-size: 14px; }
    .key:active, .key.active { background: var(--green-mid); border-color: var(--green-mid); color: #0b0c0f; transform: translateY(1px); }
    .key.spacer { background: transparent; border-color: transparent; cursor: default; }
  </style>
</head>
<body>
  <header>
    <span class="logo"></span>
    <h1>SANA-WM · Realtime</h1>
    <span class="meta">interactive world model · 16 fps · drive with WASD + arrow keys</span>
  </header>
  <main>
    <section class="video-card">
      <div class="video-stage">
        <canvas id="video" width="1280" height="704"></canvas>
        <img id="preview" class="preview" alt="" />
        <div id="overlay" class="overlay">
          <div id="spinner" class="spinner"></div>
          <div id="overlay-text" class="overlay-text">pick a scene and press Start</div>
          <div id="overlay-sub" class="overlay-sub"></div>
        </div>
      </div>
      <div class="statusbar">
        <span class="dot" id="dot"></span>
        <span id="status">ready</span>
        <span style="flex:1"></span>
        <span id="timer">0.0 / __MAXSEC__ s</span>
        <span id="buf">buf=0</span>
      </div>
      <div class="hint">
        <span class="kbd">W</span><span class="kbd">S</span> forward / back &nbsp;·&nbsp;
        <span class="kbd">A</span><span class="kbd">D</span> turn (yaw) &nbsp;·&nbsp;
        <span class="kbd">↑</span><span class="kbd">↓</span> look (pitch) &nbsp;·&nbsp;
        <span class="kbd">←</span><span class="kbd">→</span> step (strafe). On Start the camera glides forward automatically — take over any time. Release to coast to a stop.
      </div>
    </section>
    <aside class="side">
      <div class="panel">
        <h2>Scene</h2>
        <div class="scenes" id="scenes"></div>
        <div class="actions">
          <button class="btn primary" id="btn-start">Start</button>
          <button class="btn" id="btn-reset" disabled>Reset</button>
        </div>
      </div>
      <div class="panel">
        <h2>Controls</h2>
        <div class="pads">
          <div class="pad">
            <div class="pad-label">W/S move · A/D turn</div>
            <div class="keygrid">
              <div class="key spacer"></div><div class="key" data-key="w">W</div><div class="key spacer"></div>
              <div class="key" data-key="a">A</div><div class="key" data-key="s">S</div><div class="key" data-key="d">D</div>
            </div>
          </div>
          <div class="pad">
            <div class="pad-label">↑↓ look · ←→ strafe</div>
            <div class="keygrid">
              <div class="key spacer"></div><div class="key" data-key="up">↑</div><div class="key spacer"></div>
              <div class="key" data-key="left">←</div><div class="key" data-key="down">↓</div><div class="key" data-key="right">→</div>
            </div>
          </div>
        </div>
      </div>
    </aside>
  </main>
  <script>
  (() => {
    const $ = (id) => document.getElementById(id);
    const canvas = $("video"), ctx = canvas.getContext("2d");
    const dot = $("dot"), statusEl = $("status"), timerEl = $("timer"), bufEl = $("buf");
    const startBtn = $("btn-start"), resetBtn = $("btn-reset");
    const previewEl = $("preview"), overlayEl = $("overlay"), spinnerEl = $("spinner");
    const overlayTextEl = $("overlay-text"), overlaySubEl = $("overlay-sub");

    const MAX_SEC = __MAXSEC__;
    const FPS = __FPS__;
    const PREROLL_FRAMES = 3, HARD_BUFFER_CAP = 160;
    const KEYMAP = { "w":"w","a":"a","s":"s","d":"d",
      "arrowup":"up","arrowdown":"down","arrowleft":"left","arrowright":"right" };
    const ALLOWED = new Set(Object.values(KEYMAP));

    let ctrlWs = null, frameWs = null, scenes = [], selectedScene = null;
    let keysDown = new Set(), lastSent = "", running = false, firstFrameSeen = false;
    let buffer = [], playbackStarted = false, playbackTimer = null, frameCount = 0;

    function setStatus(s) { statusEl.textContent = s; }
    function setDot(c) { dot.classList.remove("green","yellow","red"); if (c) dot.classList.add(c); }
    function showOverlay(t, sub, spin) {
      overlayTextEl.textContent = t; overlaySubEl.textContent = sub || "";
      spinnerEl.style.display = spin ? "" : "none"; overlayEl.classList.add("visible");
    }
    function hideOverlay() { overlayEl.classList.remove("visible"); }
    function showPreview(id) { previewEl.src = `/scenes/${encodeURIComponent(id)}/preview`; previewEl.classList.add("visible"); }
    function hidePreview() { previewEl.classList.remove("visible"); previewEl.src = ""; }

    async function loadScenes() {
      const r = await fetch("/scenes"); scenes = await r.json();
      const el = $("scenes"); el.innerHTML = "";
      scenes.forEach((s, i) => {
        const b = document.createElement("button");
        b.className = "scene-btn" + (i === 0 ? " selected" : "");
        b.dataset.id = s.id;
        b.innerHTML = `<img src="/scenes/${encodeURIComponent(s.id)}/preview" alt=""><span>${s.label}</span>`;
        b.onclick = () => {
          document.querySelectorAll(".scene-btn").forEach(x => x.classList.remove("selected"));
          b.classList.add("selected"); selectedScene = s.id;
        };
        el.appendChild(b);
      });
      selectedScene = scenes[0]?.id ?? null;
    }

    async function loadIntro(id) {
      try {
        const r = await fetch(`/intro/${encodeURIComponent(id)}`);
        if (!r.ok) return [];
        const buf = await r.arrayBuffer(); const dv = new DataView(buf);
        let off = 0; const n = dv.getUint32(off, true); off += 4; const out = [];
        for (let i = 0; i < n; i++) {
          const len = dv.getUint32(off, true); off += 4;
          out.push(new Blob([new Uint8Array(buf, off, len)], { type: "image/jpeg" })); off += len;
        }
        return out;
      } catch (e) { return []; }
    }

    function updateKeyVisuals() {
      document.querySelectorAll(".key[data-key]").forEach(el =>
        el.classList.toggle("active", keysDown.has(el.dataset.key)));
    }
    function sendKeys() {
      const arr = Array.from(keysDown).sort(); const sig = arr.join(",");
      if (sig === lastSent) return; lastSent = sig;
      if (ctrlWs && ctrlWs.readyState === 1) ctrlWs.send(JSON.stringify({ op: "keys", keys: arr }));
    }
    function setKey(k, down) {
      if (!ALLOWED.has(k)) return; const before = keysDown.size;
      if (down) keysDown.add(k); else keysDown.delete(k);
      if (down || before !== keysDown.size) { updateKeyVisuals(); sendKeys(); }
    }
    window.addEventListener("keydown", (e) => {
      if (!running) return; const tok = KEYMAP[e.key.toLowerCase()];
      if (!tok) return; e.preventDefault(); setKey(tok, true);
    });
    window.addEventListener("keyup", (e) => {
      const tok = KEYMAP[e.key.toLowerCase()]; if (!tok) return; e.preventDefault(); setKey(tok, false);
    });
    document.querySelectorAll(".key[data-key]").forEach(el => {
      const k = el.dataset.key;
      el.addEventListener("pointerdown", (e) => { e.preventDefault(); setKey(k, true); el.setPointerCapture(e.pointerId); });
      el.addEventListener("pointerup", (e) => { e.preventDefault(); setKey(k, false); });
      el.addEventListener("pointerleave", () => setKey(k, false));
      el.addEventListener("pointercancel", () => setKey(k, false));
    });
    window.addEventListener("blur", () => { if (keysDown.size) { keysDown.clear(); updateKeyVisuals(); sendKeys(); } });

    function pushFrame(blob) {
      buffer.push(blob); while (buffer.length > HARD_BUFFER_CAP) buffer.shift(); updateBuf();
      if (!firstFrameSeen) { firstFrameSeen = true; hidePreview(); hideOverlay(); }
      if (!playbackStarted && buffer.length >= PREROLL_FRAMES) { playbackStarted = true; startPlayback(); }
    }
    function updateBuf() {
      bufEl.textContent = "buf=" + buffer.length;
      if (buffer.length >= 6) setDot("green"); else if (buffer.length >= 3) setDot("yellow"); else if (running) setDot("red");
    }
    function startPlayback() { if (playbackTimer) clearInterval(playbackTimer); playbackTimer = setInterval(drawNext, 1000 / FPS); hideOverlay(); }
    function drawNext() {
      if (buffer.length === 0) { updateBuf(); return; }
      const blob = buffer.shift(); const url = URL.createObjectURL(blob); const img = new Image();
      img.onload = () => { ctx.drawImage(img, 0, 0, canvas.width, canvas.height); URL.revokeObjectURL(url); };
      img.src = url; frameCount++;
      timerEl.textContent = (frameCount / FPS).toFixed(1) + " / " + MAX_SEC + " s"; updateBuf();
    }

    async function start() {
      if (!selectedScene) { setStatus("pick a scene"); return; }
      startBtn.disabled = true; resetBtn.disabled = false;
      buffer = []; frameCount = 0; playbackStarted = false; firstFrameSeen = false; running = true;
      keysDown.clear(); lastSent = ""; updateKeyVisuals();
      if (playbackTimer) { clearInterval(playbackTimer); playbackTimer = null; }
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      setStatus("starting · " + selectedScene); setDot(null);
      showPreview(selectedScene); showOverlay("starting…", "loading cached intro", true);

      // Instant start: prefill the buffer with the cached intro, begin playback now.
      const intro = await loadIntro(selectedScene);
      intro.forEach(b => pushFrame(b));

      ctrlWs = new WebSocket((location.protocol === "https:" ? "wss" : "ws") + "://" + location.host + "/ctrl");
      ctrlWs.onopen = () => ctrlWs.send(JSON.stringify({ op: "start", scene: selectedScene }));
      ctrlWs.onmessage = (ev) => {
        let m; try { m = JSON.parse(ev.data); } catch { return; }
        if (m.event === "started") { setStatus("streaming · " + m.scene); openFrames(); }
        else if (m.event === "finished") { setStatus("clip complete"); stopSession(false); }
        else if (m.event === "error") { setStatus("error: " + m.msg); stopSession(false); }
      };
      ctrlWs.onclose = () => { if (running) { setStatus("disconnected"); stopSession(false); } };
      ctrlWs.onerror = () => setStatus("ctrl error");
    }
    function openFrames() {
      frameWs = new WebSocket((location.protocol === "https:" ? "wss" : "ws") + "://" + location.host + "/frames");
      frameWs.binaryType = "blob";
      frameWs.onmessage = (ev) => {
        if (typeof ev.data === "string") { let m; try { m = JSON.parse(ev.data); } catch { return; }
          if (m.event === "finished") { setStatus("clip complete"); stopSession(false); } }
        else pushFrame(ev.data);
      };
      frameWs.onclose = () => { if (running) setStatus("frames disconnected"); };
    }
    function stopSession(sendStop = true) {
      running = false; keysDown.clear(); updateKeyVisuals();
      if (sendStop && ctrlWs && ctrlWs.readyState === 1) { try { ctrlWs.send(JSON.stringify({ op: "stop" })); } catch {} }
      try { ctrlWs && ctrlWs.close(); } catch {} try { frameWs && frameWs.close(); } catch {}
      ctrlWs = null; frameWs = null; startBtn.disabled = false; resetBtn.disabled = true; setDot(null); hidePreview();
    }
    resetBtn.addEventListener("click", () => stopSession(true));
    startBtn.addEventListener("click", start);
    (async () => { try { await loadScenes(); setStatus("ready"); } catch (e) { setStatus("scene load failed"); } })();
  })();
  </script>
</body>
</html>
""".replace("__MAXSEC__", f"{MAX_SECONDS_ACTUAL:.1f}").replace("__FPS__", str(FPS))


# ============================================================================
# FastAPI app + routes
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    global LOADED
    streaming_root = Path(app.state.streaming_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = _persistent_cache_dir(streaming_root, getattr(app.state, "cache_dir", None))
    LOGGER.info(f"[startup] loading realtime pipeline from {streaming_root}…")
    t0 = time.time()
    LOADED = load_pipeline(streaming_root, device, cache_dir=cache_dir, no_compile=getattr(app.state, "no_compile", False))
    LOGGER.info(f"[startup] pipeline loaded in {time.time() - t0:.1f}s")
    prepare_runtime(LOADED, do_warmup=getattr(app.state, "do_warmup", True))
    LOGGER.info("[startup] ready — open the page in a browser")
    yield


app = FastAPI(lifespan=lifespan, title="SANA-WM Realtime")


@app.get("/")
async def index():
    return HTMLResponse(INDEX_HTML)


@app.get("/scenes")
async def scenes():
    return JSONResponse([{"id": s.id, "label": s.label} for s in SCENES])


@app.get("/scenes/{scene_id}/preview")
async def scene_preview(scene_id: str):
    scene = SCENE_BY_ID.get(scene_id)
    if scene is None:
        return JSONResponse({"detail": f"unknown scene {scene_id!r}"}, status_code=404)
    return FileResponse(str(scene.image_path), media_type="image/png")


@app.get("/intro/{scene_id}")
async def intro(scene_id: str):
    """Cached intro frames for instant playback: a little-endian binary blob of
    [uint32 count]([uint32 len][jpeg bytes])*."""
    jpegs = LOADED.intro_jpegs.get(scene_id, []) if LOADED is not None else []
    out = bytearray(struct.pack("<I", len(jpegs)))
    for j in jpegs:
        out += struct.pack("<I", len(j))
        out += j
    return Response(content=bytes(out), media_type="application/octet-stream")


@app.websocket("/ctrl")
async def ws_ctrl(ws: WebSocket):
    """Control channel. client: start/keys/stop; server: started/finished/error."""
    global CURRENT_SESSION
    await ws.accept()
    if SESSION_LOCK.locked():
        await ws.send_json({"event": "error", "msg": "another session is running"})
        await ws.close(code=status.WS_1013_TRY_AGAIN_LATER)
        return

    async with SESSION_LOCK:
        session: Session | None = None
        producer: threading.Thread | None = None
        loop = asyncio.get_running_loop()
        try:
            while True:
                msg = await ws.receive_json()
                op = msg.get("op")
                if op == "start":
                    scene = SCENE_BY_ID.get(str(msg.get("scene", "demo_0")))
                    if scene is None:
                        await ws.send_json({"event": "error", "msg": "unknown scene"})
                        continue
                    if LOADED is None:
                        await ws.send_json({"event": "error", "msg": "pipeline not loaded"})
                        continue
                    if session is not None and not session.finished_event.is_set():
                        await ws.send_json({"event": "error", "msg": "session already started"})
                        continue
                    session = Session(scene=scene)
                    session.loop = loop
                    session.frame_q = asyncio.Queue(maxsize=SERVER_QUEUE_HIGHWATER)
                    CURRENT_SESSION = session
                    producer = threading.Thread(
                        target=run_session, args=(LOADED, session),
                        name=f"sana-wm-producer-{scene.id}", daemon=True,
                    )
                    producer.start()
                    await ws.send_json({"event": "started", "scene": scene.id})
                elif op == "keys":
                    if session is None:
                        continue
                    keys = msg.get("keys", [])
                    if isinstance(keys, list):
                        session.set_keys({str(k).lower() for k in keys})
                elif op == "stop":
                    if session is not None:
                        session.stop_event.set()
                else:
                    await ws.send_json({"event": "error", "msg": f"unknown op {op!r}"})
        except WebSocketDisconnect:
            pass
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("[ctrl] error")
            try:
                await ws.send_json({"event": "error", "msg": repr(exc)})
            except Exception:
                pass
        finally:
            if session is not None:
                session.stop_event.set()
            # Wait for the producer to wind down (it stops at the next chunk
            # boundary, ~1 chunk) before releasing the lock, so the next session
            # never races this one on the GPU.
            if producer is not None and producer.is_alive():
                await loop.run_in_executor(None, producer.join, 30.0)
            CURRENT_SESSION = None


@app.websocket("/frames")
async def ws_frames(ws: WebSocket):
    """Server -> client JPEG frame stream, paced at 16 fps."""
    await ws.accept()
    period = 1.0 / FPS
    deadline = time.time() + 5.0
    while CURRENT_SESSION is None and time.time() < deadline:
        await asyncio.sleep(0.05)
    session = CURRENT_SESSION
    if session is None or session.frame_q is None:
        await ws.close(code=status.WS_1011_INTERNAL_ERROR)
        return
    next_t = time.time()
    try:
        while True:
            if session.finished_event.is_set() and session.frame_q.empty():
                await ws.send_json({"event": "finished"})
                break
            try:
                jpeg = await asyncio.wait_for(session.frame_q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            await ws.send_bytes(jpeg)
            next_t += period
            sleep_for = next_t - time.time()
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            else:
                next_t = time.time()
    except WebSocketDisconnect:
        pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass


# ---- Small Gradio status panel (mounted at /panel) ------------------------


def _build_gradio_panel() -> gr.Blocks:
    scene_md = "\n".join(f"- **{s.label}** (`{s.id}`)" for s in SCENES)
    with gr.Blocks(title="SANA-WM Realtime") as demo:
        gr.Markdown(
            "## SANA-WM · Realtime\n"
            "Interactive world-model streaming demo.\n\n"
            "**▶ Open the live experience at [`/`](/)** — pick a scene, press Start, "
            "and drive with **WASD** (move) + **arrow keys** (look).\n\n"
            f"### Scenes\n{scene_md}\n\n"
            f"Each clip runs up to **{MAX_SECONDS_ACTUAL:.1f} s** at **{FPS} fps**. "
            "The first chunks are cached per scene for an instant, stutter-free start."
        )
    return demo


# ============================================================================
# CLI entrypoint
# ============================================================================


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Realtime interactive SANA-WM demo (Gradio + WebSocket canvas).")
    p.add_argument("--streaming_root", type=Path, default=DEFAULT_STREAMING_ROOT)
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--server_port", "--port", dest="port", type=int, default=DEFAULT_PORT)
    p.add_argument("--share", action="store_true")
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--no_warmup", action="store_true")
    p.add_argument("--no_compile", action="store_true")
    p.add_argument("--cuda_visible_devices", default=None)
    p.add_argument("--log_level", default="info")
    return p


def _start_share_tunnel(local_host: str, local_port: int, *, retries: int = 3) -> str:
    """Publish a *.gradio.live URL via Gradio's FRP share tunnel."""
    import secrets
    from gradio import networking

    forward_host = "127.0.0.1" if local_host in ("0.0.0.0", "") else local_host
    server_addr = os.environ.get("GRADIO_SHARE_SERVER_ADDRESS", "gradio-live.com:7000")
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return networking.setup_tunnel(forward_host, local_port, secrets.token_urlsafe(32), server_addr, None)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            LOGGER.warning(f"[share] attempt {attempt}/{retries} failed: {exc!r}")
            time.sleep(2.0)
    assert last_exc is not None
    raise last_exc


def main() -> None:
    global app
    args = _build_parser().parse_args()
    get_root_logger()
    logging.basicConfig(level=args.log_level.upper())
    app.state.streaming_root = args.streaming_root
    app.state.cache_dir = args.cache_dir
    app.state.do_warmup = not args.no_warmup
    app.state.no_compile = args.no_compile

    # Mount the Gradio status panel at /panel (keeps this a Gradio app + gives
    # the share tunnel) while the interactive canvas lives at /.
    app = gr.mount_gradio_app(app, _build_gradio_panel(), path="/panel")

    if args.share:
        try:
            url = _start_share_tunnel(args.host, args.port)
            print(f"\n[SANA-WM] Public URL: {url}\n", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[SANA-WM] WARNING: share tunnel failed ({exc!r}); use http://{args.host}:{args.port}/", flush=True)

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
