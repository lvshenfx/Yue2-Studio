"""YuE2 web backend: job queue over a resident model worker, with SQLite indexing.

Artifacts stay on disk under runs/jobs/<id>/out/<id>/ (the official protocol);
SQLite only indexes job metadata. Progress is streamed over SSE, with a 15s
poll fallback available through GET /api/jobs.
"""
from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import unicodedata
import uuid
import urllib.request
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import align, db, jobs_text, logs, score_edit, score_tools, subtitles

log = logs.setup()

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "YuE"
def _resolve_python() -> Path:
    """开发环境用 venv 的 Scripts\\python.exe；整合包用放在 .venv\\python.exe 的便携解释器。"""
    venv = KIT / ".venv"
    for candidate in (venv / "Scripts" / "python.exe", venv / "python.exe"):
        if candidate.exists():
            return candidate
    return venv / "Scripts" / "python.exe"


PYTHON_EXE = _resolve_python()
WORKER_SCRIPT = ROOT / "app" / "model_worker.py"
MODEL_DIR = KIT / "models" / "YuE2-3B"
VAE_DIR = KIT / "models" / "YuE2-Vae"
VAE_LEGACY_DIR = KIT / "models" / "YuE2-Vae-legacy"
JOBS_DIR = ROOT / "runs" / "jobs"
TRASH_DIR = ROOT / "runs" / "_trash"
TRASH_TTL_DAYS = 30
MIN_FREE_GIB = 5.0

STAGE_ZH = {
    "Resolving model files": "解析模型路径",
    "Verifying model files": "校验模型权重",
    "Loading model": "加载模型",
    "Planning score": "规划乐谱",
    "Using provided score": "采用导入乐谱",
    "Generating song": "生成语义序列",
    "Synthesizing audio": "音频合成 (NAR)",
    "Decoding audio": "解码导出音频",
    "Loading audio decoder": "加载音频解码器",
}
STAGE_PHASE = {
    "Planning score": "plan", "Using provided score": "plan",
    "Generating song": "semantic",
    "Synthesizing audio": "nar", "Decoding audio": "decode",
    "Loading audio decoder": "decode",
    "Loading model": "load", "Resolving model files": "load", "Verifying model files": "load",
}
PROGRESS_RE = re.compile(r"^\[YuE2\] (Running|Starting|Completed) ([^:]+): (.*)$")
JOB_TYPES = ("song", "plan", "rerender", "staged", "redecode")
ARTIFACT_NAMES = {
    "audio.flac": "audio/flac", "score.abc": "text/plain; charset=utf-8",
    "plan.json": "application/json", "plan_manifest.json": "application/json",
    "config.json": "application/json", "request.json": "application/json",
    "result.json": "application/json", "failure.json": "application/json",
    "latent.npy": "application/octet-stream", "semantic.npy": "application/octet-stream",
    "abc_tokens.npy": "application/octet-stream", "prefix.npy": "application/octet-stream",
}

presets = [
    {"id": "zh-citypop", "name": "华语城市流行", "language": "zh",
     "style": "Mandarin, city pop, warm female vocal, electric piano, acoustic guitar, 92 BPM",
     "lyrics": "[Verse]\n晚风轻轻吹过窗前\n你留下的笑还在昨天\n[Chorus]\n让这首歌陪你走远\n把所有想念唱成明天"},
    {"id": "zh-folk", "name": "中文民谣叙事", "language": "zh",
     "style": "Mandarin, folk ballad, acoustic guitar, soft male vocal, 76 BPM",
     "lyrics": "[Verse]\n小城的灯一盏盏亮起\n旧车站还留着你的气息\n[Chorus]\n如果时间可以倒回去\n我还是会走进那场雨"},
    {"id": "zh-rock", "name": "华语摇滚", "language": "zh",
     "style": "Mandarin, mandarin rock, electric guitar, driving drums, male vocal, 128 BPM",
     "lyrics": "[Verse]\n把所有的沉默都点燃\n让心跳盖过这座城市\n[Chorus]\n就算明天不会到来\n今夜我们也要大声唱"},
    {"id": "zh-ballad", "name": "中文抒情慢歌", "language": "zh",
     "style": "Mandarin, soft ballad, piano, strings, breathy female vocal, 68 BPM",
     "lyrics": "[Verse]\n窗外的雨下得很轻\n像你说话时的声音\n[Chorus]\n我学会了一个人生活\n却还没学会不想你"},
    {"id": "en-synthwave", "name": "Synthwave", "language": "en",
     "style": "English, synthwave, breathy female vocal, analog synth, gated drums, 104 BPM",
     "lyrics": "[Verse]\nNeon running down the boulevard\nMidnight calls and we are still awake\n[Chorus]\nHold the night a little longer\nWe were never meant to fade"},
    {"id": "en-jazz", "name": "Jazz-funk", "language": "en",
     "style": "English, jazz-funk, warm lead vocal, Rhodes, slap bass, 96 BPM",
     "lyrics": "[Verse]\nCoffee cooling on the windowsill\nCity humming like a quiet chord\n[Chorus]\nLet the rhythm take the evening\nWe can dance across the floor"},
    {"id": "en-acoustic", "name": "Acoustic pop", "language": "en",
     "style": "English, acoustic pop, soft female vocal, acoustic guitar, light percussion, 88 BPM",
     "lyrics": "[Verse]\nPaper planes above the garden\nSummer leaning on the door\n[Chorus]\nSay it slow, say it softly\nI will meet you where we were"},
    {"id": "yue-canto", "name": "粵語流行", "language": "zh",
     "style": "Cantonese, cantopop, warm female vocal, piano, strings, 84 BPM",
     "lyrics": "[Verse]\n舊街燈影照著你我的身影\n把未完的話都留給晚風\n[Chorus]\n如果這一晚可以重來\n我會握緊你的手不放開"},
]

@asynccontextmanager
async def lifespan(_: FastAPI):
    log.info("服务启动，日志写入 %s", logs.LOG_PATH)
    try:
        restore_jobs()
        seed_demo()
        prune_trash()
    except Exception as exc:  # noqa: BLE001 - the index is rebuildable; a bad DB must not block startup
        log.exception("启动时恢复任务索引失败，先以空索引运行")
        _notice("restore-failed", [], 1)
    threading.Thread(target=worker, daemon=True).start()
    yield
    with MODEL_LOCK:
        if MODEL["state"] == "loaded":
            try:
                _unload_locked()
            except Exception:  # noqa: BLE001 - shutdown must not mask the real error
                log.exception("退出时卸载模型失败")
        terminate_worker()
    log.info("服务已停止")


app = FastAPI(title="YuE2 Web", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")


_PUT_MISSING = object()


@app.middleware("http")
async def static_revalidate(request: Request, call_next):
    """静态文件回源校验；动态 API 一律禁止缓存，避免任务/日志/谱面响应被浏览器复用。"""
    response = await call_next(request)
    if request.url.path in ("/", "/legacy") or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    elif request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.middleware("http")
async def security_headers(request: Request, call_next):
    """统一安全响应头；局部响应已有的值不被覆盖（如字幕下载自带 nosniff）。"""
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
    response.headers.setdefault("Cross-Origin-Resource-Policy", "same-origin")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    return response


@app.exception_handler(405)
async def method_not_allowed(request: Request, exc):
    """把 PUT 等未注册方法的 405 映射到已注册的兼容写法（BUG-005）：改谱写回既有
    PUT /api/jobs/{id}/score 与 POST 同义，旧的单方法注册在反代/旧客户端下曾回 405。"""
    if request.url.path.endswith("/score") and request.method in ("PUT", "POST"):
        return JSONResponse({"detail": "请用 PUT /api/jobs/{id}/score 写回（POST 同义）"}, status_code=405)
    return JSONResponse({"detail": "Method Not Allowed"}, status_code=405)

MAX_BODY_BYTES = 2 * 1024 * 1024


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Flatten FastAPI's error list: the UI shows one readable line, not [object Object]."""
    errors = exc.errors()
    detail = "；".join(_validation_message(error) for error in errors[:3]) or "无法解析请求"
    log.warning("请求参数有误 %s %s：%s", request.method, request.url.path, detail)
    return JSONResponse(status_code=422,
                        content={"detail": f"请求参数有误：{detail}",
                                 "errors": jsonable_encoder(errors)})


@app.exception_handler(Exception)
async def unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
    trace = uuid.uuid4().hex[:8]
    log.error("未处理异常 trace=%s %s %s", trace, request.method, request.url.path, exc_info=exc)
    return JSONResponse(status_code=500,
                        content={"detail": f"服务内部错误（编号 {trace}），详情见 runs/app.log",
                                 "trace": trace})


VALIDATION_ZH = {
    "Field required": "必填",
    "Input should be a valid string": "必须是文本",
    "Input should be a valid integer": "必须是整数",
    "Input should be a valid number": "必须是数字",
    "Input should be a valid boolean": "必须是布尔值",
    "Input should be a valid dictionary or object to extract fields from": "必须是 JSON 对象",
    "Input should be a valid list": "必须是数组",
}


def _validation_message(error: dict) -> str:
    location = ".".join(str(part) for part in error.get("loc", ()) if part != "body") or "请求体"
    return f"{location} {VALIDATION_ZH.get(error.get('msg', ''), error.get('msg', ''))}".strip()


@app.middleware("http")
async def limit_body(request, call_next):
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > MAX_BODY_BYTES:
        # 先把请求体读完再回 413：直接返回会让连接被重置，浏览器只能看到"网络错误"
        async for _ in request.stream():
            pass
        return JSONResponse({"detail": "请求体过大"}, status_code=413)
    return await call_next(request)


# ---------------------------------------------------------------- GPU / model
def probe_gpu() -> dict:
    command = ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,compute_cap",
               "--format=csv,noheader,nounits"]
    unknown = {"available": False, "suitable": False, "name": "", "compute_capability": None,
               "reason": "未检测到 NVIDIA GPU"}
    try:
        run = subprocess.run(command, capture_output=True, text=True, timeout=10,
                             encoding="utf-8", errors="replace",
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except FileNotFoundError:
        return unknown  # 没装 CUDA 驱动的机器上这是常态，每 5 秒轮询一次，不该刷日志
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("GPU 查询失败：%s: %s", type(exc).__name__, exc)
        return {**unknown, "reason": f"GPU 查询失败（{type(exc).__name__}）"}
    line = next((x for x in (run.stdout or "").splitlines() if x.strip()), "")
    if run.returncode != 0 or not line:
        log.debug("nvidia-smi 返回 %s：%s", run.returncode, (run.stderr or line)[:200])
        return {**unknown, "reason": "未检测到可用的 NVIDIA GPU"}
    try:
        name, total, free, capability = (part.strip() for part in line.split(",")[:4])
        total_gib, free_gib, capability = float(total) / 1024, float(free) / 1024, float(capability)
    except ValueError:
        log.warning("无法解析 nvidia-smi 输出：%s", line[:200])
        return {"available": True, "suitable": False, "name": line, "compute_capability": None,
                "reason": "无法读取 GPU 信息"}
    if capability < 7.5:
        suitable, reason = False, f"{name} 算力 {capability:.1f} 过低，使用 CPU"
    elif free_gib < MIN_FREE_GIB:
        suitable, reason = False, f"{name} 空闲显存 {free_gib:.1f} GiB，低于 {MIN_FREE_GIB:.0f} GiB，使用 CPU"
    else:
        suitable, reason = True, ""
    return {"available": True, "suitable": suitable, "name": name,
            "total_gib": round(total_gib, 1), "free_gib": round(free_gib, 1),
            "compute_capability": capability, "reason": reason}


GPU = probe_gpu()
DEFAULT_DEVICE = "cuda" if GPU["suitable"] else "cpu"

_GPU_LIVE = {"at": 0.0, "data": None}
_GPU_LIVE_LOCK = threading.Lock()


def gpu_live() -> dict:
    now = time.time()
    with _GPU_LIVE_LOCK:
        if _GPU_LIVE["data"] is None or now - _GPU_LIVE["at"] > 5:
            _GPU_LIVE["data"] = probe_gpu()
            _GPU_LIVE["at"] = now
        return _GPU_LIVE["data"]

MODEL = {"state": "unloaded", "device": None, "device_name": "", "reason": "",
         "seconds": None, "load": {}}
MODEL_BUSY = False
CUDA_BLOCKED = ""
MODEL_LOCK = threading.RLock()
WORKER: dict = {"proc": None, "log": None}
EVENTS: queue.Queue = queue.Queue()
CURRENT_JOB = {"id": None}
SUBSCRIBERS: set = set()
SUB_LOCK = threading.Lock()
DOCTOR_CACHE = {"at": 0.0, "data": None}
DROPPED = {"count": 0}
EVENT_SEQ = {"value": 0}
EVENT_SEQ_LOCK = threading.Lock()
# 断开重连时按 Last-Event-ID 重放漏掉的事件；容量上限是内存与补发成本的折中，
# 超出后由 ready 帧 + 前端全量刷新兜底任务状态。
EVENT_HISTORY_MAXLEN = 1000
EVENT_HISTORY: "deque[dict]" = deque(maxlen=EVENT_HISTORY_MAXLEN)

SERVER_PORT = int(os.environ.get("YUE2_PORT", "9090"))
UPDATE_DIR = ROOT / "runs" / "_update"
UPDATE_LOCK = threading.Lock()
UPDATE_STATE = {"phase": "idle", "detail": "", "at": 0.0}
UPDATE_TIMEOUT = 180.0
VERSION_CACHE = {"at": 0.0, "data": None}
UPDATE_PENDING = {}          # 最近一次 check 的下载信息，download 必须与之一致
DOWNLOAD_LOCK = threading.Lock()


# ------------------------------------------------------------------- job models
class JobRequest(BaseModel):
    type: str = "song"
    style: Any = ""
    lyrics: Any = ""
    seed: int | None = None
    cot: str = "full"
    device: str = "auto"
    abc: str | None = None
    cfg_scale: float | None = None
    abc_sampling: dict | None = None
    semantic_sampling: dict | None = None
    ode_steps: int | None = None
    vae: str | None = None
    source: str | None = None
    from_stage: str | None = None
    title: str | None = None
    work_group: str | None = None
    work_name: str | None = None
    version_note: str | None = None


class Job:
    def __init__(self, job_id: str, job_type: str = "song", request: dict | None = None,
                 *, source: str | None = None, from_stage: str | None = None,
                 vae: str | None = None, ode_steps: int | None = None,
                 title: str = "", parent_id: str | None = None, demo: bool = False):
        self.id = job_id
        self.type = job_type
        self.request = request or {}
        self.source = source
        self.from_stage = from_stage
        self.vae = vae
        self.ode_steps = ode_steps
        self.title = title or ""
        self.parent_id = parent_id or source
        self.variant_group = None
        self.work_name = ""
        self.work_root = None
        self.version_no = None
        self.version_tag = ""
        self.version_label = ""
        self.version_note = ""
        self.kept = None
        self.hidden = False
        self.favorite = False
        self.demo = demo
        self.status = "queued"
        self.stage = ""
        self.progress = ""
        self.error = ""
        self.result: dict = {}
        self.artifacts: dict = {}
        self.effective: dict = {}
        self.created_at = time.time()
        self.updated_at = self.created_at

    def to_db(self) -> dict:
        r = self.result or {}
        truncated = r.get("truncated")
        if isinstance(truncated, dict):
            truncated = any(bool(v) for v in truncated.values())
        return {
            "id": self.id, "type": self.type, "status": self.status,
            "stage": self.stage, "progress": self.progress,
            "device": self.request.get("device", ""),
            "style": self.request.get("style", ""),
            "title": self.title, "seed": self.request.get("seed"),
            "cot": self.request.get("cot", ""),
            "created_at": self.created_at, "updated_at": time.time(),
            "parent_id": self.parent_id, "variant_group": self.variant_group,
            "work_name": self.work_name, "work_root": self.work_root,
            "version_no": self.version_no, "version_tag": self.version_tag,
            "version_label": self.version_label, "version_note": self.version_note,
            "kept": self.kept, "hidden": self.hidden,
            "from_stage": self.from_stage or "",
            "favorite": self.favorite, "error": self.error,
            "duration": r.get("audio_seconds"),
            "truncated": (bool(truncated) if truncated is not None else None),
            "ode_steps": self.ode_steps, "vae": self.vae, "demo": self.demo,
            "requested": self.request, "effective": self.effective,
            "artifacts": self.artifacts, "result": self.result,
        }

    def to_dict(self) -> dict:
        return _api_row(self.to_db())


JOBS: dict[str, Job] = {}
# 内存只当缓存，不当「这个任务存不存在」的判定依据：启动只装最近这么多条，
# 越窗的老任务由 find_job() 按需回查索引（曲库列表本来就每页走 DB）。
MEMORY_JOBS = 60
QUEUE: queue.Queue[str] = queue.Queue()
LOCK = threading.Lock()
NOTICES: list[dict] = []
# 刷新恢复用的运行快照：只活在内存里，任务结束后清空。
LIVE_LOCK = threading.Lock()
LIVE_STAGE = {"id": None, "started_at": 0.0, "load": 0.0, "labels": {}, "label": "",
              "last": 0.0, "event": None, "at": 0.0}


JOB_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_]{1,63}$")


def job_path(job_id: str) -> Path:
    """任务目录的唯一入口：字符集 + 解析后必须仍在 runs/jobs 下一层。"""
    if not JOB_ID_RE.match(job_id or ""):
        raise HTTPException(404, "job not found")
    base = JOBS_DIR.resolve()
    path = (base / job_id).resolve()
    if path.parent != base:
        raise HTTPException(404, "job not found")
    return path


def job_dir(job_id: str) -> Path:
    return job_path(job_id)


def job_artifacts(job_id: str) -> Path:
    return job_path(job_id) / "out" / job_id


PLAN_ARTIFACTS = ("plan.json", "plan_manifest.json", "abc_tokens.npy", "prefix.npy")
AUDIO_ARTIFACT = "audio.flac"
SCORE_ARTIFACT = "score.abc"
OUTPUT_LABELS = {AUDIO_ARTIFACT: "音频 audio.flac", SCORE_ARTIFACT: "乐谱 score.abc"}


def _artifact_ready(path: Path) -> bool:
    """A zero-byte file means the writer died mid-save; never treat it as a product."""
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def required_output(job_type: str) -> tuple[str, ...]:
    """只出谱要 score.abc；其余任务必须有音频。cot=off 的整曲没有乐谱，不能强求。"""
    if job_type == "plan":
        return (SCORE_ARTIFACT,)
    return (AUDIO_ARTIFACT,)


def _recover_score_from_plan(job_id: str) -> bool:
    """让完整 song 也落 score.abc（BUG-015）：官方 SongResult 只存 plan.json(abc 字段)，
    这里把 plan.json.abc 回写成 score.abc（已有 score.abc 则不覆盖），曲库/前端取谱不再 404。"""
    directory = job_artifacts(job_id)
    score_path = directory / SCORE_ARTIFACT
    if _artifact_ready(score_path):
        return True
    plan = _read_json(directory / "plan.json") or {}
    abc = plan.get("abc") if isinstance(plan, dict) else None
    if not isinstance(abc, str) or not abc.strip():
        return False
    try:
        score_path.write_text(abc, encoding="utf-8", newline="\n")
        return _artifact_ready(score_path)
    except OSError as exc:
        log.warning("从 plan.json 回写 score.abc 失败 %s：%s", job_id, exc)
        return False


def missing_output_artifacts(job_id: str, job_type: str) -> list[str]:
    directory = job_artifacts(job_id)
    return [name for name in required_output(job_type) if not _artifact_ready(directory / name)]


def missing_source_artifacts(source: Path, stage: str) -> list[str]:
    """Names a staged/redecode run needs but the source did not produce."""
    if not source.exists():
        return ["产物目录"]
    required = [*PLAN_ARTIFACTS, "semantic.npy" if stage == "semantic" else (),
                "latent.npy" if stage == "redecode" else ()]
    missing = [name for name in required if name and not _artifact_ready(source / name)]
    if _artifact_ready(source / "plan.json") and \
            (_read_json(source / "plan.json") or {}).get("abc") is not None and \
            not _artifact_ready(source / SCORE_ARTIFACT):
        missing.append(SCORE_ARTIFACT)
    return missing


def model_status_dict() -> dict:
    gpu = gpu_live()
    if MODEL["state"] == "loaded" and MODEL["device"] == "cuda":
        # 显存是我们自己的模型占的，不能据此判定「GPU 不可用、自动降级 CPU」
        gpu = {**gpu, "suitable": True, "reason": ""}
    return {"state": MODEL["state"], "device": MODEL["device"],
            "device_name": MODEL["device_name"], "reason": MODEL["reason"],
            "seconds": MODEL["seconds"], "busy": MODEL_BUSY,
            "gpu": gpu, "default_device": DEFAULT_DEVICE,
            # 顶层兼容字段（BUG-013）：前端曾按顶层取 free_gib/suitable
            "free_gib": gpu.get("free_gib"), "suitable": gpu.get("suitable")}


def busy_guard(reason: str) -> None:
    """生成中禁止动模型/开关。标志与任务出队状态都在 `LOCK` 里改，护栏也在 `LOCK` 里读，
    否则任务刚出队、标志还没写的那几毫秒里，切换和生成会各加载一次模型。
    （`model_status_dict` 里那次读只是给界面看的一拍快照，不做判定。）"""
    with LOCK:
        busy = MODEL_BUSY
    if busy:
        raise HTTPException(409, reason)


# ------------------------------------------------------------------- SSE plumbing
def _assign_event_seq(event: dict) -> dict:
    """Assign one process-wide sequence number; never mutate the caller's dict.

    同号事件同时进入历史缓冲，供断线重连按 Last-Event-ID 补发。
    """
    with EVENT_SEQ_LOCK:
        EVENT_SEQ["value"] += 1
        numbered = {**event, "seq": EVENT_SEQ["value"]}
        EVENT_HISTORY.append(numbered)
        return numbered


def _event_gap(previous: int | None, incoming: int | None) -> bool:
    """True when an event sequence proves the current window lost events."""
    if previous is None or incoming is None:
        return False
    return incoming > previous + 1


def broadcast(event: dict) -> None:
    # seq is assigned once per broadcast, so every subscriber sees the same order.
    numbered = _assign_event_seq(event)
    with SUB_LOCK:
        subscribers = list(SUBSCRIBERS)
    for sub in subscribers:
        try:
            sub.put_nowait(numbered)
        except queue.Full:
            # 队列满说明这个页面长时间没取：丢最旧的，保证最新进度一定送得出去。
            DROPPED["count"] += 1
            try:
                sub.get_nowait()
                sub.put_nowait(numbered)
            except (queue.Empty, queue.Full):
                pass
            if DROPPED["count"] % 100 == 1:
                log.warning("事件队列已满，累计丢弃 %d 条最旧事件（页面进度可能不及时）",
                            DROPPED["count"])


def drain_events() -> None:
    while True:
        try:
            EVENTS.get_nowait()
        except queue.Empty:
            return


def _persist(job: Job, *, replace: bool = True) -> None:
    try:
        if replace:
            db.upsert(job.to_db())
        else:
            db.insert_missing(job.to_db())
    except Exception as exc:  # noqa: BLE001 - the index is rebuildable from disk
        log.error("任务索引写入失败 id=%s：%s: %s", job.id, type(exc).__name__, exc)


def _live_done() -> dict:
    """已结束阶段的实测秒数；当前阶段不算完成，避免把半程当 100%。"""
    current = STAGE_PHASE.get(LIVE_STAGE["label"], "other")
    sums = {}
    for label, seconds in LIVE_STAGE["labels"].items():
        phase = STAGE_PHASE.get(label, "other")
        if phase == "other" or phase == current:
            continue
        sums[phase] = round(sums.get(phase, 0.0) + float(seconds or 0.0), 1)
    if LIVE_STAGE["load"]:
        sums["load"] = round(float(LIVE_STAGE["load"]), 1)
    return sums


def _live_begin(job_id: str) -> None:
    now = time.time()
    with LIVE_LOCK:
        LIVE_STAGE.update(id=job_id, started_at=now, load=0.0, labels={}, label="",
                          last=0.0, event=None, at=now)


def _live_loading(job_id: str) -> None:
    now = time.time()
    with LIVE_LOCK:
        if LIVE_STAGE["id"] != job_id:
            return
        LIVE_STAGE["event"] = {"label": "Loading model", "phase": "load", "completed": 0,
                               "total": None, "unit": None, "status": None,
                               "elapsed": 0, "rate": 0}
        LIVE_STAGE["at"] = now


def _live_loaded(job_id: str) -> None:
    """模型就绪后把计时起点挪到生成开始，加载耗时单独记一笔。"""
    now = time.time()
    with LIVE_LOCK:
        if LIVE_STAGE["id"] != job_id:
            return
        LIVE_STAGE["load"] = round(max(0.0, now - float(LIVE_STAGE["started_at"])), 1)
        LIVE_STAGE["started_at"] = now
        LIVE_STAGE["event"] = {"label": "Loading model", "phase": "load", "completed": 0,
                               "total": None, "unit": None, "status": None,
                               "elapsed": 0, "rate": 0}
        LIVE_STAGE["at"] = now


def _live_clear(job_id: str) -> None:
    with LIVE_LOCK:
        if LIVE_STAGE["id"] == job_id:
            LIVE_STAGE.update(id=None, started_at=0.0, load=0.0, labels={}, label="",
                              last=0.0, event=None, at=0.0)


def _apply_stage(event: dict) -> None:
    job_id = event.get("job")
    if not job_id:
        return
    job = JOBS.get(job_id)
    if job is None:
        return
    label = event.get("label") or ""
    if label != job.stage:
        job.stage = label
        job.updated_at = time.time()
        _persist(job)
    if event.get("total"):
        job.progress = f"{event.get('completed')}/{event.get('total')}"
    elif event.get("completed"):
        job.progress = str(event.get("completed"))
    phase = STAGE_PHASE.get(label, "other")
    event["stage_zh"] = STAGE_ZH.get(label, label)
    event["phase"] = phase
    elapsed = float(event.get("elapsed") or 0.0)
    now = time.time()
    with LIVE_LOCK:
        if LIVE_STAGE["id"] != job_id:
            LIVE_STAGE.update(id=job_id, started_at=now, load=0.0, labels={}, label="",
                              last=0.0, event=None)
        if LIVE_STAGE["label"] and LIVE_STAGE["label"] != label:
            prev = LIVE_STAGE["label"]
            LIVE_STAGE["labels"][prev] = max(float(LIVE_STAGE["labels"].get(prev, 0.0)),
                                             float(LIVE_STAGE["last"]))
        LIVE_STAGE["label"] = label
        LIVE_STAGE["last"] = elapsed
        if event.get("status"):
            LIVE_STAGE["labels"][label] = max(float(LIVE_STAGE["labels"].get(label, 0.0)), elapsed)
        LIVE_STAGE["event"] = {key: event.get(key) for key in
                               ("label", "completed", "total", "unit", "status", "elapsed", "rate")}
        LIVE_STAGE["event"]["phase"] = phase
        LIVE_STAGE["at"] = now
        event["started_at"] = LIVE_STAGE["started_at"]
        event["at"] = now
        event["done"] = _live_done()


def read_events(proc: subprocess.Popen) -> None:
    junk = 0
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except ValueError:
            # 协议只允许 JSON 行；混进来的 print 要留痕，但不能每条都写日志。
            junk += 1
            if junk <= 5 or junk % 100 == 0:
                log.warning("模型进程输出了非协议行（第 %d 条）：%s", junk, line[:300])
            continue
        if event.get("event") == "stage":
            event["job"] = CURRENT_JOB["id"]
            _apply_stage(event)
            broadcast(event)
            continue
        if event.get("event") == "score":
            _note_score(event.get("job") or CURRENT_JOB["id"])
            continue
        EVENTS.put(event)
    # 先把 died 放出去让 send() 解除等待；状态更新不拿 MODEL_LOCK，
    # 否则会和持有锁等 EVENTS 的 run_job 互相卡死。
    # 管道 EOF 往往比进程真正退出早几十毫秒，直接 poll() 会记成 None；
    # 先等退出码再放 died，日志顺序才是「进程结束 → 任务结束」。
    try:
        exit_code = proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        exit_code = None
    log.info("模型进程输出结束（exit=%s）", exit_code)
    EVENTS.put({"event": "died", "pid": proc.pid, "reason": "模型进程已退出"})
    # 是不是当前进程要等到写状态前再看：这 2 秒里可能已经换上新 worker，
    # 用提前算好的值会把刚加载好的模型误报成「未加载」。
    if WORKER.get("proc") is proc and MODEL["state"] in ("loaded", "loading"):
        MODEL.update(state="unloaded", device=None, device_name="", seconds=None, load={},
                     reason="模型进程已退出，请重新加载模型")
        broadcast({"event": "unloaded", "model": model_status_dict()})


def terminate_worker() -> None:
    proc, log_handle = WORKER.get("proc"), WORKER.get("log")
    WORKER["proc"], WORKER["log"] = None, None
    if proc is not None:
        if proc.poll() is None:
            try:
                proc.kill()
            except OSError as exc:
                log.warning("结束模型进程失败：%s: %s", type(exc).__name__, exc)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                log.warning("模型进程 10 秒内未退出（PID %s）", proc.pid)
        for stream in (proc.stdin, proc.stdout):
            try:
                if stream is not None:
                    stream.close()
            except OSError as exc:
                log.debug("关闭模型进程管道失败：%s", exc)
    if log_handle is not None:
        log_handle.close()
    # 强杀时 read_events 的 is_current 已经为假，这里必须自己收敛状态；
    # 否则 UI 会一直显示"已加载"，直到下一个任务静默重载。
    if MODEL["state"] in ("loaded", "loading"):
        MODEL.update(state="unloaded", device=None, device_name="", seconds=None, load={},
                     reason="模型进程已退出，请重新加载模型")
        broadcast({"event": "unloaded", "model": model_status_dict()})


def worker_process() -> subprocess.Popen:
    proc = WORKER.get("proc")
    if proc is not None and proc.poll() is None:
        return proc
    if proc is not None:
        terminate_worker()
    (ROOT / "runs").mkdir(parents=True, exist_ok=True)
    worker_log = ROOT / "runs" / "model_worker.log"
    try:
        if worker_log.stat().st_size > 5 * 1024 * 1024:
            worker_log.replace(worker_log.with_name(worker_log.name + ".1"))
    except OSError as exc:
        log.warning("轮转 model_worker.log 失败：%s: %s", type(exc).__name__, exc)
    WORKER["log"] = open(worker_log, "a", encoding="utf-8", errors="replace")
    env = {**os.environ, "YUE2_KIT": str(KIT), "HF_ENDPOINT": "https://hf-mirror.com",
           "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(
        [str(PYTHON_EXE), str(WORKER_SCRIPT)],
        cwd=str(ROOT), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=WORKER["log"],
        text=True, encoding="utf-8", errors="replace", bufsize=1, env=env,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    WORKER["proc"] = proc
    log.info("启动模型进程（PID %s）", proc.pid)
    drain_events()
    threading.Thread(target=read_events, args=(proc,), daemon=True).start()
    return proc


def worker_alive() -> bool:
    proc = WORKER.get("proc")
    return proc is not None and proc.poll() is None


class WorkerError(RuntimeError):
    def __init__(self, reason: str, cancelled: bool = False, action: str | None = None,
                 trace: str | None = None):
        super().__init__(reason)
        self.cancelled = cancelled
        self.action = action
        self.trace = trace


def send(cmd: dict, timeout: float) -> dict:
    proc = worker_process()
    rid = uuid.uuid4().hex[:8]
    cmd = {**cmd, "rid": rid}
    drain_events()
    try:
        proc.stdin.write(json.dumps(cmd, ensure_ascii=False) + "\n")
        proc.stdin.flush()
    except (BrokenPipeError, OSError, ValueError) as exc:
        # worker 已死时管道是断的：按"进程退出"上报，而不是让任务吃到 BrokenPipeError。
        log.warning("向模型进程发送 %s 失败：%s: %s", cmd.get("cmd"), type(exc).__name__, exc)
        terminate_worker()
        raise RuntimeError("模型进程已退出") from exc
    while True:
        try:
            event = EVENTS.get(timeout=timeout)
        except queue.Empty:
            log.warning("模型进程 %g 秒无响应（cmd=%s rid=%s），已强制结束",
                        timeout, cmd.get("cmd"), rid)
            terminate_worker()
            raise RuntimeError("模型进程无响应") from None
        if event.get("rid") and event["rid"] != rid:
            # 上一条已经放弃的命令迟到的回包，不能算到当前命令头上。
            log.debug("丢弃过期事件 rid=%s（当前 rid=%s）：%s",
                      event["rid"], rid, event.get("event"))
            continue
        if event.get("event") == "died":
            pid, current = event.get("pid"), WORKER.get("proc")
            if pid is not None and current is not None and pid != current.pid:
                # 上一条命令的进程晚于 drain 才投递退出：不能算到当前进程头上。
                log.debug("忽略旧进程 PID %s 的 died 事件（当前 PID %s）", pid, current.pid)
                continue
            terminate_worker()
            raise RuntimeError(event.get("reason") or "模型进程已退出")
        if event.get("event") == "error":
            log.warning("模型进程报告失败 action=%s trace=%s type=%s：%s",
                        event.get("action"), event.get("trace") or "-",
                        event.get("type"), event.get("reason"))
            raise WorkerError(event.get("reason") or "模型操作失败",
                              bool(event.get("cancelled")), event.get("action"),
                              event.get("trace"))
        return event


def _load_locked(device: str) -> str:
    global CUDA_BLOCKED
    MODEL.update(state="loading", reason="")
    broadcast({"event": "model", "model": model_status_dict()})
    common = {"model": str(MODEL_DIR), "vae": str(VAE_DIR), "budget": 12}
    try:
        event = send({"cmd": "load", "device": device, **common}, 900)
    except Exception as exc:
        terminate_worker()
        log.error("模型加载失败 device=%s：%s", device, exc)
        if device != "cuda":
            MODEL.update(state="unloaded", reason=f"模型加载失败：{exc}")
            broadcast({"event": "unloaded", "model": model_status_dict()})
            raise RuntimeError(f"模型加载失败：{exc}") from exc
        degraded = f"GPU 不可用，已自动降级为 CPU：{exc}"
        try:
            event = send({"cmd": "load", "device": "cpu", **common}, 900)
        except Exception as cpu_exc:
            terminate_worker()
            log.error("GPU 加载失败后 CPU 加载也失败：%s", cpu_exc)
            MODEL.update(state="unloaded", reason=f"模型加载失败：{cpu_exc}")
            broadcast({"event": "unloaded", "model": model_status_dict()})
            raise RuntimeError(f"{degraded}；CPU 加载也失败：{cpu_exc}") from cpu_exc
        CUDA_BLOCKED = str(exc)
        log.info("模型已加载到 CPU（GPU 降级）")
        MODEL.update(state="loaded", device="cpu", device_name=event.get("device_name", "CPU"),
                     seconds=event.get("seconds"), load=event.get("load", {}), reason=degraded)
        broadcast({"event": "loaded", "model": model_status_dict()})
        return "cpu"
    log.info("模型已加载 device=%s（%.1f 秒）", device, event.get("seconds") or 0.0)
    MODEL.update(state="loaded", device=device, device_name=event.get("device_name", device),
                 seconds=event.get("seconds"), load=event.get("load", {}), reason="")
    broadcast({"event": "loaded", "model": model_status_dict()})
    return device


def _unload_locked() -> None:
    global CUDA_BLOCKED
    MODEL.update(state="unloading")
    broadcast({"event": "model", "model": model_status_dict()})
    terminate_worker()
    log.info("模型已卸载")
    MODEL.update(state="unloaded", device=None, device_name="", seconds=None, load={}, reason="")
    CUDA_BLOCKED = ""
    broadcast({"event": "unloaded", "model": model_status_dict()})


def ensure_loaded(device: str) -> str:
    if device == "cuda" and CUDA_BLOCKED:
        # 静默换设备要留痕：任务请求写的是 cuda，跑的是 cpu，不记一句没法解释。
        log.info("GPU 已被记为不可用，本次按 CPU 运行；原因：%s", CUDA_BLOCKED[:300])
        device = "cpu"
        # 已经在 CPU 上加载时后面没有广播，界面会停在「你按了显卡」；这里补一条真实状态。
        broadcast({"event": "model", "model": model_status_dict()})
    if MODEL["state"] == "loaded" and MODEL["device"] == device and worker_alive():
        return device
    if MODEL["state"] == "loaded":
        _unload_locked()
    return _load_locked(device)


# ------------------------------------------------------------------- disk restore
def index_artifacts(job_id: str) -> dict:
    directory = job_artifacts(job_id)
    index = {}
    if directory.exists():
        for path in sorted(directory.iterdir()):
            if not (path.is_file() and path.name in ARTIFACT_NAMES):
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size > 0:
                index[path.name] = size
    return index


def job_from_row(row: dict) -> Job:
    request = row.get("requested") or {"style": row.get("style", ""), "lyrics": "",
                                       "cot": row.get("cot", "full"), "seed": row.get("seed")}
    job = Job(row["id"], row.get("type", "song"), request,
              source=row.get("parent_id"), title=row.get("title", ""),
              from_stage=row.get("from_stage") or None,
              parent_id=row.get("parent_id"))
    job.status = row.get("status", "queued")
    job.stage = row.get("stage", "")
    job.progress = row.get("progress", "")
    job.error = row.get("error", "")
    job.result = row.get("result") or {}
    job.artifacts = row.get("artifacts") or {}
    job.effective = row.get("effective") or {}
    job.favorite = bool(row.get("favorite"))
    job.demo = bool(row.get("demo"))
    job.work_name = row.get("work_name") or ""
    job.work_root = row.get("work_root")
    job.version_no = row.get("version_no")
    job.version_tag = row.get("version_tag") or ""
    job.version_label = row.get("version_label") or ""
    job.version_note = row.get("version_note") or ""
    job.kept = None if row.get("kept") is None else bool(row.get("kept"))
    job.hidden = bool(row.get("hidden"))
    job.variant_group = row.get("variant_group")
    job.ode_steps = row.get("ode_steps")
    job.vae = row.get("vae") or None
    job.created_at = row.get("created_at") or time.time()
    job.updated_at = row.get("updated_at") or job.created_at
    return job


def find_job(job_id: str) -> Job | None:
    """按 id 取任务：内存命中就返回，越窗的老任务回查索引。

    直接 `JOBS.get` 会把还在索引里的老任务当成不存在。越窗任务现查现用、不挤进内存窗口：
    一条老记录顶掉窗口里最新的一条，下次访问又得回查，等于没有缓存。没结束的任务例外，
    它必须住在内存里（worker 取任务、取消、改设备都按内存对象走）。
    """
    with LOCK:
        job = JOBS.get(job_id)
        if job is not None:
            return job
    row = db.get(job_id)
    if row is None:
        return None
    rebuilt = job_from_row(row)
    if rebuilt.status in ("queued", "running"):
        with LOCK:
            # 并发回填时以先住进来的那份为准，避免两个对象各自改状态。
            return JOBS.setdefault(rebuilt.id, rebuilt)
    return rebuilt


def _read_json(path: Path | None):
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        log.warning("读不到 %s：%s: %s", path, type(exc).__name__, exc)
        return None
    except ValueError as exc:
        log.warning("%s 不是合法 JSON，按缺失处理：%s", path, exc)
        return None


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError as exc:
        log.debug("取不到 %s 的时间：%s", path, exc)
        return None


DEMO_DIR = ROOT / "demo" / "showcase"
DEMO_ID = "song_demo_wanfeng"


def seed_demo() -> None:
    """Copy the bundled example song into the library exactly once."""
    flag = ROOT / "runs" / ".demo_seeded"
    if flag.exists() or not (DEMO_DIR / "audio.flac").exists() or db.get(DEMO_ID) is not None:
        return
    out = job_artifacts(DEMO_ID)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(DEMO_DIR, out, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("demo.json"))
    except OSError as exc:
        log.warning("示例曲导入跳过：%s: %s", type(exc).__name__, exc)
        return
    meta = _read_json(DEMO_DIR / "demo.json") or {}
    request = dict(_read_json(out / "request.json") or meta.get("request") or {})
    request.pop("id", None)
    request.setdefault("cot", "full")
    config = _read_json(out / "config.json") or {}
    generation = config.get("generation") or {}
    request.setdefault("vae", config.get("decoder_release") or "standard")
    request.setdefault("ode_steps", generation.get("ode_steps"))
    (out / "request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
    job = Job(DEMO_ID, "song", request, title=meta.get("title") or "示例曲 · 晚风",
              vae=request.get("vae"), ode_steps=request.get("ode_steps"), demo=True)
    job.status = "complete"
    job.result = _read_json(out / "result.json") or {}
    job.artifacts = index_artifacts(DEMO_ID)
    job.created_at = time.time()
    _persist(job)
    with LOCK:
        JOBS[DEMO_ID] = job
    try:
        flag.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    except OSError as exc:
        log.warning("示例曲导入标记写入失败：%s: %s", type(exc).__name__, exc)


def prune_trash() -> None:
    """Drop deleted jobs nobody came back for; the library itself is never touched."""
    if not TRASH_DIR.exists():
        return
    cutoff = time.time() - TRASH_TTL_DAYS * 86400
    removed, freed = 0, 0
    for entry in sorted(TRASH_DIR.iterdir()):
        try:
            if not entry.is_dir() or entry.stat().st_mtime >= cutoff:
                continue
            size = sum(part.stat().st_size for part in entry.rglob("*") if part.is_file())
            shutil.rmtree(entry)
            removed, freed = removed + 1, freed + size
        except OSError as exc:
            log.warning("清理回收站失败 %s：%s: %s", entry.name, type(exc).__name__, exc)
    if removed:
        log.info("回收站清理：删除 %d 个超过 %d 天的任务，释放 %.1f MB",
                 removed, TRASH_TTL_DAYS, freed / 1024 / 1024)


def _disk_outcome(job_id: str, job_type: str = "song") -> tuple[str, str, dict] | None:
    """Finished artifacts on disk win over a late failure marker.

    The worker writes result.json only after a handler returns, so a success can
    be followed by a failure.json caused by the final event not reaching a dead
    server. Neither should downgrade the other.
    """
    out_dir = JOBS_DIR / job_id / "out" / job_id
    result_path, failure_path = out_dir / "result.json", out_dir / "failure.json"
    result = _read_json(result_path) if result_path.exists() else None
    if result is not None:
        missing = missing_output_artifacts(job_id, job_type)
        if missing:
            names = "、".join(OUTPUT_LABELS.get(name, name) for name in missing)
            failure = _read_json(failure_path) if failure_path.exists() else None
            if isinstance(failure, dict) and failure.get("reason"):
                status = "cancelled" if failure.get("status") == "cancelled" else "failed"
                return status, failure["reason"], {}
            return "failed", f"任务结果已写出，但缺少{names}：可能在保存产物时被中断，请重新生成", {}
        return "complete", "", result
    failure = _read_json(failure_path) if failure_path.exists() else None
    if failure is not None:
        status = "cancelled" if failure.get("status") == "cancelled" else "failed"
        return status, failure.get("reason", "unknown"), {}
    return None


def _reconcile_from_disk() -> list[dict]:
    """Disk results win over stale queue rows, including rows a dead predecessor marked failed."""
    fixed = []
    rows = [row for status in ("queued", "running", "failed")
            for row in db.list_jobs(status=status, limit=None)]
    for row in rows:
        outcome = _disk_outcome(str(row["id"]), row.get("type") or "song")
        if outcome is None:
            continue
        status, error, result = outcome
        if row.get("status") == "failed" and status != "complete":
            continue  # 真正的失败保留原文，不被 failure.json 重写一遍
        truncated = result.get("truncated")
        row.update(status=status, error=error, result=result, stage="", progress="",
                   artifacts=index_artifacts(str(row["id"])),
                   duration=result.get("audio_seconds"),
                   truncated=1 if (truncated is True or
                                   (isinstance(truncated, dict) and any(truncated.values()))) else 0,
                   updated_at=time.time())
        # 崩溃恢复的完成产物走不到 run_job 的收尾，这里补上同样的自动保留规则。
        if status == "complete" and row.get("kept") is None and row.get("work_root") \
                and not db.group_has_kept(str(row["work_root"])):
            row["kept"] = 1
            log.info("重启收敛：组内首个完成版本自动保留 id=%s root=%s",
                     row["id"], row["work_root"])
        db.upsert(row)
        fixed.append({"id": row["id"], "status": status, "duration": row.get("duration"),
                      "title": row.get("title") or ""})
    return fixed


def _notice(kind: str, jobs: list[dict], count: int | None = None) -> None:
    if not jobs and not count:
        return
    with LOCK:
        NOTICES.append({"id": f"{kind}-{int(time.time() * 1000)}-{uuid.uuid4().hex[:4]}",
                        "kind": kind, "at": time.time(), "count": count or len(jobs),
                        "jobs": jobs[:10]})


def restore_jobs() -> None:
    """Reconcile disk artifacts into the DB, then keep the newest `MEMORY_JOBS` rows in memory."""
    db.init()
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    known = db.all_ids()
    for directory in sorted(JOBS_DIR.iterdir()):
        if not directory.is_dir() or directory.name in known:
            continue
        out_dir = directory / "out" / directory.name
        meta = directory / ".meta"
        request = _read_json(next((path for path in (out_dir / "request.json", meta / "request.json",
                                                     meta / "job_request.json") if path.exists()), None))
        if request is None:
            continue
        job_type = _infer_type(request, directory)
        outcome = _disk_outcome(directory.name, job_type)
        status, error, result = outcome or ("failed", "任务在写出产物前被终止", {})
        # 执行参数（种子外的 ode_steps/vae/staged 源）不进官方 request，单独落 .meta；
        # 索引丢失后重建时也要还原，否则「渲染成歌 / 重解码」会静默回到默认值。
        extra = _read_json(meta / "job.json") or {}
        source = extra.get("source") or request.get("source")
        job = Job(directory.name, job_type, request,
                  source=source,
                  from_stage=extra.get("from_stage") or request.get("from_stage"),
                  vae=extra.get("vae") or request.get("vae"),
                  ode_steps=extra.get("ode_steps", request.get("ode_steps")),
                  title=str(extra.get("title") or ""),
                  parent_id=source, demo=directory.name == DEMO_ID)
        if job.demo:
            job.title = "示例曲 · 晚风"
        job.status, job.result, job.error = status, result, error
        job.artifacts = index_artifacts(directory.name)
        if request.get("device") is None:
            for candidate in (meta / "request.json", meta / "job_request.json"):
                extra = _read_json(candidate)
                if extra and extra.get("device"):
                    request["device"] = extra["device"]
                    break
        job.created_at = _mtime(out_dir / "result.json") or _mtime(directory) or time.time()
        # 只补缺：已入库的行带着收藏/标题/参数，不能被磁盘重建结果覆盖。
        _persist(job, replace=False)
    recovered = _reconcile_from_disk()
    if recovered:
        log.info("重启收敛：%d 个任务从磁盘恢复最终状态", len(recovered))
        _notice("recovered", recovered)
    # 必须在状态收敛之后、装入内存之前：kept 默认取组内最新完成的版本，读的是最终 status。
    backfilled = db.backfill_work_fields()
    if backfilled:
        log.info("重启收敛：%d 个存量任务回填了作品/版本族谱字段", backfilled)
    pending = [{"id": row["id"], "title": row.get("title") or ""}
               for status in ("queued", "running")
               for row in db.list_jobs(status=status, limit=None)]
    interrupted = db.fail_unfinished("服务重启，任务中断")
    if interrupted:
        log.info("重启收敛：%d 个未完成任务标记为失败", interrupted)
        _notice("interrupted", pending, interrupted)
    with LOCK:
        JOBS.clear()
        # 内存只装最近 MEMORY_JOBS 条；更早的任务由 find_job() 按需回查，不是不存在。
        for row in db.list_jobs(limit=MEMORY_JOBS):
            JOBS[row["id"]] = job_from_row(row)


def _infer_type(request: dict, directory: Path | None = None) -> str:
    """Disk-only restore has no DB row; use explicit metadata first, then artifacts."""
    declared = str(request.get("type") or "").strip()
    if declared in JOB_TYPES:
        return declared
    if directory is not None:
        try:
            stored = (directory / ".meta" / "type").read_text(encoding="utf-8").strip()
        except OSError:
            stored = ""
        if stored in JOB_TYPES:
            return stored
    if request.get("source") and request.get("from"):
        return "staged"
    if directory is not None:
        out = directory / "out" / directory.name
        # 只出谱没有音频，但一定写了 plan.json 和 score.abc。
        if (out / "result.json").exists() and not _artifact_ready(out / AUDIO_ARTIFACT) \
                and _artifact_ready(out / "plan.json"):
            return "plan"
    return "song"


def cancel_requested(job: Job) -> bool:
    return (job_dir(job.id) / ".meta" / "cancel").exists()


# ------------------------------------------------------------------- job execution
def build_worker_command(job: Job) -> tuple[dict, float]:
    directory = job_dir(job.id)
    (directory / ".meta").mkdir(parents=True, exist_ok=True)
    try:
        (directory / ".meta" / "type").write_text(job.type, encoding="utf-8")
        (directory / ".meta" / "job.json").write_text(json.dumps({
            "type": job.type, "source": job.source, "from_stage": job.from_stage,
            "vae": job.vae, "ode_steps": job.ode_steps, "title": job.title,
        }, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        log.warning("写入任务类型失败 %s：%s: %s", job.id, type(exc).__name__, exc)
    cancel_file = directory / ".meta" / "cancel"
    log_path = directory / ".meta" / "run.log"
    log_path.write_text("", encoding="utf-8")
    out = str(directory / "out")
    if job.type in ("song", "plan", "rerender"):
        request_path = directory / ".meta" / "request.json"
        request_path.write_text(json.dumps({**job.request, "id": job.id}, ensure_ascii=False, indent=2),
                                encoding="utf-8")
        cmd = {"cmd": "plan" if job.type == "plan" else "generate",
               "request": str(request_path), "output": out, "id": job.id,
               "log": str(log_path), "cancel_file": str(cancel_file),
               "ode_steps": job.ode_steps, "attention": db.get_settings()}
        return cmd, 6 * 3600
    source = job_artifacts(job.source) if job.source else None
    stage = "redecode" if job.type == "redecode" else (job.from_stage or "plan")
    missing = missing_source_artifacts(source, stage) if source is not None else ["产物目录"]
    if missing:
        raise RuntimeError(f"源任务缺少产物：{', '.join(missing)}")
    if job.type == "staged":
        cmd = {"cmd": "staged", "from": job.from_stage or "plan", "src": str(source),
               "output": out, "id": job.id, "log": str(log_path), "cancel_file": str(cancel_file),
               "ode_steps": job.ode_steps, "attention": db.get_settings()}
        return cmd, 6 * 3600
    if job.type == "redecode":
        cmd = {"cmd": "redecode", "src": str(source), "output": out, "id": job.id,
               "log": str(log_path), "cancel_file": str(cancel_file),
               "vae": job.vae or "standard"}
        return cmd, 3600
    raise RuntimeError(f"未知任务类型：{job.type}")


def _source_request(job: Job) -> dict | None:
    source = find_job(job.source) if job.source else None
    return {"requested": source.request} if source is not None else None


def _source_lyrics(job: Job) -> str:
    return jobs_text.effective_lyrics("redecode", job.request, None,
                                      lambda: _source_request(job))


def _effective_lyrics(job: Job) -> str:
    """Derived jobs keep a lyrics snapshot so exports survive source deletion."""
    snapshot = None
    if job.type in ("staged", "redecode"):
        snapshot = _read_json(job_dir(job.id) / ".meta" / "effective_request.json")
    return jobs_text.effective_lyrics(job.type, job.request, snapshot,
                                      lambda: _source_request(job))


def _job_lyrics(job: Job) -> str:
    return _effective_lyrics(job)


def schedule_alignment(job: Job) -> None:
    """Queue lyric timing once the audio exists; the align thread runs on CPU."""
    if job.status != "complete":
        return
    directory = job_artifacts(job.id)
    audio = align.find_audio(directory)
    lyrics = _job_lyrics(job)
    out = directory / "lyrics.json"
    if audio is None or not lyrics.strip() or out.exists():
        return
    align.schedule(job.id, audio, lyrics, out,
                   lambda ok: broadcast({"event": "lyrics", "job": job.id, "ok": bool(ok)}))


def _adopt_disk_result(job: Job) -> bool:
    """A dead worker can still have finished; never downgrade a complete result.json."""
    outcome = _disk_outcome(job.id, job.type)
    if not outcome or outcome[0] != "complete":
        return False
    job.status, job.error, job.result = "complete", "", outcome[2]
    job.stage, job.progress = "", ""
    job.artifacts = index_artifacts(job.id)
    return True


def run_job(job: Job) -> None:
    global MODEL_BUSY
    with LOCK:
        if job.status == "cancelled":
            return
        job.status = "running"
        # 和「切换设备时改排队任务」共用一把锁：开跑后才读设备，改设备不会漏掉已出队的任务。
        device = job.request.get("device") or "auto"
        if device == "auto":
            device = MODEL["device"] if MODEL["state"] == "loaded" else DEFAULT_DEVICE
        MODEL_BUSY = True
    job.stage, job.progress = "", ""
    needs_load = not (MODEL["state"] == "loaded" and MODEL["device"] == device)
    if needs_load:
        job.stage = "Loading model"
    CURRENT_JOB["id"] = job.id
    _live_begin(job.id)
    if needs_load:
        _live_loading(job.id)
    _persist(job)
    broadcast(_job_event(job, status="running"))
    broadcast({"event": "model", "model": model_status_dict()})
    log.info("任务开始 id=%s type=%s device=%s", job.id, job.type, device)
    try:
        with MODEL_LOCK:
            device = ensure_loaded(device)
        # 已加载的模型没有加载耗时，别把几毫秒记成一段“准备模型”
        if needs_load:
            _live_loaded(job.id)
        job.request["device"] = device
        if job.status == "cancelled" or cancel_requested(job):
            raise WorkerError("已取消（尚未开始生成）", cancelled=True)
        cmd, timeout = build_worker_command(job)
        if job.status == "cancelled" or cancel_requested(job):
            raise WorkerError("已取消（尚未开始生成）", cancelled=True)
        with MODEL_LOCK:
            send(cmd, timeout)
        result_path = job_artifacts(job.id) / "result.json"
        if result_path.exists():
            job.result = _read_json(result_path) or {}
        job.artifacts = index_artifacts(job.id)
        _recover_score_from_plan(job.id)
        job.artifacts = index_artifacts(job.id)
        missing = missing_output_artifacts(job.id, job.type)
        if missing:
            names = "、".join(OUTPUT_LABELS.get(name, name) for name in missing)
            raise WorkerError(f"生成流程结束，但没有写出{names}（文件为空或不存在）；"
                              "请查看任务日志后重试")
        job.status = "complete"
        job.stage, job.progress = "", ""
        schedule_alignment(job)
    except WorkerError as exc:
        if _adopt_disk_result(job):
            log.warning("任务回传失败但磁盘产物完整，按成功收录 id=%s：%s", job.id, exc)
            schedule_alignment(job)
        else:
            job.status = "cancelled" if exc.cancelled else "failed"
            job.error = str(exc) if exc.cancelled or not exc.trace \
                else f"{exc}（日志编号 {exc.trace}）"
            if job.status == "cancelled":
                _mark_cancelled(job, str(exc))
            else:
                log.error("任务失败 id=%s type=%s trace=%s：%s", job.id, job.type,
                          exc.trace or "-", exc)
                _ensure_failure_log(job)
    except Exception as exc:  # noqa: BLE001
        if _adopt_disk_result(job):
            log.warning("任务异常但磁盘产物完整，按成功收录 id=%s：%s", job.id, exc)
            schedule_alignment(job)
        else:
            job.status = "failed"
            job.error = str(exc)
            log.exception("任务异常 id=%s type=%s", job.id, job.type)
            _ensure_failure_log(job)
    finally:
        CURRENT_JOB["id"] = None
        with LOCK:
            MODEL_BUSY = False
        broadcast({"event": "model", "model": model_status_dict()})
        job.artifacts = job.artifacts or index_artifacts(job.id)
        # 组内还没有保留版本时，第一条跑完的版本自动置 ●；之后完成的候选不抢标记。
        if job.status == "complete" and job.kept is None and job.work_root \
                and not db.group_has_kept(job.work_root):
            job.kept = True
            log.info("组内首个完成版本自动保留 id=%s root=%s", job.id, job.work_root)
        _persist(job)
        broadcast(_job_event(job, status=job.status, error=job.error,
                             duration=(job.result or {}).get("audio_seconds")))
        _live_clear(job.id)
        log.info("任务结束 id=%s status=%s", job.id, job.status)


def _ensure_failure_log(job: Job) -> None:
    """让失败任务总有可读日志（BUG-014）：worker 异常路径若未落 run.log，这里补一条，
    避免前端日志弹窗空白、无法定位根因。"""
    path = job_dir(job.id) / ".meta" / "run.log"
    try:
        if path.exists() and path.stat().st_size > 0:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"任务失败 id={job.id} type={job.type} error={job.error}\n")
    except OSError as exc:
        log.warning("补写失败日志失败 id=%s：%s", job.id, exc)


def _mark_cancelled(job: Job, reason: str) -> None:
    directory = job_artifacts(job.id)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "failure.json").write_text(
            json.dumps({"status": "cancelled", "reason": reason}, ensure_ascii=False, indent=2),
            encoding="utf-8")
    except OSError as exc:
        log.warning("写入取消标记失败 id=%s：%s: %s", job.id, type(exc).__name__, exc)


def worker() -> None:
    while True:
        job_id = QUEUE.get()
        job = JOBS.get(job_id)
        if job is None or job.status == "cancelled":
            continue
        try:
            run_job(job)
        except Exception as exc:  # noqa: BLE001 - last resort; run_job already persists
            job.status = "failed"
            job.error = str(exc)
            _persist(job)
            broadcast(_job_event(job, status="failed", error=job.error))
            log.exception("任务线程兜底捕获 id=%s", job.id)


# ------------------------------------------------------------------- create jobs
def _new_id(prefix: str = "song") -> str:
    return f"{prefix}_" + time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:4]


def _official_request(body: JobRequest, source: dict | None = None) -> dict:
    request: dict = {}
    if source:
        for key in ("style", "lyrics", "cot", "seed", "cfg_scale", "abc",
                    "abc_sampling", "semantic_sampling"):
            if source.get(key) is not None:
                request[key] = source[key]
    for key in ("style", "lyrics", "seed", "cfg_scale", "abc_sampling", "semantic_sampling"):
        value = getattr(body, key)
        if value is not None and value != "":
            request[key] = value
    if body.abc is not None:
        request["abc"] = body.abc
    for key in ("style", "lyrics"):
        if isinstance(request.get(key), str):
            request[key] = request[key].lstrip("\ufeff")
    if body.type in ("song", "plan") and request.get("seed") is None:
        # 界面写「留空 = 随机」，官方 SongRequest 默认却是固定 831001：这里补真正的随机种子。
        # 上限取 2**53：更大的整数在浏览器 JSON 解析时会丢精度，回填的种子就不再是实际用的那个。
        request["seed"] = secrets.randbelow(2 ** 53)
    request["cot"] = body.cot or request.get("cot") or "full"
    request["device"] = body.device
    return request


_TEXT_TOKENIZER: dict = {"value": None, "failed": False}
_LYRICS_CHAR_CAP = 100000


def _text_tokenizer():
    """Official YuE2 text BPE, loaded once. None if the server env lacks tiktoken."""
    if _TEXT_TOKENIZER["value"] is not None or _TEXT_TOKENIZER["failed"]:
        return _TEXT_TOKENIZER["value"]
    try:
        source = str(KIT / "src")
        if source not in sys.path:
            sys.path.insert(0, source)
        from yue2.tokenization_yue2 import YuE2TextTokenizer
        _TEXT_TOKENIZER["value"] = YuE2TextTokenizer(MODEL_DIR / "qwen.tiktoken")
    except Exception as exc:  # noqa: BLE001 - fall back to a conservative byte budget
        _TEXT_TOKENIZER["failed"] = True
        log.warning("分词器不可用，改用字符数估算上下文：%s: %s", type(exc).__name__, exc)
    return _TEXT_TOKENIZER["value"]


def _text_tokens(text: str) -> int | None:
    tokenizer = _text_tokenizer()
    if tokenizer is None:
        return None
    try:
        return len(tokenizer.encode(text))
    except Exception:  # noqa: BLE001 - never block a job on the estimator
        return None


def _check_context_budget(body: JobRequest) -> None:
    """Reject lyrics that cannot fit the engine context before queuing."""
    if body.type not in ("song", "plan", "rerender"):
        return
    if len(body.lyrics) > _LYRICS_CHAR_CAP:
        raise HTTPException(400, f"歌词过长：超过 {_LYRICS_CHAR_CAP} 字符，请先精简")
    if body.type == "rerender" and len(body.abc or "") > _LYRICS_CHAR_CAP:
        raise HTTPException(400, "谱面过长：请先精简谱面")
    from yue2.protocol import CONTEXT, INSTRUCTIONS, GenerationConfig
    text = f"{INSTRUCTIONS[body.cot]}\n[Tags]\n{body.style}\n[Lyrics]\n{body.lyrics}\n"
    tokens = _text_tokens(text)
    if tokens is None:
        if len(body.lyrics) > 12000:
            raise HTTPException(400, "歌词过长：服务端 tokenizer 不可用，按保守上限判定超过 12000 字符")
        return
    defaults = GenerationConfig()
    abc_max = (body.abc_sampling or {}).get("max_tokens", defaults.abc.max_tokens)
    semantic_max = (body.semantic_sampling or {}).get("max_tokens", defaults.semantic.max_tokens)
    if body.type == "plan":
        needed = tokens + 3 + abc_max
    elif body.type == "rerender":
        abc_tokens = _text_tokens(body.abc or "") or 0
        needed = tokens + 5 + abc_tokens + semantic_max
    else:
        needed = tokens + 5 + abc_max + semantic_max
    if needed > CONTEXT:
        raise HTTPException(
            400,
            f"歌词或风格过长：约 {tokens} 个文本 token，加上生成预算后超过模型上下文 {CONTEXT}；"
            "请删减歌词/风格，或把 max_tokens 调小")


def _clean_title(value) -> str:
    try:
        return jobs_text.clean_title(value)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


SAMPLING_LIMITS = {
    "temperature": (0, 5, False),
    "top_p": (0.01, 1, False),
    "top_k": (1, 1000, True),
    "repetition_penalty": (0.01, 5, False),
    "penalty_window": (1, 100, True),
    "min_tokens": (0, 20000, True),
    "max_tokens": (1, 20000, True),
}


def _check_sampling(name: str, payload: dict | None) -> None:
    if payload is None:
        return
    if not isinstance(payload, dict):
        raise HTTPException(400, "采样参数必须是对象")
    for key, value in payload.items():
        if key not in SAMPLING_LIMITS:
            raise HTTPException(400, f"不支持的采样参数：{name}.{key}")
        low, high, is_int = SAMPLING_LIMITS[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise HTTPException(400, f"{name}.{key} 必须是数字")
        if is_int and not isinstance(value, int):
            raise HTTPException(400, f"{name}.{key} 必须是整数")
        if not low <= value <= high:
            raise HTTPException(400, f"{name}.{key} 必须在 {low} 到 {high} 之间")
    from yue2.protocol import GenerationConfig
    defaults = GenerationConfig()
    base = defaults.abc if name == "abc_sampling" else defaults.semantic
    low = payload.get("min_tokens", base.min_tokens)
    high = payload.get("max_tokens", base.max_tokens)
    if not 0 <= low <= high:
        raise HTTPException(400, f"min_tokens 不能大于 max_tokens（当前 {low} > {high}）")


_PROMPT_MARKER_RE = re.compile(r"\[\s*(?:tags|lyrics)\s*\]", re.IGNORECASE)
_FULLWIDTH_TAG_RE = re.compile(r"【[^】]{1,20}】")


def _validate(body: JobRequest) -> None:
    if body.type not in JOB_TYPES:
        raise HTTPException(400, f"type must be one of {JOB_TYPES}")
    if body.device not in ("cpu", "cuda", "auto"):
        raise HTTPException(400, "device must be cpu, cuda or auto")
    if body.cot not in ("full", "melody", "off"):
        raise HTTPException(400, "cot must be full, melody or off")
    if body.type in ("song", "plan", "rerender"):
        if not isinstance(body.style, str):
            raise HTTPException(400, "风格字段必须是文本")
        if not isinstance(body.lyrics, str):
            raise HTTPException(400, "歌词字段必须是文本")
        # 从文本编辑器粘贴常带 BOM；先去掉再判空，否则只有 BOM 的字段会通过必填检查。
        body.style, body.lyrics = body.style.lstrip("\ufeff"), body.lyrics.lstrip("\ufeff")
        if not body.style.strip():
            raise HTTPException(400, "缺少 style（风格）")
        if not body.lyrics.strip():
            raise HTTPException(400, "缺少 lyrics（歌词）")
        if _PROMPT_MARKER_RE.search(body.style):
            raise HTTPException(400, "风格里不能出现 [Tags] / [Lyrics]：它们是提示词分区，会把歌词区提前截断")
        if _PROMPT_MARKER_RE.search(body.lyrics):
            raise HTTPException(400, "歌词里不能出现 [Tags] / [Lyrics] 提示词分区标记，请删掉")
    if body.type == "rerender" and not (body.abc or "").strip():
        raise HTTPException(400, "abc score is empty")
    if body.work_group and len(body.work_group) > 120:
        raise HTTPException(400, "work_group 最长 120 个字符")
    if body.type in ("staged", "redecode") and not body.source:
        raise HTTPException(400, "source job id is required")
    if body.type == "staged" and (body.from_stage or "plan") not in ("plan", "semantic"):
        raise HTTPException(400, "from_stage must be plan or semantic")
    if body.type == "redecode":
        decoder = body.vae or "standard"
        if decoder == "legacy" and not VAE_LEGACY_DIR.exists():
            raise HTTPException(409, "YuE2-Vae-legacy 未安装，无法用该解码器重渲")
        if decoder not in ("standard", "legacy"):
            path = Path(decoder)
            if not path.is_dir():
                raise HTTPException(400, f"解码器必须是 standard、legacy 或本地解码器目录：{decoder}")
            if not (path / "config.json").is_file():
                raise HTTPException(400, f"本地解码器目录缺少 config.json：{decoder}")
    if body.seed is not None and not 0 <= body.seed < 2 ** 53:
        raise HTTPException(400, "seed must be an integer in [0, 2**53)：更大的整数在浏览器里会丢精度")
    if body.ode_steps is not None and not 1 <= body.ode_steps <= 200:
        raise HTTPException(400, "ode_steps must be between 1 and 200")
    if body.cfg_scale is not None and not 0 <= body.cfg_scale <= 20:
        raise HTTPException(400, "cfg_scale must be between 0 and 20")
    _check_sampling("abc_sampling", body.abc_sampling)
    _check_sampling("semantic_sampling", body.semantic_sampling)
    _check_context_budget(body)


def _job_truncated(job: Job) -> bool:
    value = (job.result or {}).get("truncated")
    if isinstance(value, dict):
        return any(bool(item) for item in value.values())
    return bool(value)


def _derive_work(job: "Job", body: "JobRequest", source_job: "Job | None") -> None:
    """建任务时一次性写定作品族谱字段（docs/歌曲管理优化方案-2026-11-12.md）。

    三条来源，优先级从高到低：
    1. 派生任务（rerender/staged/redecode）：继承源任务的 work_root / work_name；
    2. 候选批次（song + body.work_group）：跟组内已有行归组，首个候选自成根；
    3. 独立任务：work_root = 自身，作品名取歌名 > 歌词首行。
    version_tag（自动标签）同步写定；title 为空时用「作品名 · 标签」兜底。
    """
    lyrics_head = _lyric_head(str(body.lyrics or ""))
    note = str(body.version_note or "").strip()[:500]
    if source_job is not None:
        job.work_root = source_job.work_root or source_job.id
        job.work_name = source_job.work_name or source_job.title or lyrics_head or "未命名作品"
        job.variant_group = job.work_root
        verb = {"rerender": "改谱", "staged": "成歌", "redecode": "换解码器"}.get(body.type)
        if verb:
            job.version_tag = f"{verb} {db.next_type_index(job.work_root, body.type, verb)}"
        job.version_note = note
    elif str(body.work_group or "").strip():
        group_id = str(body.work_group).strip()
        group = db.find_group_row(group_id)
        if group is not None:
            job.work_root = group.get("work_root") or group["id"]
            job.work_name = group.get("work_name") or job.title or lyrics_head or "未命名作品"
            index = db.next_candidate_index(job.work_root)
        else:
            job.work_root = job.id
            job.work_name = _clean_title(body.work_name) or _clean_title(body.title) or lyrics_head or "未命名作品"
            index = 0
        job.variant_group = group_id
        job.version_tag = f"候选 {chr(ord('A') + min(index, 25))}"
        job.version_note = note
    else:
        job.work_root = job.id
        job.work_name = _clean_title(body.work_name) or _clean_title(body.title) or lyrics_head or "未命名作品"
        job.variant_group = job.id
        job.version_note = note
    job.version_no = db.next_version_no(job.work_root)
    if not job.title:
        job.title = f"{job.work_name} · {job.version_tag}" if job.version_tag else job.work_name


@app.post("/api/jobs")
def create_job(body: JobRequest) -> dict:
    _validate(body)
    source_job = None
    if body.source:
        # 内存只常驻最近 MEMORY_JOBS 条；更早的任务还在索引里，不能当成不存在
        source_job = find_job(body.source)
        if source_job is None:
            raise HTTPException(404, "source job not found")
        # 重渲染借用源的 cot 等谱系参数，风格/歌词以请求体为准，谱面由请求体自带，不依赖源产物；
        # 分阶段渲染与重解码才必须从完整的源产物继续。
        if body.type in ("staged", "redecode"):
            if source_job.status != "complete":
                raise HTTPException(404, "source job not found or not complete")
            if body.type == "staged" and _job_truncated(source_job):
                raise HTTPException(409, "源任务的乐谱被截断，直接渲染会得到残曲；请改短歌词后重新出谱")
            stage = "redecode" if body.type == "redecode" else (body.from_stage or "plan")
            missing = missing_source_artifacts(job_artifacts(body.source), stage)
            if missing:
                raise HTTPException(409, f"源任务缺少产物：{', '.join(missing)}")
    source_request = source_job.request if source_job else None
    request = _official_request(body, source_request)
    # 快照必须在任务目录创建前算出；不能用 _job_lyrics 反查尚未落盘的快照。
    effective_lyrics = ""
    if body.type in ("staged", "redecode"):
        effective_lyrics = jobs_text.effective_lyrics(
            body.type, request, None, lambda: source_job.request if source_job else None)
    prefix = "plan" if body.type == "plan" else "batch" if body.type == "redecode" else "song"
    job_id = _new_id(prefix)
    job = Job(job_id, body.type, request,
              source=body.source, from_stage=body.from_stage, vae=body.vae,
              ode_steps=body.ode_steps, title=_clean_title(body.title),
              parent_id=body.source)
    if source_job and body.type != "redecode":
        job.effective = source_job.effective
    # 分配（读组信息、算版本号/标签）、派生歌词快照与落库必须处于同一临界区：
    # 放在锁外时，并发建任务会各自读到同一个 MAX(version_no)/候选序号再先后插入，
    # 造成重号、同批候选分裂成多个作品。锁顺序：本 LOCK → SQLite 写事务；
    # move_version 只拿 db.MOVE_VERSION_LOCK，不回调任何持本 LOCK 的代码，两条链不交叉。
    with LOCK:
        try:
            _derive_work(job, body, source_job)
        except HTTPException:
            raise
        except Exception:
            log.exception("写作品族谱失败 id=%s type=%s", job_id, body.type)
            raise
        if effective_lyrics:
            try:
                (job_dir(job_id) / ".meta").mkdir(parents=True, exist_ok=True)
                (job_dir(job_id) / ".meta" / "effective_request.json").write_text(
                    json.dumps({"lyrics": effective_lyrics, "source": body.source},
                               ensure_ascii=False, indent=2), encoding="utf-8")
            except OSError as exc:
                log.warning("写入派生任务有效歌词失败 %s：%s", job_id, exc)
        JOBS[job_id] = job
        _persist(job)
        QUEUE.put(job_id)
    broadcast(_job_event(job, status="queued"))
    return job.to_dict()


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    with LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(404, "任务不存在")
        if job.status not in ("queued", "running"):
            raise HTTPException(409, "任务已结束，无需取消")
        queued = job.status == "queued"
        if queued:
            job.status = "cancelled"
            job.error = "已取消（尚未开始）"
    directory = job_dir(job_id) / ".meta"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "cancel").write_text("1", encoding="utf-8")
    if queued:
        _persist(job)
        broadcast(_job_event(job, status="cancelled"))
    log.info("取消请求 id=%s 状态=%s，生效=%s", job_id, "queued" if queued else "running",
             "已取消" if queued else "下一阶段边界")
    return {"cancelled": job_id, "effective": "queued" if queued else "next stage boundary",
            "row": _api_row(job.to_db()) if queued else None}


# ------------------------------------------------------------------- read jobs
def _lyric_head(text: str) -> str:
    return jobs_text.lyric_head(text)


def _lyric_units(text: str) -> float:
    return jobs_text.lyric_units(text)


def _api_row(row: dict, full: bool = False) -> dict:
    """full=False 是列表口径：requested 只留摘要，歌词正文和谱面不进 JSON。"""
    request = dict(row.get("requested") or {})
    lyrics = str(request.get("lyrics") or "")
    lyrics_source = lyrics
    if not lyrics and row.get("parent_id"):
        parent = db.get(row["parent_id"])
        if parent is not None:
            lyrics_source = str((parent.get("requested") or {}).get("lyrics") or "")
    if not full:
        # 列表只带摘要：歌词正文一条就能到几 MB，而列表渲染一个字段都不用它
        request.pop("lyrics", None)
        request.pop("abc", None)
    result = row.get("result") or {}
    truncated = row.get("truncated")
    if truncated is None:
        truncated = result.get("truncated")
        if isinstance(truncated, dict):
            truncated = any(bool(v) for v in truncated.values())
    payload = {
        "id": row["id"], "type": row.get("type", "song"), "status": row.get("status"),
        "stage": STAGE_ZH.get(row.get("stage", ""), row.get("stage", "")),
        "stage_raw": row.get("stage", ""), "progress": row.get("progress", ""),
        "device": row.get("device") or request.get("device", ""),
        "style": row.get("style") or request.get("style", ""),
        "seed": row.get("seed"), "cot": row.get("cot", ""),
        "title": row.get("title", ""), "parent_id": row.get("parent_id"),
        "variant_group": row.get("variant_group"), "favorite": bool(row.get("favorite")),
        "work_name": row.get("work_name") or "", "work_root": row.get("work_root"),
        "version_no": row.get("version_no"), "version_tag": row.get("version_tag") or "",
        "version_label": row.get("version_label") or "",
        "version_note": row.get("version_note") or "",
        "kept": None if row.get("kept") is None else bool(row.get("kept")),
        "hidden": bool(row.get("hidden")),
        "created_at": row.get("created_at"), "updated_at": row.get("updated_at"),
        "error": row.get("error", ""), "duration": row.get("duration") or result.get("audio_seconds"),
        "truncated": truncated,
        "seconds": (result.get("timing") or {}).get("e2e_seconds") or result.get("seconds"),
        "ode_steps": row.get("ode_steps"), "vae": row.get("vae") or None,
        "demo": bool(row.get("demo")),
        "requested": request, "effective": row.get("effective"),
        "artifacts": row.get("artifacts") or {}, "result": result,
        "lyrics_head": _lyric_head(lyrics_source),
        "lyrics_units": _lyric_units(lyrics_source),
        "lyrics_len": len(lyrics_source),
    }
    return _with_live(payload)


def _with_live(payload: dict) -> dict:
    """Overlay in-memory stage/progress so a refresh mid-run does not lose it."""
    with LOCK:
        live = JOBS.get(payload["id"])
        if live is None or live.status not in ("running", "queued"):
            return payload
        payload["status"] = live.status
        payload["stage"] = STAGE_ZH.get(live.stage, live.stage)
        payload["stage_raw"] = live.stage
        payload["progress"] = live.progress
        if live.error:
            payload["error"] = live.error
        if live.from_stage:
            payload["from_stage"] = live.from_stage
        if live.source:
            payload["source"] = live.source
    with LIVE_LOCK:
        if LIVE_STAGE["id"] != payload["id"] or LIVE_STAGE["event"] is None:
            return payload
        payload["stage_live"] = {**LIVE_STAGE["event"],
                                 "started_at": LIVE_STAGE["started_at"],
                                 "at": LIVE_STAGE["at"], "done": _live_done()}
    return payload


def _job_event(job: "Job", **extra) -> dict:
    """job 事件附带整行（列表口径）：前端合并这一条就行，不必再拉 /api/jobs。

    调用方必须不在 LOCK 里：_with_live() 自己会拿锁，且 LOCK 不可重入。
    """
    return {"event": "job", "job": job.id, **extra, "row": _api_row(job.to_db())}


def _note_score(job_id: str | None) -> None:
    """模型刚写出乐谱：把产物刷进任务行再推一次，页面不用等整首歌结束。

    只认还在跑的任务；跑完的那次广播由 run_job 收尾时负责，别在这里插一条旧状态。
    """
    if not job_id:
        return
    with LOCK:
        job = JOBS.get(job_id)
    if job is None or job.status not in ("queued", "running"):
        return
    job.artifacts = index_artifacts(job.id)
    log.info("乐谱已写出（任务仍在跑）id=%s：%d 个产物", job.id, len(job.artifacts))
    _persist(job)
    broadcast(_job_event(job, status=job.status))


@app.get("/api/jobs")
def list_jobs(status: str | None = None, q: str | None = None, search: str | None = None,
              filter: str | None = None, favorite: bool = False,
              limit: int = Query(200, ge=1, le=500),
              offset: int = Query(0, ge=0)) -> dict:
    # 兼容别名：search= 按 q 处理；filter=failed 等价于 status=failed（前端曲库用的就是这两个词）。
    q = q or search
    if filter and not status:
        status = None if filter == "all" else filter
    rows = db.list_jobs(status=status, q=q, favorite=favorite or None, limit=limit, offset=offset)
    with LOCK:
        notices = list(NOTICES)
    return {"jobs": [_api_row(row) for row in rows],
            "total": db.count(status=status, q=q, favorite=favorite or None),
            "notices": notices}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    row = db.get(job_id)
    if row is None:
        raise HTTPException(404, "job not found")
    return _api_row(row, full=True)


@app.patch("/api/jobs/{job_id}")
def patch_job(job_id: str, body: dict) -> dict:
    row = db.get(job_id)
    if row is None:
        raise HTTPException(404, "job not found")
    updates = {}
    if "favorite" in body:
        if not isinstance(body["favorite"], bool):
            raise HTTPException(400, "favorite must be a boolean")
        updates["favorite"] = body["favorite"]
    if "title" in body:
        updates["title"] = _clean_title(body["title"])
    if "version_label" in body:
        updates["version_label"] = str(body["version_label"] or "").strip()[:60]
    if "version_note" in body:
        updates["version_note"] = str(body["version_note"] or "").strip()[:500]
    if "kept" in body:
        if not isinstance(body["kept"], bool):
            raise HTTPException(400, "kept must be a boolean")
        updates["kept"] = body["kept"]
    if "hidden" in body:
        if not isinstance(body["hidden"], bool):
            raise HTTPException(400, "hidden must be a boolean")
        updates["hidden"] = body["hidden"]
    if updates:
        db.update(job_id, **updates)
        row = db.get(job_id)
    with LOCK:
        job = JOBS.get(job_id)
        if job is not None:
            if "favorite" in updates:
                job.favorite = updates["favorite"]
            if "title" in updates:
                job.title = updates["title"]
            for key in ("version_label", "version_note"):
                if key in updates:
                    setattr(job, key, updates[key])
            if "kept" in updates:
                job.kept = updates["kept"]
            if "hidden" in updates:
                job.hidden = updates["hidden"]
    if row is not None and updates:
        broadcast({"event": "job", "job": job_id, "row": _api_row(row)})
    return {"id": job_id, **updates}


@app.patch("/api/jobs/{job_id}/work")
def rename_work(job_id: str, body: dict) -> dict:
    """改作品名：组内所有版本一条事务刷 work_name，并广播每行。"""
    row = db.get(job_id)
    if row is None:
        raise HTTPException(404, "job not found")
    if row.get("demo"):
        raise HTTPException(403, "示例曲不能重命名")
    name = _clean_title(body.get("name"))
    if not name:
        raise HTTPException(400, "作品名不能为空")
    root = row.get("work_root")
    try:
        if not root:
            # 存量行没回填过 work_root：先就地补根。直接 UPDATE ... WHERE work_root=job_id
            # 会一行都命中不到，却仍返回成功。
            log.warning("改作品名遇到缺失 work_root，就地补根 id=%s", job_id)
            db.update(job_id, work_root=job_id, work_name=name)
            root, members = job_id, [job_id]
        else:
            updated = db.bulk_set_work_name(str(root), name)
            members = db.group_member_ids(str(root))
            if updated == 0:
                log.warning("改作品名未命中组内任何行，回退单行更新 id=%s root=%s", job_id, root)
                db.update(job_id, work_root=job_id, work_name=name)
                root, members = job_id, [job_id]
    except HTTPException:
        raise
    except Exception:
        log.exception("改作品名失败 id=%s 目标名=%s", job_id, name)
        raise HTTPException(500, "改名失败，请查看服务日志") from None
    with LOCK:
        for member in members:
            live = JOBS.get(member)
            if live is not None:
                live.work_name = name
    fresh_rows = []
    for member in members:
        fresh = db.get(member)
        if fresh is None:
            continue
        payload = _api_row(fresh)
        fresh_rows.append(payload)
        with LOCK:
            live = JOBS.get(member)
        if live is not None:
            broadcast(_job_event(live))
        else:
            # 不在内存里的老版本也要推：否则其他页面只能等下一次整表刷新。
            broadcast({"event": "job", "job": member, "row": payload})
    log.info("改作品名 root=%s name=%s 影响 %d 个版本", root, name, len(fresh_rows))
    return {"work_root": root, "work_name": name, "members": members, "rows": fresh_rows}


@app.post("/api/jobs/{job_id}/move")
def move_job(job_id: str, body: dict) -> dict:
    """版本移动：并入另一作品（target_root 给任一成员 id）或拆分为独立作品（split）。"""
    row = db.get(job_id)
    if row is None:
        raise HTTPException(404, "job not found")
    if row.get("demo"):
        raise HTTPException(403, "示例任务不能移动")
    with LOCK:
        job = JOBS.get(job_id)
    split = bool(body.get("split"))
    try:
        if split:
            new_root = job_id
            # 优先用组内作品名：新格式标题是「作品名 · 版本标签」，按标题反推会把标签带进新作品名。
            new_name = row.get("work_name") or _strip_work_name(row.get("title")) or "未命名作品"
            # 拆出来的组只剩这一条：它若已完成，就是组里唯一的保留版本。
            keep = row.get("status") == "complete"
        else:
            target_id = str(body.get("target_root") or "")
            target = db.get(target_id)
            if target is None:
                raise HTTPException(404, "target job not found")
            if target.get("demo"):
                raise HTTPException(403, "不能并入示例任务")
            new_root = target.get("work_root") or target_id
            if new_root == (row.get("work_root") or job_id):
                raise HTTPException(409, "这条版本已经在该作品里")
            new_name = target.get("work_name") or "未命名作品"
            # 目标组已有 ● 就不抢；没有而这条已完成，并入后它就是组里的保留版本。
            keep = row.get("status") == "complete" and not db.group_has_kept(str(new_root))
        no = db.move_version(job_id, str(new_root), new_name, str(new_root), keep, split=split)
    except HTTPException:
        raise
    except Exception:
        log.exception("移动版本失败 id=%s split=%s", job_id, split)
        raise HTTPException(500, "移动版本失败，请查看服务日志") from None
    fields = {"work_root": new_root, "work_name": new_name, "variant_group": new_root,
              "version_no": no, "kept": True if keep else None}
    fresh = db.get(job_id)
    if job is not None:
        with LOCK:
            job.work_root = fields["work_root"]
            job.work_name = fields["work_name"]
            job.variant_group = fields["variant_group"]
            job.version_no = fields["version_no"]
            job.kept = fields.get("kept")
    if fresh is not None:
        broadcast({"event": "job", "job": job_id, "row": _api_row(fresh)})
    log.info("版本移动 id=%s root=%s no=%s split=%s keep=%s",
             job_id, new_root, no, split, keep)
    return {"id": job_id, **fields}


def _strip_work_name(title) -> str:
    """从任务标题反推可用的作品名（拆分场景）：剥派生/候选旧后缀。"""
    text = re.sub(r"^候选 \d+ · ", "", str(title or "").strip())
    text = re.sub(r" · (改谱|成歌|换解码器)(?: \d+)?$", "", text).strip()
    return text[:60]


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> dict:
    if job_id == DEMO_ID:
        raise HTTPException(409, "示例曲不能删除（重启后会自动重建）")
    directory = job_path(job_id)
    if align.busy(job_id):
        raise HTTPException(409, "正在对齐歌词，完成后再删除")
    with LOCK:
        job = JOBS.get(job_id)
        if job is not None and job.status == "running":
            raise HTTPException(409, "任务正在运行，不能删除；请先取消并等待停止")
    if job is None and db.get(job_id) is None:
        log.info("删除请求：任务不存在 id=%s", job_id)
        return {"deleted": job_id, "existed": False}
    queued_deleted = bool(job is not None and job.status == "queued")
    if queued_deleted:
        job.status = "cancelled"
        job.error = "已取消（任务已删除）"
        _persist(job)
        remaining = [item for item in list(QUEUE.queue) if item != job_id]
        QUEUE.queue.clear()
        for item in remaining:
            QUEUE.put(item)
    target, moved = None, False
    try:
        if directory.exists():
            TRASH_DIR.mkdir(parents=True, exist_ok=True)
            target = TRASH_DIR / directory.name
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
            try:
                shutil.move(str(directory), str(target))
            except OSError as exc:
                raise HTTPException(409, "文件被占用（可能正在对齐、试听或被其他程序读取），请稍后再删") from exc
            moved = True
            try:
                os.utime(target)
            except OSError as exc:
                log.debug("标记回收站时间失败 %s：%s", target.name, exc)
        try:
            db.delete(job_id)
        except Exception as exc:
            raise HTTPException(500, f"任务索引删除失败，任务未删除：{exc}") from exc
    except Exception as exc:
        if queued_deleted:
            with LOCK:
                if job is not None:
                    job.status = "queued"
                    job.error = ""
                    _persist(job)
                    queue_ids = list(QUEUE.queue)
                    QUEUE.queue.clear()
                    for item in queue_ids:
                        QUEUE.put(item)
                    if job.id not in queue_ids:
                        QUEUE.put(job.id)
            broadcast(_job_event(job, status="queued"))
        if moved and target is not None:
            try:
                shutil.move(str(target), str(directory))
                log.info("已把任务目录移回 jobs：%s", directory.name)
            except OSError as move_exc:
                log.error("回滚回收站移动失败 id=%s：%s: %s", job_id,
                          type(move_exc).__name__, move_exc)
        raise
    with LOCK:
        JOBS.pop(job_id, None)
    broadcast({"event": "jobs"})
    log.info("任务已删除 id=%s（移入回收站=%s）", job_id, moved)
    return {"deleted": job_id, "existed": True}
# ------------------------------------------------------------------- model / system
@app.get("/api/model")
def model_status() -> dict:
    return model_status_dict()


def _reassign_queued(device: str) -> list[Job]:
    """切换设备后把排队任务改到同一设备，避免它们开跑时把模型再切回去。"""
    changed = []
    with LOCK:
        for job in JOBS.values():
            if job.status != "queued" or job.request.get("device") == device:
                continue
            job.request = {**job.request, "device": device}
            _persist(job)
            changed.append(job)
    return changed


@app.post("/api/model/load")
def model_load(body: dict | None = None) -> dict:
    busy_guard("正在生成，暂不能加载或切换模型")
    body = body or {}
    device = body.get("device") or DEFAULT_DEVICE
    if device not in ("cpu", "cuda"):
        raise HTTPException(400, "device 必须是 cpu 或 cuda")
    try:
        with MODEL_LOCK:
            if not (MODEL["state"] == "loaded" and MODEL["device"] == device and worker_alive()):
                if MODEL["state"] == "loaded":
                    _unload_locked()
                device = _load_locked(device)
    except RuntimeError as exc:
        raise HTTPException(500, str(exc)) from exc
    with _GPU_LIVE_LOCK:
        _GPU_LIVE["data"] = None
        _GPU_LIVE["at"] = 0.0
    # 排队任务改到「最终真跑的那台设备」：GPU 加载失败自动降级时这里拿到的是 cpu，
    # 不会出现任务写着显卡、实际跑 CPU；加载抛错则根本走不到这一步，一条都不会改。
    changed = _reassign_queued(device) if body.get("reassign_queued") else []
    for job in changed:
        broadcast(_job_event(job, status=job.status))
    return {**model_status_dict(), "reassigned": len(changed)}


@app.post("/api/model/unload")
def model_unload() -> dict:
    busy_guard("正在生成，完成后再卸载模型")
    with MODEL_LOCK:
        if MODEL["state"] == "loaded":
            _unload_locked()
    # 卸载后显存统计约 2 秒滞后（BUG-001/016）：清 5 秒缓存并再探一次，
    # 让紧跟其后的 GET 立刻看到回升，而不是闪一下 CPU。
    with _GPU_LIVE_LOCK:
        _GPU_LIVE["data"] = None
        _GPU_LIVE["at"] = 0.0
    return model_status_dict()


# ------------------------------------------------------------------- settings
def _flash_attn_installed() -> bool:
    from importlib.util import find_spec
    try:
        return find_spec("flash_attn") is not None
    except (ImportError, ValueError):
        return False


def attention_capabilities() -> dict:
    """What each acceleration switch can really do here, without loading torch."""
    gpu = gpu_live()
    flash = _flash_attn_installed()
    capability = gpu.get("compute_capability")
    label = f"{capability:.1f}" if isinstance(capability, (int, float)) else "未知"
    widen = "，开启后改用 K/V 扩展走融合 SDPA 内核"
    nar = {"mode": "probe", "reason": "开启后由 cuDNN 自测决定，跑不赢基准就自动退回"}
    if not gpu["available"]:
        ar = {"mode": "baseline", "reason": "未检测到 NVIDIA GPU，开启后仍按基准路径运行"}
        nar = {"mode": "baseline", "reason": "未检测到 NVIDIA GPU，音频合成阶段无可加速内核"}
    elif not flash:
        ar = {"mode": "widen", "reason": f"未安装 flash_attn（算力 {label}）" + widen}
    elif capability is None:
        ar = {"mode": "probe", "reason": "无法读取显卡算力，开启后由运行期探测决定内核"}
    elif capability < 8.0:
        ar = {"mode": "widen", "reason": f"算力 {label} 不支持 FlashAttention-2（需 8.0+）" + widen}
    else:
        ar = {"mode": "flash_attn", "reason": f"FlashAttention-2 可用（算力 {label}）"}
    return {"cuda": bool(gpu["available"]), "flash_attn": flash,
            "compute_capability": gpu.get("compute_capability"), "ar": ar, "nar": nar}


@app.get("/api/settings")
def settings_read() -> dict:
    return {"settings": db.get_settings(), "capabilities": attention_capabilities()}


@app.patch("/api/settings")
def settings_update(body: dict | None = None) -> dict:
    busy_guard("正在生成，完成后再切换加速开关")
    try:
        saved = db.save_settings(body or {})
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"settings": saved, "capabilities": attention_capabilities()}


@app.post("/api/settings/reset")
def settings_reset() -> dict:
    busy_guard("正在生成，完成后再恢复默认设置")
    return {"settings": db.reset_settings(), "capabilities": attention_capabilities()}


@app.get("/api/schema")
def api_schema() -> dict:
    from yue2.protocol import CONTEXT

    def sampling_fields(defaults):
        return [
            {"key": "temperature", "min": 0, "max": 5, "step": 0.01, "default": defaults.temperature, "integer": False},
            {"key": "top_p", "min": 0.01, "max": 1, "step": 0.01, "default": defaults.top_p, "integer": False},
            {"key": "top_k", "min": 1, "max": 1000, "step": 1, "default": defaults.top_k, "integer": True},
            {"key": "repetition_penalty", "min": 0.01, "max": 5, "step": 0.005, "default": defaults.repetition_penalty, "integer": False},
            {"key": "penalty_window", "min": 1, "max": 100, "step": 1, "default": defaults.penalty_window, "integer": True},
            {"key": "min_tokens", "min": 0, "max": defaults.max_tokens, "step": 1, "default": defaults.min_tokens, "integer": True},
            {"key": "max_tokens", "min": 1, "max": 20000, "step": 1, "default": defaults.max_tokens, "integer": True},
        ]

    from yue2.protocol import GenerationConfig
    defaults = GenerationConfig()
    return {
        "types": list(JOB_TYPES),
        "cot": [
            {"value": "full", "guidance": 1.0, "editable_score": True},
            {"value": "melody", "guidance": 1.0, "editable_score": True},
            {"value": "off", "guidance": 1.01, "editable_score": False},
        ],
        "devices": ["auto", "cpu", "cuda"],
        "gpu": GPU, "default_device": DEFAULT_DEVICE,
        "seed": {"min": 0, "max": 2 ** 63 - 1},
        "cfg_scale": {"min": 0, "max": 20, "step": 0.01, "default": None},
        "ode_steps": {"min": 1, "max": 200, "step": 1, "default": defaults.ode_steps},
        "ode_method": {"value": "midpoint", "readonly": True},
        "context": {"value": CONTEXT, "readonly": True},
        "sampling": {
            "abc": {"fields": sampling_fields(defaults.abc)},
            "semantic": {"fields": sampling_fields(defaults.semantic)},
        },
        "decoders": [
            {"value": "standard", "label": "YuE2-Vae（试听）", "available": VAE_DIR.exists()},
            {"value": "legacy", "label": "YuE2-Vae-legacy（论文基准）", "available": VAE_LEGACY_DIR.exists()},
        ],
        "staged_from": ["plan", "semantic"],
        "score_tools": score_tools.available(),
    }


@app.get("/api/presets")
def api_presets() -> dict:
    return {"presets": presets}


LYRIC_MARKERS = {"[Verse]", "[Chorus]", "[Bridge]", "[Intro]", "[Outro]", "[Pre-Chorus]", "[Hook]"}


def _unusual_lyric_chars(text: str) -> list[str]:
    """Emoji/符号/控制字符：模型会编码，但不保证能唱出来，还会占用上下文。"""
    found: list[str] = []
    for char in text:
        if char in "\t\r\n":
            continue
        category = unicodedata.category(char)
        if category[0] in ("S", "C") and char not in found:
            found.append(char)
    return found


def _singable_units(text: str) -> int:
    """按字/词估算音节数：非 ASCII 字母逐字算，拉丁文按单词算。"""
    wide = sum(1 for char in text if char.isalpha() and not char.isascii())
    latin = len(re.findall(r"[A-Za-z']+", text))
    return wide + latin


@app.post("/api/lyrics/lint")
def lyrics_lint(body: dict) -> dict:
    raw = body.get("lyrics", "")
    if raw is None:
        raise HTTPException(400, "歌词为空")
    if not isinstance(raw, str):
        raise HTTPException(400, "歌词字段必须是文本")
    lyrics = raw.lstrip("\ufeff")
    if not lyrics.strip():
        raise HTTPException(400, "歌词为空")
    if len(lyrics) > _LYRICS_CHAR_CAP:
        raise HTTPException(400, f"歌词过长：超过 {_LYRICS_CHAR_CAP} 字符，请先精简")
    from yue2.protocol import CONTEXT
    tokens = _text_tokens(lyrics)
    if tokens is not None and tokens > CONTEXT:
        raise HTTPException(400, f"歌词过长：约 {tokens} 个 token，超过模型上下文 {CONTEXT}")
    lines, sections, warnings, blockers = [], [], [], []
    sung = 0
    for index, raw_line in enumerate(lyrics.splitlines()):
        text = raw_line.strip()
        if not text:
            continue
        if _PROMPT_MARKER_RE.search(text):
            blockers.append("歌词里出现了 [Tags] / [Lyrics] 提示词分区标记，请删掉")
            continue
        tag = re.fullmatch(r"\[([^\[\]]{1,40})\]", text)
        if tag:
            if text not in LYRIC_MARKERS:
                lines.append({"line": index, "text": text, "units": 0, "chars": [],
                              "kind": "tag",
                              "warning": "未知段落标记，模型只会把它当普通文本"})
                continue
            sections.append({"marker": text, "line": index})
            continue
        units = _singable_units(text)
        chars = _unusual_lyric_chars(text)
        singable = any(char.isalnum() for char in text)
        flag, kind = "", ""
        if not singable:
            flag, kind = "没有可唱的字，只会占用上下文", "empty"
        elif _FULLWIDTH_TAG_RE.search(text):
            flag, kind = "全角段落标记【】无法识别，模型会把它当普通文字；请改成半角 [Verse] / [Chorus]", "tag"
        elif chars:
            flag, kind = "含模型可能无法演唱的符号，仍会占用上下文", "chars"
        elif units and units < 5:
            flag, kind = "过短，旋律容易拖沓", "short"
        elif units > 16:
            flag, kind = "过长，单行容易唱飘", "long"
        if singable:
            sung += 1
        lines.append({"line": index, "text": text, "units": units, "chars": chars,
                      "kind": kind, "warning": flag})
    if not sung:
        blockers.append("歌词里没有可唱的字：只有段落标记、标点或符号")
    if not sections:
        warnings.append("缺少段落标记（[Verse] / [Chorus]），模型按段落组织，建议补上")
    elif not any(section["marker"] == "[Chorus]" for section in sections):
        warnings.append("没有 [Chorus]，副歌重复句式能让成品更稳")
    return {"empty": False, "ok": not blockers, "sections": sections, "lines": lines,
            "warnings": warnings, "blockers": list(dict.fromkeys(blockers))}


# ------------------------------------------------------------------- score tools
def _score_call(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except score_tools.ScoreToolsUnavailable as exc:
        raise HTTPException(503, f"官方乐谱脚本不可用：{exc}") from exc
    except ValueError as exc:
        log.info("乐谱校验未通过：%s", exc)
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/score/inspect")
def score_inspect(body: dict) -> dict:
    abc = str(body.get("abc", ""))
    if not abc.strip():
        raise HTTPException(400, "abc is empty")
    report = _score_call(score_tools.inspect, abc)
    voices = {}
    for name, voice in report.get("voices", {}).items():
        bars = voice.get("bars") or []
        keys = voice.get("keys") or []
        voices[name] = {
            "notes": voice.get("sounding_notes"),
            "measures": voice.get("measures"),
            "chords": len(voice.get("chords") or []),
            "meter": f"{bars[0][2][0]}/{bars[0][2][1]}" if bars else None,
            "key": keys[0][1] if keys else None,
        }
    vocal = voices.get("Vocal", {})
    return {
        "bpm": report.get("bpm"),
        "meter": vocal.get("meter"),
        "key": vocal.get("key"),
        "notes": vocal.get("notes"),
        "measures": vocal.get("measures"),
        "chords": vocal.get("chords"),
        "nominal_seconds": report.get("nominal_duration_seconds"),
        "voices": voices,
        "sha256": report.get("sha256"),
        "scope": report.get("scope"),
    }


@app.post("/api/score/compare")
def score_compare(body: dict) -> dict:
    before = str(body.get("before", body.get("a", "")))
    after = str(body.get("after", body.get("b", "")))
    if not before.strip() or not after.strip():
        raise HTTPException(400, "缺少 before/after（也可传 a/b）")
    return _score_call(score_tools.compare, before, after,
                       allow_tempo_change=bool(body.get("allow_tempo_change")))


@app.post("/api/score/strip-chords")
def score_strip_chords(body: dict) -> dict:
    abc = str(body.get("abc", ""))
    if not abc.strip():
        raise HTTPException(400, "abc is empty")
    return {"abc": _score_call(score_tools.strip_chords, abc, body.get("keep_voice", "both"))}


@app.post("/api/score/instrumentalize")
def score_instrumentalize(body: dict) -> dict:
    abc = str(body.get("abc", ""))
    if not abc.strip():
        raise HTTPException(400, "abc is empty")
    return _score_call(score_tools.instrumentalize, abc,
                       overlap=body.get("overlap", "vocal"),
                       keep_chords=bool(body.get("keep_chords", True)))


@app.post("/api/score/plan")
def score_plan(body: dict) -> dict:
    abc = str(body.get("abc", ""))
    if not abc.strip():
        raise HTTPException(400, "abc is empty")
    return _score_call(score_edit.plan, abc)


@app.post("/api/score/edit")
def score_edit_apply(body: dict) -> dict:
    abc = str(body.get("abc", ""))
    if not abc.strip():
        raise HTTPException(400, "abc is empty")
    ops = body.get("ops")
    return _score_call(score_edit.apply, abc, ops)


# ------------------------------------------------------------------- artifacts
@app.api_route("/api/jobs/{job_id}/artifacts/{name}", methods=["GET", "HEAD"])
def job_artifact(job_id: str, name: str) -> FileResponse:
    if name not in ARTIFACT_NAMES:
        raise HTTPException(404, "unknown artifact")
    path = job_artifacts(job_id) / name
    if not _artifact_ready(path):
        # 区分缺失与空文件（BUG-009）：文案说明是真没有产物，还是 0 字节坏文件
        if path.is_file():
            raise HTTPException(404, "产物文件为空（写入中断），请重试生成")
        raise HTTPException(404, "该任务没有这个产物（例如仅出谱任务不产生 config.json）")
    return FileResponse(path, media_type=ARTIFACT_NAMES[name], filename=f"{job_id}_{name}")


@app.api_route("/api/jobs/{job_id}/audio", methods=["GET", "HEAD"])
def job_audio(job_id: str) -> FileResponse:
    path = job_artifacts(job_id) / AUDIO_ARTIFACT
    if not _artifact_ready(path):
        raise HTTPException(404, "该任务还没有音频产物（例如仅出谱任务不产生音频）")
    return FileResponse(path, media_type="audio/flac", content_disposition_type="inline")


@app.get("/api/jobs/{job_id}/log")
def job_log(job_id: str,
            offset: int = Query(0, ge=0),
            limit: int = Query(200, ge=1, le=500),
            lines: int | None = Query(None, ge=1, le=4000)) -> JSONResponse:
    """按行号增量读日志；lines 是旧版“末尾 N 行”快捷参数，仍给旧客户端用。"""
    path = job_dir(job_id) / ".meta" / "run.log"
    if not path.exists():
        all_lines = []
        size = 0
    else:
        try:
            all_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            size = path.stat().st_size
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            log.warning("读取任务日志失败 id=%s：%s: %s", job_id, type(exc).__name__, exc)
            all_lines = []
            size = 0
    if lines is not None:
        start = max(0, len(all_lines) - lines)
        count = len(all_lines) - start
    else:
        start = min(offset, len(all_lines))
        count = min(limit, len(all_lines) - start)
    sliced = all_lines[start:start + count]
    return JSONResponse({"job": job_id, "path": str(path), "size": size,
                         "offset": start, "limit": count, "total_lines": len(all_lines),
                         "text": "\n".join(sliced), "lines": sliced},
                        headers={"Cache-Control": "no-store"})


@app.post("/api/jobs/{job_id}/reveal")
def reveal_job_dir(job_id: str) -> dict:
    directory = job_dir(job_id)
    if not directory.exists():
        raise HTTPException(404, "任务目录不存在")
    if not hasattr(os, "startfile"):
        raise HTTPException(501, "当前系统不支持直接打开目录")
    try:
        os.startfile(str(directory))
    except OSError as exc:
        log.warning("打开任务目录失败 %s：%s: %s", job_id, type(exc).__name__, exc)
        raise HTTPException(500, f"打开目录失败：{exc}；目录位置：{directory}") from exc
    return {"opened": str(directory)}


@app.get("/api/jobs/{job_id}/score")
def job_score(job_id: str) -> JSONResponse:
    path = job_artifacts(job_id) / SCORE_ARTIFACT
    if not _artifact_ready(path):
        _recover_score_from_plan(job_id)
    if not _artifact_ready(path):
        raise HTTPException(404, "该任务没有乐谱产物")
    try:
        abc = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        log.warning("乐谱不是 UTF-8 文本 %s：%s", job_id, exc)
        raise HTTPException(500, "乐谱文件不是 UTF-8 文本，无法打开；可在任务目录里查看 score.abc") from exc
    except OSError as exc:
        log.warning("读取乐谱失败 %s：%s: %s", job_id, type(exc).__name__, exc)
        raise HTTPException(500, f"读取乐谱失败：{exc}") from exc
    # 谱面会经 PUT 写回与备份恢复变化，绝不能让浏览器启发式缓存旧响应。
    digest = _score_digest(abc)
    return JSONResponse({"abc": abc, "sha256": digest},
                        headers={"Cache-Control": "no-store", "ETag": f'"{digest}"'})


# P0-2 写回改版谱面：覆盖前备份为 score.orig.abc<ts>，复用同目录文件锁。
# 前端下落 /api/jobs/{id}/score/download 时若已有备份则优先给改版，不再静默回原谱。
SCORE_BACKUP_LOCK = threading.Lock()
SCORE_BACKUP_PREFIX = "score.orig.abc"


def _score_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_score_state(job_id: str) -> dict:
    directory = job_artifacts(job_id)
    cur_path = directory / SCORE_ARTIFACT
    has_backup = any(directory.glob(SCORE_BACKUP_PREFIX + "*"))
    cur = cur_path.read_text(encoding="utf-8") if _artifact_ready(cur_path) else ""
    backup = ""
    if has_backup:
        latest = max(directory.glob(SCORE_BACKUP_PREFIX + "*"), key=lambda p: p.stat().st_mtime)
        try:
            backup = latest.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            backup = ""
    modified = bool(backup and backup != cur)
    return {"job": job_id, "sha256": _score_digest(cur) if cur else "",
            "has_backup": has_backup, "modified": modified,
            "backup_preview": backup[:200] if backup else ""}


@app.get("/api/jobs/{job_id}/score/download")
def download_job_score(job_id: str):
    path = job_artifacts(job_id) / SCORE_ARTIFACT
    if not _artifact_ready(path):
        _recover_score_from_plan(job_id)
    if not _artifact_ready(path):
        raise HTTPException(404, "该任务没有乐谱产物")
    return FileResponse(path, media_type="text/plain; charset=utf-8", filename="score.abc")


@app.api_route("/api/jobs/{job_id}/score", methods=["PUT", "POST"])
def put_job_score(job_id: str, payload: dict, request: Request) -> dict:
    if not isinstance(payload, dict) or not isinstance(payload.get("abc"), str):
        raise HTTPException(400, "缺少 abc 字段")
    abc = payload.get("abc", "")
    if not abc.strip():
        raise HTTPException(400, "谱面不能为空")
    expected = payload.get("expected_sha256")
    if not isinstance(expected, str) or not expected.strip():
        expected = request.headers.get("if-match", "").strip()
    expected = expected.strip().strip('"')
    if not expected:
        raise HTTPException(428, "写回乐谱前请先读取当前版本的 sha256")
    directory = job_artifacts(job_id)
    cur_path = directory / SCORE_ARTIFACT
    if not _artifact_ready(cur_path):
        raise HTTPException(404, "score is empty or not found")
    with SCORE_BACKUP_LOCK:
        try:
            original = cur_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as exc:
            raise HTTPException(500, f"读取乐谱失败：{exc}") from exc
        if expected != _score_digest(original):
            raise HTTPException(409, "乐谱已被其他页面修改，请刷新后再写回")
        if abc.strip() == original.strip():
            return {"status": "unchanged", "job": job_id, **_read_score_state(job_id)}
        ts = time.strftime("%Y%m%d-%H%M%S")
        backup_path = directory / f"{SCORE_BACKUP_PREFIX}-{job_id[-6:]}-{ts}"
        counter = 0
        while backup_path.exists():
            counter += 1
            backup_path = directory / f"{SCORE_BACKUP_PREFIX}-{job_id[-6:]}-{ts}-{counter}"
        try:
            backup_path.write_text(original, encoding="utf-8")
            cur_path.write_text(abc, encoding="utf-8", newline="\n")
        except OSError as exc:
            log.warning("写回谱面失败 %s：%s: %s", job_id, type(exc).__name__, exc)
            # 保留备份，避免半写丢谱
            if backup_path.exists():
                try:
                    backup_path.unlink()
                except OSError:
                    pass
            raise HTTPException(500, f"写回谱面失败：{exc}") from exc
    return {"status": "saved", "job": job_id, "backup": str(backup_path), **_read_score_state(job_id)}


@app.get("/api/jobs/{job_id}/score/state")
def get_job_score_state(job_id: str) -> JSONResponse:
    return JSONResponse(_read_score_state(job_id), headers={"Cache-Control": "no-store"})


@app.post("/api/jobs/{job_id}/score/restore")
def restore_job_score(job_id: str) -> dict:
    directory = job_artifacts(job_id)
    cur_path = directory / SCORE_ARTIFACT
    if not _artifact_ready(cur_path):
        raise HTTPException(404, "score is empty or not found")
    with SCORE_BACKUP_LOCK:
        backups = sorted(directory.glob(SCORE_BACKUP_PREFIX + "*"), key=lambda p: p.stat().st_mtime)
        if not backups:
            raise HTTPException(404, "没有可用的备份")
        latest = backups[-1]
        try:
            backup_abc = latest.read_text(encoding="utf-8")
            current = cur_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as exc:
            raise HTTPException(500, f"读取备份失败：{exc}") from exc
        ts = time.strftime("%Y%m%d-%H%M%S")
        rollback_path = directory / f"{SCORE_BACKUP_PREFIX}-rollback-{ts}"
        try:
            rollback_path.write_text(current, encoding="utf-8")
            cur_path.write_text(backup_abc, encoding="utf-8", newline="\n")
        except OSError as exc:
            raise HTTPException(500, f"恢复原谱失败：{exc}") from exc
    return {"status": "restored", "job": job_id, "from": str(latest), **_read_score_state(job_id)}


@app.get("/api/jobs/{job_id}/lyrics")
def job_lyrics(job_id: str) -> JSONResponse:
    directory = job_artifacts(job_id)
    lyrics_path = directory / "lyrics.json"
    payload = _read_json(lyrics_path)
    if payload is not None:
        if not isinstance(payload, dict):
            return JSONResponse({"status": "failed",
                                 "reason": "歌词时间轴文件格式异常（不是对象），可以点「重新对齐」重试"})
        return JSONResponse(payload)
    if lyrics_path.exists():
        return JSONResponse({"status": "failed",
                             "reason": "歌词时间轴文件损坏，无法解析；可以点「重新对齐」重试"})
    error = _read_json(directory / "lyrics.error.json")
    if error:
        return JSONResponse({"status": "failed", "reason": error.get("reason", "")})
    job = find_job(job_id)
    if job is None:
        if directory.exists():
            return JSONResponse({"status": "unavailable"})
        raise HTTPException(404, "job not found")
    if job.status != "complete":
        return JSONResponse({"status": "idle", "reason": "任务未完成"})
    if align.find_audio(directory) is None:
        return JSONResponse({"status": "idle", "reason": "没有音频产物"})
    if not _job_lyrics(job).strip():
        return JSONResponse({"status": "idle", "reason": "没有歌词"})
    schedule_alignment(job)
    return JSONResponse({"status": "pending"})


@app.get("/api/jobs/{job_id}/lyrics/subtitle")
def job_lyrics_subtitle(job_id: str, format: str = Query("srt"),
                        enhanced: bool = Query(False)) -> Response:
    key = (format or "").strip().lower()
    if key not in subtitles.FORMATS:
        raise HTTPException(400, "字幕格式只能是 srt 或 lrc")
    if key == "srt" and enhanced:
        raise HTTPException(400, "SRT 不支持逐字时间轴，请改用 LRC（逐字）")
    directory = job_artifacts(job_id)
    if not (directory / "lyrics.json").exists():
        error = _read_json(directory / "lyrics.error.json")
        if error and error.get("reason"):
            raise HTTPException(409, f"歌词对齐失败：{error['reason']}")
        job = find_job(job_id)
        if job is None:
            if directory.exists():
                raise HTTPException(409, "歌词时间轴还没生成，等对齐完成后再导出")
            raise HTTPException(404, "job not found")
        if job.status != "complete":
            raise HTTPException(409, "任务未完成，暂无字幕可导出")
        if align.find_audio(directory) is None:
            raise HTTPException(409, "没有音频产物，无法生成字幕")
        if not _job_lyrics(job).strip():
            raise HTTPException(409, "没有歌词，无法生成字幕")
        schedule_alignment(job)
        raise HTTPException(409, "歌词时间轴正在生成，请稍后再导出")
    try:
        text = subtitles.read_export(directory, key, enhanced)
    except subtitles.SubtitleError as exc:
        raise HTTPException(409, str(exc)) from exc
    job = find_job(job_id)
    name = subtitles.download_name(job, job_id, key, enhanced)
    media = ("application/x-subrip; charset=utf-8" if key == "srt"
             else "text/plain; charset=utf-8")
    return Response(content=text, media_type=media, headers={
        "Content-Disposition": subtitles.content_disposition(name),
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff",
    })


@app.post("/api/jobs/{job_id}/lyrics/retry")
def retry_lyrics(job_id: str) -> dict:
    job = find_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if job.status != "complete":
        raise HTTPException(409, "任务未完成，无法对齐歌词")
    directory = job_artifacts(job_id)
    if align.find_audio(directory) is None or not _job_lyrics(job).strip():
        raise HTTPException(409, "没有可用于对齐的音频或歌词")
    # 对齐线程还在跑这一首时，删掉它的产物再排队只会两头空：新任务被去重吞掉，
    # 旧任务又没有可写的结果。等它自己写完/写错，用户再点重试即可。
    if align.busy(job_id):
        return {"status": "pending"}
    for name in ("lyrics.error.json", "lyrics.json"):
        path = directory / name
        if path.exists():
            path.unlink()
    schedule_alignment(job)
    return {"status": "pending"}


# ------------------------------------------------------------------- update
def _version_tuple(value: str) -> tuple:
    parts = []
    for chunk in str(value or "").split("."):
        num = ""
        for ch in chunk:
            if ch.isdigit():
                num += ch
            else:
                break
        parts.append(int(num) if num else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


def current_version(refresh: bool = False) -> dict:
    """版本身份 + 更新源配置。默认带 30s 缓存；检查更新时强制读盘。"""
    now = time.time()
    if not refresh and VERSION_CACHE["data"] and now - VERSION_CACHE["at"] < 30:
        return VERSION_CACHE["data"]
    info = _read_json(ROOT / "version.json") or {}
    config = _read_json(ROOT / "update.json") or {}
    result = {
        "version": str(info.get("version") or "dev"),
        "commit": str(info.get("commit") or ""),
        "built_at": str(info.get("built_at") or ""),
        "update_url": str(config.get("update_url") or ""),
        "channel": str(config.get("channel") or "stable"),
        "auto_check": bool(config.get("auto_check", False)),
    }
    VERSION_CACHE["data"] = result
    VERSION_CACHE["at"] = now
    return result


@app.get("/api/system/version")
def system_version() -> dict:
    return current_version()


@app.post("/api/system/update/check")
def update_check() -> dict:
    info = current_version(refresh=True)
    url = info["update_url"]
    if not url:
        raise HTTPException(400, "没有配置更新源（update.json 的 update_url）")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "YuE2-Updater/1.0"})
        with urllib.request.urlopen(request, timeout=20) as response:
            manifest = json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - 网络/格式问题统一回报
        raise HTTPException(502, f"检查更新失败：{exc}")
    latest = str(manifest.get("version") or "")
    if not latest:
        raise HTTPException(502, "更新清单缺少版本号")
    current = info["version"]
    has_update = _version_tuple(latest) > _version_tuple(current)
    min_version = str(manifest.get("min_version") or "")
    blocked = bool(min_version) and _version_tuple(current) < _version_tuple(min_version)
    pkg_url = str(manifest.get("url") or "")
    pkg_sha = str(manifest.get("sha256") or "").lower()
    if not pkg_url:
        raise HTTPException(502, "更新清单缺少下载地址")
    if not pkg_sha:
        raise HTTPException(502, "更新清单缺少 sha256，拒绝下载")
    try:
        pkg_size = int(manifest.get("size") or 0)
    except (TypeError, ValueError):
        pkg_size = 0
    global UPDATE_PENDING
    UPDATE_PENDING = {"url": pkg_url, "sha256": pkg_sha, "size": pkg_size, "version": latest}
    return {
        "current": current,
        "latest": latest,
        "has_update": has_update,
        "url": pkg_url,
        "sha256": pkg_sha,
        "size": pkg_size,
        "changelog": str(manifest.get("changelog") or ""),
        "published_at": str(manifest.get("published_at") or ""),
        "min_version": min_version,
        "blocked": blocked,
        "reason": (f"当前版本 {current} 低于该更新要求的最低版本 {min_version}，"
                   f"请下载完整整合包覆盖安装" if blocked else ""),
    }


@app.post("/api/system/update/download")
def update_download(payload: dict) -> dict:
    url = str(payload.get("url") or "")
    sha256 = str(payload.get("sha256") or "").lower()
    pending = dict(UPDATE_PENDING)
    if not pending:
        raise HTTPException(400, "请先检查更新")
    # 只允许下载服务端刚检查出的那个包，避免客户端把下载地址换成任意 URL
    if url != pending.get("url") or sha256 != pending.get("sha256"):
        raise HTTPException(400, "下载信息与检查结果不一致，请重新检查更新")
    if not sha256:
        raise HTTPException(400, "缺少 sha256")
    with DOWNLOAD_LOCK:
        with UPDATE_LOCK:
            if UPDATE_STATE["phase"] not in ("idle", "downloaded"):
                raise HTTPException(409, f"更新正在进行：{UPDATE_STATE['phase']}")
            UPDATE_STATE.update(phase="downloading", detail="正在下载更新包", at=time.time())
        UPDATE_DIR.mkdir(parents=True, exist_ok=True)
        target = UPDATE_DIR / "update.zip"
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "YuE2-Updater/1.0"})
            with urllib.request.urlopen(request, timeout=UPDATE_TIMEOUT) as response, \
                    target.open("wb") as out:
                shutil.copyfileobj(response, out, length=256 * 1024)
        except Exception as exc:  # noqa: BLE001
            target.unlink(missing_ok=True)
            with UPDATE_LOCK:
                UPDATE_STATE.update(phase="idle", detail=f"下载失败：{exc}", at=time.time())
            raise HTTPException(502, f"下载失败：{exc}")
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if digest != sha256:
            target.unlink(missing_ok=True)
            with UPDATE_LOCK:
                UPDATE_STATE.update(phase="idle", detail="校验失败，已删除更新包", at=time.time())
            raise HTTPException(400, f"校验和不匹配：{digest}")
        expected = int(pending.get("size") or 0)
        actual = target.stat().st_size
        if expected and actual != expected:
            target.unlink(missing_ok=True)
            with UPDATE_LOCK:
                UPDATE_STATE.update(phase="idle", detail="文件大小与清单不一致", at=time.time())
            raise HTTPException(400, f"文件大小与清单不一致：{actual} != {expected}")
        with UPDATE_LOCK:
            UPDATE_STATE.update(phase="downloaded", detail="下载并校验完成", at=time.time())
    return {"path": str(target), "sha256": digest, "size": actual}


@app.post("/api/system/update/apply")
def update_apply(payload: dict | None = None) -> dict:
    payload = payload or {}
    zip_path = UPDATE_DIR / "update.zip"
    if not zip_path.is_file():
        raise HTTPException(400, "还没有下载更新包")
    if MODEL_BUSY or QUEUE.qsize() > 0:
        raise HTTPException(409, "有任务正在生成，等它完成后再更新")
    port = SERVER_PORT  # 固定用服务自身端口，忽略客户端传值
    updater = ROOT / "app" / "updater.py"
    if not updater.is_file():
        raise HTTPException(400, "更新程序缺失：app/updater.py")
    with UPDATE_LOCK:
        UPDATE_STATE.update(phase="applying", detail="正在安装更新", at=time.time())
    try:
        # cmd /c start 让 updater 成为独立进程，stop.bat 的 taskkill /T 不会带走它
        subprocess.Popen(
            ["cmd.exe", "/c", "start", "", "/b", str(PYTHON_EXE), str(updater),
             "--zip", str(zip_path), "--port", str(port), "--timeout", "150"],
            cwd=str(ROOT), close_fds=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        with UPDATE_LOCK:
            UPDATE_STATE.update(phase="idle", detail=f"启动更新失败：{exc}", at=time.time())
        raise HTTPException(500, f"启动更新失败：{exc}")
    return {"status": "updating", "port": port, "detail": "更新中，服务将自动重启"}


@app.get("/api/system/update/status")
def update_status() -> dict:
    data = _read_json(UPDATE_DIR / "status.json") or {}
    with UPDATE_LOCK:
        if not data or UPDATE_STATE["at"] > data.get("at", 0):
            return {"phase": UPDATE_STATE["phase"], "detail": UPDATE_STATE["detail"],
                    "at": UPDATE_STATE["at"], "ok": False,
                    "version": current_version()["version"]}
    data = dict(data)
    data.setdefault("version", current_version()["version"])
    return data


# ------------------------------------------------------------------- system
@app.get("/api/system/doctor")
def system_doctor(refresh: bool = False) -> dict:
    now = time.time()
    if not refresh and DOCTOR_CACHE["data"] and now - DOCTOR_CACHE["at"] < 60:
        return DOCTOR_CACHE["data"]
    env = {**os.environ, "YUE2_KIT": str(KIT), "PYTHONUTF8": "1"}
    try:
        run = subprocess.run([str(PYTHON_EXE), "-m", "yue2.cli", "doctor"],
                             cwd=str(ROOT), capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=120, env=env,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        data = json.loads(run.stdout) if run.stdout.strip().startswith("{") else {
            "dependencies_ready": False, "error": (run.stderr or run.stdout or "doctor failed")[-800:]}
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        data = {"dependencies_ready": False, "error": f"{type(exc).__name__}: {exc}"}
    DOCTOR_CACHE["at"], DOCTOR_CACHE["data"] = now, data
    return data


@app.get("/api/health")
def health() -> JSONResponse:
    """部署探活：只读内存队列与 SQLite 计数，不加载模型、不扫描磁盘、不跑 doctor。"""
    try:
        jobs = db.count()
    except Exception as exc:  # noqa: BLE001 - 探活只报状态，不抛栈
        log.warning("健康检查数据库不可达：%s", exc)
        jobs = None
    with SUB_LOCK:
        subscribers = len(SUBSCRIBERS)
    ok = jobs is not None
    return JSONResponse(
        status_code=200 if ok else 503,
        content={"status": "ok" if ok else "degraded",
                 "version": current_version()["version"],
                 "jobs": jobs, "queued": QUEUE.qsize(),
                 "subscribers": subscribers, "model_busy": MODEL_BUSY,
                 "time": time.time()})


@app.get("/api/events")


@app.get("/api/events")
def events(last_event_id: str | None = Header(default=None, alias="Last-Event-ID")) -> StreamingResponse:
    sub: queue.Queue = queue.Queue(maxsize=2000)
    after: int | None = None
    if last_event_id is not None:
        try:
            after = int(last_event_id)
        except (TypeError, ValueError):
            after = None
    replay: list[dict] = []
    with SUB_LOCK:
        SUBSCRIBERS.add(sub)
    with EVENT_SEQ_LOCK:
        # 新订阅从这里开始；上次连接时的序号用于发现重连期间的空档。
        connected_seq = EVENT_SEQ["value"]
        # 与 seq 同锁取快照：避免重放集里混入已经进队的新事件造成重复投递。
        if after is not None and after < connected_seq:
            replay = [item for item in EVENT_HISTORY if item["seq"] > after]

    def stream():
        try:
            # 先补发断开期间漏掉的事件（浏览器重连会带 Last-Event-ID），再声明基线。
            for item in replay:
                yield f"id: {item['seq']}\ndata: {json.dumps(item, ensure_ascii=False)}\n\n"
            yield f"event: ready\ndata: {json.dumps({'seq': connected_seq})}\n\n"
            while True:
                try:
                    event = sub.get(timeout=15)
                except queue.Empty:
                    yield ": keepalive\n\n"
                    continue
                yield f"id: {event['seq']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
        finally:
            with SUB_LOCK:
                SUBSCRIBERS.discard(sub)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ------------------------------------------------------------------- app lifecycle
@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "app" / "static" / "index.html")


@app.get("/legacy")
def legacy_index() -> FileResponse:
    return FileResponse(ROOT / "app" / "static" / "index.legacy.html")
