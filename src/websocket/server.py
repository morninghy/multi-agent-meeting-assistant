"""
WebSocket 服务器 - 实时音频流接入和结果推送

支持两种模式:
1. 实时模式: 客户端通过 WebSocket 发送音频流，服务端实时返回转写结果
2. 文件模式: 通过 REST API 上传音频文件，异步处理后推送结果
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import (
    BackgroundTasks,
    FastAPI,
    HTTPException,
    Query,
    UploadFile,
    File,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel

from ..graph.meeting_graph import run_meeting_pipeline
from ..models.schemas import MeetingStatus


app = FastAPI(
    title="多Agent智能会议助手",
    description="企业级5-Agent会议全流程自动化系统",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 存储活跃的 WebSocket 连接和会议状态
active_connections: dict[str, WebSocket] = {}
meeting_results: dict[str, dict] = {}
meeting_metadata: dict[str, dict] = {}

# 支持的音频格式
SUPPORTED_AUDIO_FORMATS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
MAX_FILE_SIZE_BYTES = 500 * 1024 * 1024  # 500MB


# ============================================================
# 请求/响应模型
# ============================================================


class MeetingStartRequest(BaseModel):
    title: str = ""
    participants: list[str] = []
    language: str = "zh"


class ErrorResponse(BaseModel):
    detail: str
    error_code: str = ""
    meeting_id: str = ""
    timestamp: str = ""
    recoverable: bool = False


# ============================================================
# 错误处理
# ============================================================


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _raise_not_found(meeting_id: str) -> None:
    raise HTTPException(
        status_code=404,
        detail=f"Meeting not found: {meeting_id}",
    )


def _raise_bad_request(detail: str, error_code: str = "BAD_REQUEST") -> None:
    raise HTTPException(status_code=400, detail=detail)


# ============================================================
# WebSocket 端点
# ============================================================


@app.websocket("/ws/meeting/{meeting_id}")
async def websocket_meeting(websocket: WebSocket, meeting_id: str):
    """
    WebSocket 会议端点

    协议:
    - 客户端发送: 音频二进制帧 / JSON控制消息
    - 服务端返回: JSON格式的处理结果

    控制消息:
    - {"type": "start"}: 开始录制
    - {"type": "stop"}: 停止录制，触发Pipeline处理
    - {"type": "demo"}: 运行演示模式
    - {"type": "ping"}: 心跳
    """
    await websocket.accept()
    active_connections[meeting_id] = websocket
    audio_buffer = bytearray()
    recording_start_time: float | None = None
    pipeline_start_time: float | None = None

    logger.info(f"WebSocket connected: {meeting_id}")

    try:
        await websocket.send_json({
            "type": "connected",
            "meeting_id": meeting_id,
            "server_time": _now_iso(),
            "message": "会议助手已连接，发送音频数据开始录制",
        })

        while True:
            data = await websocket.receive()

            if "bytes" in data and data["bytes"]:
                audio_buffer.extend(data["bytes"])
                if recording_start_time is None:
                    recording_start_time = time.time()
                duration_s = time.time() - recording_start_time
                await websocket.send_json({
                    "type": "recording",
                    "buffer_size": len(audio_buffer),
                    "duration_s": round(duration_s, 1),
                })

            elif "text" in data and data["text"]:
                message = json.loads(data["text"])
                msg_type = message.get("type", "")

                if msg_type == "start":
                    recording_start_time = time.time()
                    audio_buffer.clear()
                    await websocket.send_json({
                        "type": "recording",
                        "buffer_size": 0,
                        "duration_s": 0.0,
                    })

                elif msg_type == "stop":
                    pipeline_start_time = time.time()
                    await websocket.send_json({
                        "type": "processing",
                        "stage": "transcription",
                        "message": "正在处理音频，请稍候...",
                        "progress": 0.0,
                    })

                    result = await run_meeting_pipeline(
                        meeting_id=meeting_id,
                        audio_data=bytes(audio_buffer),
                    )
                    meeting_results[meeting_id] = result

                    processing_time = time.time() - pipeline_start_time
                    await _send_results(websocket, result, processing_time)
                    audio_buffer.clear()
                    recording_start_time = None

                elif msg_type == "demo":
                    pipeline_start_time = time.time()
                    await websocket.send_json({
                        "type": "processing",
                        "stage": "transcription",
                        "message": "运行演示模式...",
                        "progress": 0.0,
                    })
                    result = await run_meeting_pipeline(
                        meeting_id=meeting_id,
                        audio_data=b"",
                    )
                    meeting_results[meeting_id] = result
                    processing_time = time.time() - pipeline_start_time
                    await _send_results(websocket, result, processing_time)

                elif msg_type == "ping":
                    await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {meeting_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {meeting_id} - {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "error": "INTERNAL_ERROR",
                "message": str(e),
                "recoverable": True,
            })
        except Exception:
            pass
    finally:
        active_connections.pop(meeting_id, None)


async def _send_results(
    websocket: WebSocket,
    state: dict,
    processing_time_s: float = 0.0,
):
    """将 Pipeline 处理结果分步推送给客户端"""
    # 转写结果 — 按文档格式逐段发送
    transcript = state.get("transcript")
    if transcript and hasattr(transcript, "segments"):
        for seg in transcript.segments:
            await websocket.send_json({
                "type": "transcript",
                "data": {
                    "speaker": seg.speaker,
                    "text": seg.text,
                    "timestamp": seg.start,
                    "is_final": True,
                    "confidence": seg.confidence,
                },
            })

    # 摘要结果
    summary = state.get("summary")
    if summary:
        await websocket.send_json({
            "type": "summary",
            "data": summary.model_dump() if hasattr(summary, "model_dump") else {},
        })

    # 待办结果
    actions = state.get("actions")
    if actions:
        await websocket.send_json({
            "type": "actions",
            "data": actions.model_dump() if hasattr(actions, "model_dump") else {},
        })

    # 洞察结果
    insights = state.get("insights")
    if insights:
        await websocket.send_json({
            "type": "insights",
            "data": insights.model_dump() if hasattr(insights, "model_dump") else {},
        })

    # 跟进结果
    followup = state.get("followup")
    if followup:
        await websocket.send_json({
            "type": "followup",
            "data": followup.model_dump() if hasattr(followup, "model_dump") else {},
        })

    # 完成通知
    errors = state.get("errors", [])
    await websocket.send_json({
        "type": "completed",
        "meeting_id": state.get("meeting_id"),
        "processing_time_s": round(processing_time_s, 1),
    })


# ============================================================
# REST API 端点
# ============================================================


@app.get("/")
async def root():
    """健康检查 — 返回系统状态和各 Agent 就绪信息"""
    return {
        "name": "多Agent智能会议助手",
        "version": "1.0.0",
        "status": "healthy",
        "agents": {
            "transcription": "ready",
            "summary": "ready",
            "action": "ready",
            "insight": "ready",
            "followup": "ready",
        },
        "docs": "/docs",
        "websocket": "ws://localhost:8000/ws/meeting/{meeting_id}",
    }


@app.post("/api/v1/meeting/start", status_code=201)
async def start_meeting(request: MeetingStartRequest):
    """创建新会议"""
    if request.language not in ("zh", "en"):
        _raise_bad_request(
            f"Invalid language: {request.language}, supported: zh, en",
            "INVALID_LANGUAGE",
        )

    meeting_id = f"m-{uuid.uuid4().hex[:12]}"
    now = _now_iso()

    meeting_metadata[meeting_id] = {
        "title": request.title,
        "participants": request.participants,
        "language": request.language,
        "created_at": now,
    }

    return {
        "meeting_id": meeting_id,
        "websocket_url": f"ws://localhost:8000/ws/meeting/{meeting_id}",
        "status": "created",
        "created_at": now,
    }


async def _process_upload_background(
    meeting_id: str,
    audio_data: bytes,
    title: str = "",
    participants: list[str] | None = None,
    language: str = "zh",
):
    """后台异步处理上传的音频文件"""
    try:
        result = await run_meeting_pipeline(
            meeting_id=meeting_id,
            audio_data=audio_data,
            title=title,
            participants=participants,
            language=language,
        )
        meeting_results[meeting_id] = result
        logger.info(f"Background processing complete: {meeting_id}")
    except Exception as e:
        logger.error(f"Background processing failed: {meeting_id} - {e}")


@app.post("/api/v1/meeting/{meeting_id}/upload", status_code=202)
async def upload_audio(
    meeting_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    language: str | None = None,
):
    """上传音频文件并异步处理"""
    # 格式校验
    if file.filename:
        ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
        if ext not in SUPPORTED_AUDIO_FORMATS:
            _raise_bad_request(
                f"Unsupported file format: {ext}. Supported: {', '.join(sorted(SUPPORTED_AUDIO_FORMATS))}",
                "INVALID_AUDIO_FORMAT",
            )

    audio_data = await file.read()

    # 大小校验
    if len(audio_data) > MAX_FILE_SIZE_BYTES:
        size_mb = len(audio_data) / (1024 * 1024)
        _raise_bad_request(
            f"File size {size_mb:.0f}MB exceeds maximum allowed size 500MB",
            "FILE_TOO_LARGE",
        )

    logger.info(
        f"Received audio upload: {meeting_id}, size={len(audio_data)} bytes"
    )

    meta = meeting_metadata.get(meeting_id, {})
    lang = language or meta.get("language", "zh")

    # 异步后台处理
    background_tasks.add_task(
        _process_upload_background,
        meeting_id=meeting_id,
        audio_data=audio_data,
        title=meta.get("title", ""),
        participants=meta.get("participants"),
        language=lang,
    )

    return {
        "meeting_id": meeting_id,
        "status": "processing",
        "file_name": file.filename,
        "file_size_bytes": len(audio_data),
        "message": "音频已接收，正在处理中。通过 WebSocket 获取实时进度。",
    }


@app.post("/api/v1/meeting/{meeting_id}/demo")
async def run_demo(meeting_id: str = "demo"):
    """运行演示模式（无需音频）"""
    pipeline_start = time.time()
    result = await run_meeting_pipeline(
        meeting_id=meeting_id,
        audio_data=b"",
    )
    meeting_results[meeting_id] = result
    processing_time = time.time() - pipeline_start

    response: dict[str, Any] = {
        "meeting_id": meeting_id,
        "status": result.get("status", "completed"),
        "processing_time_s": round(processing_time, 1),
    }

    for key in ("transcript", "summary", "actions", "insights", "followup"):
        val = result.get(key)
        if val and hasattr(val, "model_dump"):
            response[key] = val.model_dump()

    response["errors"] = result.get("errors", [])
    return response


@app.get("/api/v1/meeting/{meeting_id}/transcript")
async def get_transcript(meeting_id: str):
    """获取转写结果"""
    result = meeting_results.get(meeting_id)
    if not result:
        _raise_not_found(meeting_id)
    transcript = result.get("transcript")
    if transcript and hasattr(transcript, "model_dump"):
        return transcript.model_dump()
    return {"error": "Transcript not available"}


@app.get("/api/v1/meeting/{meeting_id}/summary")
async def get_summary(meeting_id: str):
    """获取会议纪要"""
    result = meeting_results.get(meeting_id)
    if not result:
        _raise_not_found(meeting_id)
    summary = result.get("summary")
    if summary and hasattr(summary, "model_dump"):
        return summary.model_dump()
    return {"error": "Summary not available"}


@app.get("/api/v1/meeting/{meeting_id}/actions")
async def get_actions(
    meeting_id: str,
    status: str | None = Query(None, description="过滤状态: pending/in_progress/completed/all"),
    assignee: str | None = Query(None, description="过滤负责人"),
):
    """获取待办事项"""
    result = meeting_results.get(meeting_id)
    if not result:
        _raise_not_found(meeting_id)
    actions = result.get("actions")
    if actions and hasattr(actions, "model_dump"):
        return actions.model_dump()
    return {"error": "Actions not available"}


@app.get("/api/v1/meeting/{meeting_id}/insights")
async def get_insights(meeting_id: str):
    """获取会议洞察"""
    result = meeting_results.get(meeting_id)
    if not result:
        _raise_not_found(meeting_id)
    insights = result.get("insights")
    if insights and hasattr(insights, "model_dump"):
        return insights.model_dump()
    return {"error": "Insights not available"}


@app.get("/api/v1/meeting/{meeting_id}/report")
async def get_full_report(meeting_id: str):
    """获取完整报告"""
    result = meeting_results.get(meeting_id)
    if not result:
        _raise_not_found(meeting_id)

    response = {"meeting_id": meeting_id}
    for key in ("transcript", "summary", "actions", "insights", "followup"):
        val = result.get(key)
        if val and hasattr(val, "model_dump"):
            response[key] = val.model_dump()

    response["errors"] = result.get("errors", [])
    return response
