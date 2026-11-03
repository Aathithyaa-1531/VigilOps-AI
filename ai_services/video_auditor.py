"""
VigilOps AI - High-Performance Parallel Video Audit Engine
Optimized for Apple Silicon (Unified Memory, MPS Acceleration)

Architecture:
- Automatic 30-second video chunking with configurable overlap.
- Parallel CPU video decoding (2 threads) with bounded backpressure queue.
- Centralized batch inference on Apple Silicon MPS device (single YOLOv8 instance).
- Cross-chunk ByteTrack tracking ID reconciliation.
- Continuous progress reporting (FPS, chunks, ETA, memory status, detected hazards).
- Deduplication of continuous incidents across chunk boundaries.
"""

import os
import time
import math
import queue
import threading
import concurrent.futures
from typing import Dict, Any, List, Optional, Callable
import cv2
import numpy as np
import torch
import httpx
from ultralytics import YOLO

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from risk_engine import risk_engine
from agent_brain import run_agent_reasoning
from main import evaluate_ppe_for_person

VIGILOPS_CONFIG = {
    "model": os.path.join(os.path.dirname(__file__), "models", "yolov8n_ppe.pt"),
    "fallback_model": os.path.join(os.path.dirname(__file__), "yolov8n.pt"),
    "device": "mps" if torch.backends.mps.is_available() else "cpu",
    "image_size": 640,
    "chunk_duration": 30.0,       # 30-second chunking
    "decode_workers": 2,          # 2 parallel CPU decoding threads
    "inference_workers": 1,       # 1 centralized MPS inference thread
    "batch_size": 2,              # 2 frames batched per inference step
    "frame_stride": 6,            # 1 frame every 6 frames (~5 FPS for 30 FPS video)
    "max_decoded_chunks": 2,      # Backpressure queue limit
    "max_queue_batches": 2,
    "tracker": "bytetrack.yaml",
    "overlap_seconds": 2.0,       # 2s overlap for boundary tracking continuity
    "memory_optimization": True,
    "fallback_device": "cpu",
    "node_gateway_url": "http://localhost:5000"
}

def compute_box_iou(boxA: List[float], boxB: List[float]) -> float:
    """Calculates Intersection-over-Union (IoU) between two normalized [x1, y1, x2, y2] bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union_area = areaA + areaB - inter_area
    return float(inter_area / max(1e-6, union_area))

class VideoAuditor:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = dict(VIGILOPS_CONFIG)
        if config:
            self.config.update(config)

        self.device = self.config["device"]
        self.model = self._load_central_model()
        self.fallback_model = self._load_fallback_model()
        self.is_running = False
        self.stop_requested = False

    def _load_central_model(self) -> YOLO:
        """Loads a single centralized YOLO instance on MPS with CPU fallback."""
        model_path = self.config["model"]
        if not os.path.exists(model_path):
            model_path = self.config["fallback_model"]

        try:
            print(f"⚡ [VideoAuditor] Loading centralized YOLO model ({model_path}) on {self.device}...")
            model = YOLO(model_path)
            # Try moving model to MPS
            if self.device == "mps":
                try:
                    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                    model(dummy, device="mps", verbose=False)
                    print(f"✅ [VideoAuditor] Centralized YOLO model initialized on Apple Silicon MPS.")
                except Exception as mps_err:
                    print(f"⚠️ [VideoAuditor] MPS initialization warning ({mps_err}). Falling back to CPU.")
                    self.device = "cpu"
            return model
        except Exception as e:
            print(f"❌ [VideoAuditor] Error loading model: {e}. Falling back to default CPU.")
            self.device = "cpu"
            return YOLO(self.config["fallback_model"])

    def _load_fallback_model(self) -> Optional[YOLO]:
        """Loads fallback COCO model for person detection if custom model has lower recall on wild video."""
        fb_path = self.config["fallback_model"]
        if os.path.exists(fb_path):
            try:
                return YOLO(fb_path)
            except Exception:
                return None
        return None

    @staticmethod
    def get_video_metadata(video_path: str) -> Dict[str, Any]:
        """Extracts complete resolution, FPS, total frames, and duration from video file."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
        duration_sec = round(frame_count / fps, 2) if fps > 0 else 0.0
        cap.release()

        return {
            "path": video_path,
            "filename": os.path.basename(video_path),
            "width": width,
            "height": height,
            "fps": round(float(fps), 2),
            "frame_count": frame_count,
            "duration_sec": duration_sec,
            "duration_formatted": f"{int(duration_sec // 60):02d}:{int(duration_sec % 60):02d}"
        }

    def create_chunks(self, duration_sec: float) -> List[Dict[str, Any]]:
        """Divides video into 30-second segments with 2-second overlap."""
        chunk_dur = float(self.config["chunk_duration"])
        overlap = float(self.config["overlap_seconds"])
        chunks = []

        curr_start = 0.0
        chunk_id = 0

        while curr_start < duration_sec:
            curr_end = min(curr_start + chunk_dur, duration_sec)
            chunks.append({
                "chunk_id": chunk_id,
                "start_sec": round(curr_start, 2),
                "end_sec": round(curr_end, 2),
                "duration": round(curr_end - curr_start, 2),
                "status": "PENDING"
            })
            chunk_id += 1
            if curr_end >= duration_sec:
                break
            curr_start = max(curr_start + chunk_dur - overlap, curr_start + 1.0)

        return chunks

    def audit_video_parallel(
        self,
        video_path: str,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        source_id: str = "Video Audit",
        inference_target: str = "local_m2"
    ) -> Dict[str, Any]:
        """
        Executes high-performance parallel chunking and centralized MPS/Cloud GPU inference.
        Returns aggregated incident history, metrics, and benchmarks.
        """
        start_time = time.time()
        self.is_running = True
        self.stop_requested = False

        from cloud_gpu_adapter import cloud_gpu_adapter
        use_cloud = inference_target == "cloud_gpu" and cloud_gpu_adapter.is_configured()
        active_device = "CLOUD_GPU" if use_cloud else self.device.upper()
        if inference_target == "cloud_gpu" and not use_cloud:
            print(f"ℹ️ [VideoAuditor] Cloud GPU target requested but remote endpoint is unconfigured. Operating via Local Processing ({self.device.upper()}).")

        meta = self.get_video_metadata(video_path)
        chunks = self.create_chunks(meta["duration_sec"])
        total_chunks = len(chunks)

        print(f"🚀 [VideoAuditor] Starting Parallel Audit: '{meta['filename']}' ({meta['duration_formatted']}, {total_chunks} Chunks)")

        frame_stride = int(self.config["frame_stride"])
        batch_size = int(self.config["batch_size"])
        decode_workers = int(self.config["decode_workers"])

        # Per-chunk thread-safe queues preserving chronological ordering with bounded memory backpressure
        chunk_queues: Dict[int, queue.Queue] = {chunk["chunk_id"]: queue.Queue(maxsize=12) for chunk in chunks}

        # Tracking state and results container
        all_detections: List[Dict[str, Any]] = []
        confirmed_incidents: List[Dict[str, Any]] = []
        incident_track_map: Dict[int, Dict[str, Any]] = {}
        active_hazard_windows: Dict[str, Dict[str, Any]] = {}
        global_track_map: Dict[int, int] = {}
        last_chunk_tracks: Dict[int, List[float]] = {}

        processed_frames = 0
        total_objects_detected = 0
        completed_chunks = 0

        # Decode task helper
        def decode_chunk(chunk_info: Dict[str, Any]):
            c_id = chunk_info["chunk_id"]
            start_s = chunk_info["start_sec"]
            end_s = chunk_info["end_sec"]
            c_queue = chunk_queues[c_id]

            cap = cv2.VideoCapture(video_path)
            fps = meta["fps"]
            start_frame = int(start_s * fps)
            end_frame = int(end_s * fps)

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            current_frame_idx = start_frame

            batch_frames = []
            batch_timestamps = []

            while cap.isOpened() and current_frame_idx <= end_frame and not self.stop_requested:
                ret, frame = cap.read()
                if not ret:
                    break

                if (current_frame_idx - start_frame) % frame_stride == 0:
                    t_sec = round(current_frame_idx / fps, 2)
                    h, w = frame.shape[:2]
                    scale = self.config["image_size"] / max(h, w)
                    if scale < 1.0:
                        scaled_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                    else:
                        scaled_frame = frame

                    batch_frames.append(scaled_frame)
                    batch_timestamps.append(t_sec)

                    if len(batch_frames) >= batch_size:
                        c_queue.put({
                            "chunk_id": c_id,
                            "frames": list(batch_frames),
                            "timestamps": list(batch_timestamps),
                            "orig_w": w,
                            "orig_h": h
                        })
                        batch_frames.clear()
                        batch_timestamps.clear()

                current_frame_idx += 1

            if batch_frames:
                c_queue.put({
                    "chunk_id": c_id,
                    "frames": list(batch_frames),
                    "timestamps": list(batch_timestamps),
                    "orig_w": w,
                    "orig_h": h
                })

            cap.release()
            c_queue.put(None)
            return c_id

        # Start background decoder thread pool
        def decoder_worker():
            with concurrent.futures.ThreadPoolExecutor(max_workers=decode_workers) as executor:
                futures = {executor.submit(decode_chunk, chunk): chunk["chunk_id"] for chunk in chunks}
                for future in concurrent.futures.as_completed(futures):
                    c_id = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        print(f"⚠️ [VideoAuditor] Decoder exception in chunk {c_id}: {exc}")
                        chunk_queues[c_id].put(None)

        decode_thread = threading.Thread(target=decoder_worker, daemon=True)
        decode_thread.start()

        # Centralized MPS Inference Loop (Strictly Chronological Chunk Consumption)
        chunks_done_set = set()

        for chunk_info in chunks:
            if self.stop_requested:
                break

            chunk_id = chunk_info["chunk_id"]
            c_queue = chunk_queues[chunk_id]

            while not self.stop_requested:
                try:
                    item = c_queue.get(timeout=30.0)
                except queue.Empty:
                    if not decode_thread.is_alive():
                        break
                    continue

                if item is None:
                    break

                chunk_id = item["chunk_id"]
                frames = item["frames"]
                timestamps = item["timestamps"]

                # Optional Cloud GPU Offloading with seamless Local M2 fallback
                results = None
                if use_cloud:
                    try:
                        cloud_dets = cloud_gpu_adapter.infer_batch(frames, confidence=0.20)
                        if cloud_dets is not None:
                            # Use cloud detections
                            results = cloud_dets
                    except Exception as cg_err:
                        print(f"⚠️ [VideoAuditor] Cloud GPU inference exception: {cg_err}. Falling back to Local Processing.")

                if results is None:
                    try:
                        results = self.model.track(
                            frames,
                            persist=True,
                            tracker=self.config["tracker"],
                            device=self.device,
                            conf=0.20,
                            verbose=False
                        )
                    except Exception as inf_err:
                        print(f"⚠️ [VideoAuditor] Inference error on {self.device}: {inf_err}. Switching to CPU.")
                        self.device = "cpu"
                        results = self.model.track(
                            frames,
                            persist=True,
                            tracker=self.config["tracker"],
                            device="cpu",
                            conf=0.20,
                            verbose=False
                        )

                # Reconcile detections: if fine-tuned PPE model didn't catch person, query fallback base model
                for idx, res in enumerate(results):
                    has_person = res.boxes is not None and any(self.model.names.get(int(b.cls[0])) in ["person", "worker"] for b in res.boxes)
                    if not has_person and self.fallback_model is not None:
                        try:
                            fb_res = self.fallback_model.track(
                                [frames[idx]],
                                persist=True,
                                tracker=self.config["tracker"],
                                device=self.device,
                                conf=0.25,
                                verbose=False
                            )
                            if fb_res and len(fb_res) > 0 and fb_res[0].boxes is not None and len(fb_res[0].boxes) > 0:
                                results[idx] = fb_res[0]
                        except Exception:
                            pass

                for idx, res in enumerate(results):
                    t_sec = timestamps[idx]
                    processed_frames += 1

                    if res.boxes is not None:
                        boxes = res.boxes
                        for b_idx, box in enumerate(boxes):
                            total_objects_detected += 1
                            cls_id = int(box.cls[0])
                            cls_name = (res.names.get(cls_id) if (hasattr(res, "names") and res.names) else self.model.names.get(cls_id, f"class_{cls_id}")).lower()
                            conf = round(float(box.conf[0]), 2)
                            raw_track_id = int(box.id[0]) if (box.id is not None and len(box.id) > 0) else (b_idx + 1)
                            xyxy = [float(v) for v in box.xyxy[0]]

                            norm_box = [
                                round(xyxy[0] / frames[idx].shape[1], 3),
                                round(xyxy[1] / frames[idx].shape[0], 3),
                                round(xyxy[2] / frames[idx].shape[1], 3),
                                round(xyxy[3] / frames[idx].shape[0], 3)
                            ]

                            # Spatial IoU matching across chunk boundaries
                            if raw_track_id not in global_track_map and last_chunk_tracks:
                                best_match_id = None
                                best_iou = 0.45
                                for prev_id, prev_box in last_chunk_tracks.items():
                                    iou = compute_box_iou(norm_box, prev_box)
                                    if iou > best_iou:
                                        best_iou = iou
                                        best_match_id = prev_id
                                if best_match_id is not None:
                                    global_track_map[raw_track_id] = best_match_id

                            track_id = global_track_map.get(raw_track_id, raw_track_id)
                            last_chunk_tracks[track_id] = norm_box

                            det_item = {
                                "timestamp": t_sec,
                                "formatted_time": f"{int(t_sec // 60):02d}:{int(t_sec % 60):02d}",
                                "chunk_id": chunk_id,
                                "track_id": track_id,
                                "class": cls_name,
                                "confidence": conf,
                                "bbox": norm_box
                            }
                            all_detections.append(det_item)

                            # Evaluate hazards: PPE violations on person or direct hazard detections
                            is_hazard = False
                            hazard_type = cls_name.upper()

                            if cls_name == "person":
                                pixel_box = [int(v) for v in xyxy]
                                ppe = evaluate_ppe_for_person(frames[idx], pixel_box, detected_objects=all_detections)
                                det_item["ppe_evaluation"] = ppe
                                if not ppe["has_helmet"]:
                                    is_hazard = True
                                    hazard_type = "NO_HELMET"
                                    conf = ppe["helmet_conf"]
                                    det_item["class"] = "no_helmet"
                                elif not ppe["has_vest"]:
                                    is_hazard = True
                                    hazard_type = "NO_VEST"
                                    conf = ppe["vest_conf"]
                                    det_item["class"] = "no_vest"
                                else:
                                    det_item["class"] = "person (compliant)"
                            elif any(h in cls_name for h in ["no_helmet", "no_vest", "crane", "forklift"]):
                                is_hazard = True
                                hazard_type = cls_name.upper()

                            if is_hazard:
                                # Cross-chunk boundary tracking & deduplication
                                active_h = active_hazard_windows.get(hazard_type)
                                if active_h and (t_sec - active_h["last_seen"]) <= 12.0:
                                    active_h["last_seen"] = t_sec
                                    active_h["max_conf"] = max(active_h["max_conf"], conf)
                                    active_h["duration"] = round(t_sec - active_h["first_seen"], 1)
                                    if active_h["duration"] >= 2.0 and not active_h["reported"]:
                                        active_h["reported"] = True
                                    
                                        agent_eval = run_agent_reasoning(
                                            source_id=f"{source_id} (t={det_item['formatted_time']})",
                                            raw_detections=[det_item],
                                            duration=active_h["duration"],
                                            proximity=0.75,
                                            worker_speed=1.2
                                        )

                                        incident_record = {
                                            "id": f"INC-AUDIT-{track_id}-{int(t_sec)}",
                                            "timestamp": int(t_sec * 1000),
                                            "formattedTime": det_item["formatted_time"],
                                            "source": f"{meta['filename']} @ {det_item['formatted_time']}",
                                            "hazard": hazard_type,
                                            "level": agent_eval.get("threat_level", "CRITICAL"),
                                            "score": agent_eval.get("predictive_risk_score", 0.85),
                                            "sop_id": agent_eval.get("retrieved_sop", {}).get("sop_id", "SOP-EHS-042"),
                                            "standard_code": agent_eval.get("retrieved_sop", {}).get("standard_code", "OSHA 1910.135"),
                                            "action_plan": agent_eval.get("action_plan", []),
                                            "track_id": track_id,
                                            "status": "ACTIVE",
                                            "evidence": det_item
                                        }
                                        confirmed_incidents.append(incident_record)

                                        # Forward confirmed incident to Node.js Central Registry
                                        try:
                                            gw_url = self.config.get("node_gateway_url", "http://localhost:5000")
                                            httpx.post(
                                                f"{gw_url}/api/metrics",
                                                json={
                                                    "source": incident_record["source"],
                                                    "score": incident_record["score"],
                                                    "level": incident_record["level"],
                                                    "hazard": hazard_type,
                                                    "protocol": agent_eval.get("retrieved_sop", {}).get("summary", ""),
                                                    "sop_details": agent_eval.get("retrieved_sop", {}),
                                                    "hazards": [{"label": hazard_type, "confidence": conf}],
                                                    "operational_stages": agent_eval.get("operational_stages", []),
                                                    "logs": agent_eval.get("agent_logs", [])
                                                },
                                                timeout=1.0
                                            )
                                        except Exception:
                                            pass
                                elif not active_h or (t_sec - active_h["last_seen"]) > 12.0:
                                    active_hazard_windows[hazard_type] = {
                                        "first_seen": t_sec,
                                        "last_seen": t_sec,
                                        "violation": hazard_type,
                                        "max_conf": conf,
                                        "duration": 0.0,
                                        "reported": False
                                    }

            chunks_done_set.add(chunk_id)
            completed_chunks = len(chunks_done_set)
            elapsed_sec = round(time.time() - start_time, 1)
            fps_speed = round(processed_frames / max(0.1, elapsed_sec), 1)

            est_total_frames = int((meta["frame_count"] / frame_stride))
            remain_frames = max(0, est_total_frames - processed_frames)
            eta_sec = round(remain_frames / max(1.0, fps_speed), 1)
            progress_pct = min(100.0, round((completed_chunks / max(1, total_chunks)) * 100.0, 1))

            progress_payload = {
                "video_filename": meta["filename"],
                "video_duration_sec": meta["duration_sec"],
                "total_chunks": total_chunks,
                "completed_chunks": completed_chunks,
                "active_workers": decode_workers,
                "device": active_device,
                "processed_frames": processed_frames,
                "total_objects": total_objects_detected,
                "current_fps": fps_speed,
                "elapsed_sec": elapsed_sec,
                "eta_sec": eta_sec,
                "progress_percent": progress_pct,
                "incidents_count": len(confirmed_incidents),
                "latest_detections": all_detections[-4:] if all_detections else []
            }

            if progress_callback:
                progress_callback(progress_payload)

            # Stream progress to Node.js Matrix Gateway
            try:
                gw_url = self.config.get("node_gateway_url", "http://localhost:5000")
                httpx.post(f"{gw_url}/api/audit/progress", json=progress_payload, timeout=0.4)
            except Exception:
                pass

        total_elapsed = round(time.time() - start_time, 2)
        overall_fps = round(processed_frames / max(0.1, total_elapsed), 1)

        print(f"🏁 [VideoAuditor] Audit Complete: {processed_frames} frames in {total_elapsed}s ({overall_fps} FPS, Device: {active_device}). Incidents: {len(confirmed_incidents)}")

        return {
            "metadata": meta,
            "total_chunks": total_chunks,
            "completed_chunks": completed_chunks,
            "device": active_device,
            "processed_frames": processed_frames,
            "total_objects": total_objects_detected,
            "elapsed_seconds": total_elapsed,
            "processing_speed_fps": overall_fps,
            "incidents": confirmed_incidents,
            "detections_sample": all_detections[:50]
        }

    def audit_video_sequential(self, video_path: str, max_frames: Optional[int] = None) -> Dict[str, Any]:
        """Baseline sequential decoding and inference for benchmark comparison."""
        t0 = time.time()
        meta = self.get_video_metadata(video_path)
        cap = cv2.VideoCapture(video_path)
        frame_stride = int(self.config["frame_stride"])

        frames_processed = 0
        total_objects = 0
        current_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if current_idx % frame_stride == 0:
                h, w = frame.shape[:2]
                scale = self.config["image_size"] / max(h, w)
                if scale < 1.0:
                    scaled = cv2.resize(frame, (int(w * scale), int(h * scale)))
                else:
                    scaled = frame

                res = self.model(scaled, device="cpu", verbose=False)
                frames_processed += 1
                if res[0].boxes is not None:
                    total_objects += len(res[0].boxes)

                if max_frames and frames_processed >= max_frames:
                    break

            current_idx += 1

        cap.release()
        total_time = round(time.time() - t0, 2)
        fps = round(frames_processed / max(0.1, total_time), 1)

        return {
            "mode": "sequential_cpu",
            "frames_processed": frames_processed,
            "total_objects": total_objects,
            "elapsed_seconds": total_time,
            "fps": fps
        }

video_auditor = VideoAuditor()
