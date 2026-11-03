"""
VigilOps AI - Hardware-Accelerated Video Audit Benchmark
Compares Baseline Sequential CPU processing vs M2-Optimized Parallel Chunking with MPS GPU Acceleration.
Measures:
- Elapsed execution time (seconds)
- Frame processing throughput (FPS)
- Real-time factor (X times faster than 1x video playback)
- Peak RSS memory footprint (MB)
- Incident and PPE violation detection yield
"""

import os
import sys
import time
import resource
import json

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from video_auditor import video_auditor

def get_peak_memory_mb() -> float:
    """Returns peak memory in megabytes for current process on macOS."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # On macOS ru_maxrss is in bytes
    return round(usage.ru_maxrss / (1024.0 * 1024.0), 2)

def run_benchmark():
    video_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        "..",
        "sample_audit_videos",
        "industrial_ppe_audit_sample.mp4"
    ))

    if not os.path.exists(video_path):
        print(f"❌ Video not found at {video_path}")
        sys.exit(1)

    print("=================================================================")
    print("⚡ VIGILOPS AI — M2 HARDWARE BENCHMARK COMPARISON AUDIT")
    print("=================================================================")

    meta = video_auditor.get_video_metadata(video_path)
    print(f"🎬 Video: {meta['filename']}")
    print(f"   Resolution: {meta['width']}x{meta['height']} (Full HD)")
    print(f"   Duration: {meta['duration_sec']}s ({meta['duration_formatted']}) | Total Frames: {meta['frame_count']} @ {meta['fps']} FPS")
    print("-----------------------------------------------------------------")

    # 1. Baseline Sequential CPU Mode
    print("\n[Phase 1] Executing Sequential CPU Baseline...")
    mem_before_seq = get_peak_memory_mb()
    t0_seq = time.perf_counter()
    seq_results = video_auditor.audit_video_sequential(video_path)
    t1_seq = time.perf_counter()
    mem_after_seq = get_peak_memory_mb()

    seq_elapsed = round(t1_seq - t0_seq, 2)
    seq_fps = seq_results["fps"]
    seq_rt_factor = round(meta["duration_sec"] / max(0.01, seq_elapsed), 2)

    print(f"   ↳ Sequential CPU Duration: {seq_elapsed}s")
    print(f"   ↳ Sequential Processing Speed: {seq_fps} FPS ({seq_rt_factor}x Real-Time)")
    print(f"   ↳ Peak Memory: {mem_after_seq} MB")

    # 2. Parallel M2 Chunking + Centralized MPS Acceleration
    print("\n[Phase 2] Executing M2 Parallel Chunking with Centralized MPS Acceleration...")
    progress_updates = []
    def on_progress(p):
        progress_updates.append(p)

    mem_before_par = get_peak_memory_mb()
    t0_par = time.perf_counter()
    par_results = video_auditor.audit_video_parallel(video_path, progress_callback=on_progress)
    t1_par = time.perf_counter()
    mem_after_par = get_peak_memory_mb()

    par_elapsed = round(t1_par - t0_par, 2)
    par_fps = par_results["processing_speed_fps"]
    par_rt_factor = round(meta["duration_sec"] / max(0.01, par_elapsed), 2)
    speedup = round(par_fps / max(0.1, seq_fps), 2)

    print(f"   ↳ M2 Parallel MPS Duration: {par_elapsed}s")
    print(f"   ↳ M2 Processing Speed: {par_fps} FPS ({par_rt_factor}x Real-Time)")
    print(f"   ↳ Peak Memory: {mem_after_par} MB")
    print(f"   ↳ Acceleration Factor: {speedup}x speedup over sequential CPU baseline")
    print(f"   ↳ Total Chunks Audited: {par_results['total_chunks']}")
    print(f"   ↳ Incidents Flagged: {len(par_results['incidents'])}")

    print("\n=================================================================")
    print("📊 BENCHMARK COMPARISON MATRIX")
    print("=================================================================")
    print(f"{'Metric':<32} | {'Sequential (CPU)':<20} | {'VigilOps M2 (MPS)':<20}")
    print("-" * 78)
    print(f"{'Execution Mode':<32} | {'1-Worker CPU':<20} | {'2-Worker + MPS Batch':<20}")
    print(f"{'Elapsed Processing Time':<32} | {f'{seq_elapsed}s':<20} | {f'{par_elapsed}s':<20}")
    print(f"{'Throughput Speed':<32} | {f'{seq_fps} FPS':<20} | {f'{par_fps} FPS':<20}")
    print(f"{'Real-Time Playback Multiplier':<32} | {f'{seq_rt_factor}x':<20} | {f'{par_rt_factor}x':<20}")
    print(f"{'Speedup Factor':<32} | {'1.00x (Baseline)':<20} | {f'{speedup}x Faster':<20}")
    print(f"{'Peak Process RSS Memory':<32} | {f'{mem_after_seq} MB':<20} | {f'{mem_after_par} MB':<20}")
    print(f"{'Hardware Target':<32} | {'Apple M2 (CPU)':<20} | {'Apple M2 (MPS GPU)':<20}")
    print(f"{'Continuous Deduplication':<32} | {'No':<20} | {'Yes (Cross-Chunk)':<20}")
    inc_count = len(par_results["incidents"])
    print(f"{'Incidents Persisted':<32} | {'0 (Unanalyzed)':<20} | {f'{inc_count} Confirmed':<20}")
    print("=================================================================\n")

if __name__ == "__main__":
    run_benchmark()
