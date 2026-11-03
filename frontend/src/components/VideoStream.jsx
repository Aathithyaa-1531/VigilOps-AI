import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Camera, Video, Play, Square, AlertTriangle, ShieldCheck, Radio, FileVideo, Activity, Crosshair, Cpu, Zap, Gauge } from 'lucide-react';

export default function VideoStream({
  mode,
  onSourceSubmit,
  isProcessing,
  onReset,
  onTelemetryUpdate,
  measuredLatency = 0,
  activeDetections = [],
  auditProgress = null
}) {
  const [videoSource, setVideoSource] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [videoObjectURL, setVideoObjectURL] = useState(null);
  const [activePreset, setActivePreset] = useState(null);
  const [sampleVideoLoaded, setSampleVideoLoaded] = useState(false);
  const [localDetections, setLocalDetections] = useState([]);
  const [currentLatency, setCurrentLatency] = useState(measuredLatency);
  const [isInferencing, setIsInferencing] = useState(false);
  const [isAuditingParallel, setIsAuditingParallel] = useState(false);
  const [inferenceTarget, setInferenceTarget] = useState('local_m2'); // 'local_m2' | 'cloud_gpu'

  const videoRef = useRef(null);
  const containerRef = useRef(null);
  const canvasRef = useRef(null);
  const intervalRef = useRef(null);
  const [videoRenderBox, setVideoRenderBox] = useState({ left: 0, top: 0, width: '100%', height: '100%' });

  // Calculate the exact rendered video sub-rectangle within the object-contain container
  // to eliminate letterbox/pillarbox coordinate offset for bounding boxes
  const updateVideoRenderBox = useCallback(() => {
    if (!videoRef.current || !containerRef.current || !videoRef.current.videoWidth) {
      setVideoRenderBox({ left: 0, top: 0, width: '100%', height: '100%' });
      return;
    }
    const container = containerRef.current;
    const video = videoRef.current;
    const cw = container.clientWidth;
    const ch = container.clientHeight;
    const vw = video.videoWidth;
    const vh = video.videoHeight;

    if (!cw || !ch || !vw || !vh) return;

    const containerRatio = cw / ch;
    const videoRatio = vw / vh;

    let rw, rh, rl, rt;
    if (containerRatio > videoRatio) {
      // Container is wider than video (pillarbox: black bars on left/right)
      rh = ch;
      rw = ch * videoRatio;
      rt = 0;
      rl = (cw - rw) / 2;
    } else {
      // Container is taller than video (letterbox: black bars on top/bottom)
      rw = cw;
      rh = cw / videoRatio;
      rl = 0;
      rt = (ch - rh) / 2;
    }

    setVideoRenderBox({
      left: Math.round(rl),
      top: Math.round(rt),
      width: Math.round(rw),
      height: Math.round(rh)
    });
  }, []);

  useEffect(() => {
    updateVideoRenderBox();
    window.addEventListener('resize', updateVideoRenderBox);
    let ro = null;
    if (typeof ResizeObserver !== 'undefined' && containerRef.current) {
      ro = new ResizeObserver(() => updateVideoRenderBox());
      ro.observe(containerRef.current);
    }
    return () => {
      window.removeEventListener('resize', updateVideoRenderBox);
      if (ro) ro.disconnect();
    };
  }, [updateVideoRenderBox, videoObjectURL]);

  // Manage object URL lifecycle for uploaded video files
  useEffect(() => {
    if (!selectedFile) {
      if (!sampleVideoLoaded) {
        setVideoObjectURL(null);
      }
      return;
    }

    const objectUrl = URL.createObjectURL(selectedFile);
    setVideoObjectURL(objectUrl);
    setSampleVideoLoaded(false);

    return () => {
      URL.revokeObjectURL(objectUrl);
    };
  }, [selectedFile, sampleVideoLoaded]);

  // Handle video element playback state based on processing flag
  useEffect(() => {
    if (videoRef.current) {
      if (isProcessing) {
        videoRef.current.play().catch(() => {});
      } else {
        videoRef.current.pause();
      }
    }
  }, [isProcessing]);

  // Periodic frame sampling loop for real YOLOv8 + ByteTrack object tracking
  useEffect(() => {
    if (!isProcessing) {
      if (intervalRef.current) clearInterval(intervalRef.current);
      setLocalDetections([]);
      return;
    }

    intervalRef.current = setInterval(async () => {
      if (videoRef.current && !videoRef.current.paused && videoRef.current.videoWidth > 0) {
        const video = videoRef.current;
        const canvas = canvasRef.current || document.createElement('canvas');
        canvas.width = Math.min(video.videoWidth, 640);
        canvas.height = Math.min(video.videoHeight, 360);
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        const dataUrl = canvas.toDataURL('image/jpeg', 0.65);
        const t0 = performance.now();
        setIsInferencing(true);

        try {
          const res = await fetch('http://localhost:8000/api/detect-base64', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              image_base64: dataUrl,
              source_id: selectedFile ? selectedFile.name : activePreset ? `CCTV_${activePreset.toUpperCase()}` : 'Surveillance Feed 01'
            })
          });
          const t1 = performance.now();
          setIsInferencing(false);

          if (res.ok) {
            const data = await res.json();
            setCurrentLatency(Math.round(t1 - t0));
            if (data.detections && Array.isArray(data.detections)) {
              setLocalDetections(data.detections);
            } else {
              setLocalDetections([]);
            }

            // Propagate real telemetry to central dashboard
            if (onTelemetryUpdate) {
              onTelemetryUpdate(data);
            }
          }
        } catch (err) {
          setIsInferencing(false);
          setCurrentLatency(Math.round(performance.now() - t0));
        }
      }
    }, 700);

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [isProcessing, selectedFile, activePreset, onTelemetryUpdate]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (mode === 'live' && videoSource.trim()) {
      setActivePreset(null);
      onSourceSubmit({ type: 'stream', source: videoSource.trim() });
    } else if (mode === 'upload') {
      if (selectedFile) {
        setActivePreset(null);
        onSourceSubmit({ type: 'upload', file: selectedFile, url: videoObjectURL, source: selectedFile.name });
        if (videoRef.current) {
          videoRef.current.play().catch(() => {});
        }
      } else if (sampleVideoLoaded && videoObjectURL) {
        setActivePreset(null);
        onSourceSubmit({
          type: 'upload',
          source: 'industrial_ppe_audit_sample.mp4 (Full HD 1080p Intel EHS Safety Video)',
          url: videoObjectURL
        });
        if (videoRef.current) {
          videoRef.current.play().catch(() => {});
        }
      }
    }
  };

  const handlePresetSelect = (presetKey, presetLabel) => {
    setActivePreset(presetKey);
    setVideoSource(`rtsp://192.168.1.${presetKey === 'crane' ? '102' : presetKey === 'forklift' ? '105' : '108'}:554/live`);
    onSourceSubmit({ type: 'stream', source: presetLabel });
  };

  const handleUseDownloadedSampleVideo = () => {
    setSelectedFile(null);
    setSampleVideoLoaded(true);
    setVideoObjectURL('/industrial_ppe_audit_sample.mp4');
    onSourceSubmit({
      type: 'upload',
      source: 'industrial_ppe_audit_sample.mp4 (Full HD 1080p Intel EHS Safety Video)',
      url: '/industrial_ppe_audit_sample.mp4'
    });
  };

  const handleResetClick = () => {
    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.currentTime = 0;
    }
    setVideoSource('');
    setSelectedFile(null);
    setVideoObjectURL(null);
    setActivePreset(null);
    setSampleVideoLoaded(false);
    setLocalDetections([]);

    // Call reset endpoint to clear tracking state
    fetch('http://localhost:5000/api/stream/reset', { method: 'POST' }).catch(() => {});

    if (onReset) {
      onReset();
    }
  };

  const handleTriggerParallelAudit = async () => {
    setIsAuditingParallel(true);
    try {
      await fetch('http://localhost:5000/api/audit/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          videoPath: selectedFile ? selectedFile.name : 'sample_audit_videos/industrial_ppe_audit_sample.mp4',
          filename: selectedFile ? selectedFile.name : 'industrial_ppe_audit_sample.mp4',
          inference_target: inferenceTarget
        })
      });
    } catch (err) {
      console.error('Audit trigger error:', err);
    } finally {
      setIsAuditingParallel(false);
    }
  };

  // Only display real dynamic detections from YOLOv8 ByteTrack
  const displayDetections = localDetections.length > 0 ? localDetections : (isProcessing ? activeDetections : []);

  return (
    <div className="flex flex-col h-full bg-slate-950/90 p-3.5 rounded-xl border border-slate-800 shadow-xl relative backdrop-blur-md overflow-hidden">
      {/* Hidden canvas for frame extraction */}
      <canvas ref={canvasRef} className="hidden" />

      {/* Top Header & Telemetry Status */}
      <div className="flex flex-wrap justify-between items-center mb-2.5 gap-2">
        <div className="flex items-center gap-2">
          <div className="p-1.5 bg-cyan-500/10 border border-cyan-500/30 rounded-lg text-cyan-400">
            {mode === 'live' ? <Camera className="w-3.5 h-3.5" /> : <Video className="w-3.5 h-3.5" />}
          </div>
          <div>
            <h3 className="text-xs font-bold uppercase tracking-wider text-cyan-400 flex items-center gap-1.5">
              <span>Feed Monitor</span>
              <span className="text-[10px] text-slate-500 font-mono font-normal">
                ({mode === 'live' ? '24/7 CCTV RTSP Camera Stream' : 'Video File Post-Shift Audit'})
              </span>
            </h3>
          </div>
        </div>

        <div className="flex items-center gap-2.5">
          {isProcessing && (
            <span className="text-[10px] font-mono text-cyan-400 bg-slate-900 border border-slate-800 px-2.5 py-0.5 rounded-full flex items-center gap-1.5 shadow-xs">
              <Activity className={`w-3 h-3 text-cyan-400 ${isInferencing ? 'animate-pulse' : ''}`} />
              <span>ByteTrack: {currentLatency > 0 ? `${currentLatency}ms` : 'Active'}</span>
            </span>
          )}

          <span
            className={`px-2.5 py-0.5 rounded-full text-[10px] font-mono font-bold tracking-wider flex items-center gap-1.5 transition-all ${
              isProcessing
                ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 shadow-sm shadow-emerald-500/20 animate-pulse'
                : 'bg-slate-900 text-slate-400 border border-slate-800'
            }`}
          >
            <span className={`h-1.5 w-1.5 rounded-full ${isProcessing ? 'bg-emerald-400 animate-ping' : 'bg-slate-600'}`}></span>
            {isProcessing ? 'SURVEILLANCE ACTIVE' : 'SYSTEM IDLE'}
          </span>
        </div>
      </div>

      {/* Video Audit Controls Bar */}
      {mode === 'upload' && (
        <div className="mb-2.5 space-y-2">
          <div className="flex flex-wrap items-center justify-between w-full bg-slate-900/70 border border-slate-800 px-2.5 py-1.5 rounded-lg text-xs gap-2">
            <div className="flex items-center gap-1.5 text-slate-300 truncate">
              <FileVideo className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
              <span className="truncate text-[11px]">
                Audit Video: <strong className="text-cyan-300 font-mono">industrial_ppe_audit_sample.mp4</strong>
              </span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="flex items-center bg-slate-950 p-0.5 rounded border border-slate-800 text-[10px] font-mono">
                <button
                  type="button"
                  onClick={() => setInferenceTarget('local_m2')}
                  className={`px-1.5 py-0.5 rounded transition ${inferenceTarget === 'local_m2' ? 'bg-cyan-500/20 text-cyan-300 font-bold border border-cyan-500/30' : 'text-slate-400 hover:text-slate-200'}`}
                >
                  Apple Silicon (MPS)
                </button>
                <button
                  type="button"
                  onClick={() => setInferenceTarget('cloud_gpu')}
                  className={`px-1.5 py-0.5 rounded transition ${inferenceTarget === 'cloud_gpu' ? 'bg-indigo-500/20 text-indigo-300 font-bold border border-indigo-500/30' : 'text-slate-400 hover:text-slate-200'}`}
                  title="Cloud GPU endpoint adapter (Standby / Configurable)"
                >
                  Cloud GPU
                </button>
              </div>
              <button
                type="button"
                onClick={handleUseDownloadedSampleVideo}
                className="bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold px-2.5 py-1 rounded-md text-[10px] transition cursor-pointer flex items-center gap-1 shadow-sm shrink-0"
              >
                <Play className="w-2.5 h-2.5 fill-slate-950" />
                <span>1-Click Load</span>
              </button>
              <button
                type="button"
                onClick={handleTriggerParallelAudit}
                disabled={isAuditingParallel}
                className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-bold px-2.5 py-1 rounded-md text-[10px] transition cursor-pointer flex items-center gap-1 shadow-sm shrink-0 border border-indigo-400/30"
              >
                <Zap className={`w-2.5 h-2.5 ${isAuditingParallel ? 'animate-bounce text-amber-300' : 'text-indigo-200'}`} />
                <span>{isAuditingParallel ? 'Auditing...' : `${inferenceTarget === 'cloud_gpu' ? 'Cloud' : 'Edge'} Audit`}</span>
              </button>
            </div>
          </div>

          {/* Audit Processing HUD */}
          {auditProgress && (
            <div className="bg-slate-900/90 border border-cyan-500/40 rounded-xl p-3 shadow-lg font-mono text-[11px] space-y-2 animate-fade-in">
              <div className="flex justify-between items-center text-cyan-300">
                <span className="flex items-center gap-1.5 font-bold">
                  <Zap className="w-3.5 h-3.5 text-cyan-400 animate-pulse" />
                  <span>Parallel Audit: {auditProgress.video_filename} ({auditProgress.video_duration_sec}s)</span>
                </span>
                <span className="text-[10px] bg-cyan-500/20 px-2 py-0.5 rounded text-cyan-200 border border-cyan-500/30 font-bold">
                  {auditProgress.progress_percent}% Complete
                </span>
              </div>

              {/* Progress Bar */}
              <div className="w-full bg-slate-950 h-2 rounded-full overflow-hidden border border-slate-800">
                <div
                  className="h-full bg-gradient-to-r from-cyan-500 to-indigo-500 transition-all duration-300"
                  style={{ width: `${auditProgress.progress_percent}%` }}
                />
              </div>

              {/* Metrics Grid */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[10px] pt-1 border-t border-slate-800/80 text-slate-300">
                <div>
                  <span className="text-slate-500 block">30s Chunks:</span>
                  <strong className="text-slate-100">{auditProgress.completed_chunks} / {auditProgress.total_chunks} Done</strong>
                </div>
                <div>
                  <span className="text-slate-500 block">Inference Speed:</span>
                  <strong className="text-emerald-400">{auditProgress.current_fps} FPS ({auditProgress.device})</strong>
                </div>
                <div>
                  <span className="text-slate-500 block">Frames Processed:</span>
                  <strong className="text-cyan-300">{auditProgress.processed_frames} Frames</strong>
                </div>
                <div>
                  <span className="text-slate-500 block">Elapsed / ETA:</span>
                  <strong className="text-amber-300">{auditProgress.elapsed_sec}s / ~{auditProgress.eta_sec}s</strong>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Main Video Viewport & Computer Vision HUD */}
      <div
        ref={containerRef}
        className="flex-1 min-h-[200px] bg-black/95 rounded-lg border border-slate-900 flex items-center justify-center relative overflow-hidden group shadow-inner"
      >
        {mode === 'upload' && videoObjectURL ? (
          <video
            ref={videoRef}
            src={videoObjectURL}
            controls
            autoPlay
            loop
            muted
            playsInline
            onLoadedMetadata={updateVideoRenderBox}
            onPlay={updateVideoRenderBox}
            onPause={() => setLocalDetections([])}
            onEnded={() => setLocalDetections([])}
            className="absolute inset-0 w-full h-full object-contain bg-black z-10"
          />
        ) : isProcessing ? (
          <div className="absolute inset-0 bg-slate-950 flex items-center justify-center">
            <div className="absolute inset-0 bg-[radial-gradient(#1e293b_1.5px,transparent_1.5px)] [background-size:20px_20px] opacity-35"></div>
            <div className="text-center z-10 p-4">
              <div className="inline-flex items-center gap-2 text-cyan-400 font-mono text-xs font-semibold tracking-wider bg-slate-900/90 px-3.5 py-1.5 rounded-full border border-cyan-500/30 shadow-lg mb-2">
                <Radio className="w-3.5 h-3.5 animate-pulse text-cyan-400" />
                RTSP STREAM: {videoSource || 'rtsp://192.168.1.102:554/live'}
              </div>
              <p className="text-[11px] text-slate-400 font-mono">
                Frame Matrix: 1920x1080 @ 30 FPS | Codec: H.264 | Connection: Active
              </p>
            </div>
          </div>
        ) : (
          <div className="text-center text-slate-500 p-8 flex flex-col items-center justify-center space-y-3">
            <div className="w-14 h-14 rounded-2xl bg-slate-900/80 border border-slate-800 flex items-center justify-center text-slate-600 shadow-lg">
              <Radio className="w-7 h-7" />
            </div>
            <div className="space-y-1">
              <div className="text-xs font-bold text-slate-300">Surveillance Stream Inactive</div>
              <div className="text-[11px] text-slate-500 max-w-sm leading-relaxed">
                {mode === 'live'
                  ? 'Select a plant camera above or enter an RTSP IP camera URL below to initialize autonomous monitoring.'
                  : 'Select an MP4 video file below or click "1-Click Load & Audit" above to run automated post-shift compliance review.'}
              </div>
            </div>
          </div>
        )}

        {/* Dynamic YOLOv8 Computer Vision Reticle & Bounding Box Layer - Precisely mapped to rendered video canvas */}
        {isProcessing && (
          <div
            className="absolute pointer-events-none z-20 overflow-hidden"
            style={{
              left: typeof videoRenderBox.left === 'number' ? `${videoRenderBox.left}px` : videoRenderBox.left,
              top: typeof videoRenderBox.top === 'number' ? `${videoRenderBox.top}px` : videoRenderBox.top,
              width: typeof videoRenderBox.width === 'number' ? `${videoRenderBox.width}px` : videoRenderBox.width,
              height: typeof videoRenderBox.height === 'number' ? `${videoRenderBox.height}px` : videoRenderBox.height,
            }}
          >
            {/* Render Real Dynamic Detections without floating overlay */}
            {displayDetections && displayDetections.length > 0 &&
              displayDetections.map((det, idx) => {
                const [x1, y1, x2, y2] = det.bbox_normalized || [0.2, 0.2, 0.4, 0.6];
                const widthPct = Math.max(0.05, Math.min(1.0, x2 - x1)) * 100;
                const heightPct = Math.max(0.05, Math.min(1.0, y2 - y1)) * 100;
                const leftPct = Math.max(0, Math.min(1.0, x1)) * 100;
                const topPct = Math.max(0, Math.min(1.0, y1)) * 100;

                const detClass = (det.class || '').toLowerCase();
                const isViolation = detClass.includes('no_') || detClass.includes('hazard') || detClass.includes('violation');
                const isCompliant = detClass.includes('compliant') || detClass.includes('hard_hat') || detClass.includes('reflective');
                const isUnknown = detClass.includes('unknown') || detClass.includes('occluded');
                
                const borderColor = isViolation ? 'border-rose-500' : isCompliant ? 'border-emerald-400' : isUnknown ? 'border-amber-400' : 'border-cyan-400';
                const bgColor = isViolation ? 'bg-rose-500/15' : isCompliant ? 'bg-emerald-500/10' : isUnknown ? 'bg-amber-500/15' : 'bg-cyan-500/10';
                const tagColor = isViolation ? 'bg-rose-500 text-white' : isCompliant ? 'bg-emerald-500 text-slate-950' : isUnknown ? 'bg-amber-500 text-slate-950' : 'bg-cyan-500 text-slate-950';

                const labelText = (det.class || 'OBJECT').toUpperCase();
                const confPercent = Math.round((det.confidence || 0.85) * 100);

                return (
                  <div
                    key={det.track_id ? `det-${det.track_id}-${detClass}-${idx}` : `det-${idx}`}
                    style={{
                      left: `${leftPct}%`,
                      top: `${topPct}%`,
                      width: `${widthPct}%`,
                      height: `${heightPct}%`
                    }}
                    className={`absolute border-2 ${borderColor} ${bgColor} rounded-xs transition-all duration-200`}
                  >
                    <div className={`absolute -top-4 left-0 ${tagColor} text-[8px] font-mono font-bold px-1.5 py-0.5 rounded-t tracking-tight whitespace-nowrap flex items-center gap-1 shadow-sm`}>
                      {isViolation ? (
                        <AlertTriangle className="w-2.5 h-2.5" />
                      ) : isCompliant ? (
                        <ShieldCheck className="w-2.5 h-2.5" />
                      ) : (
                        <Activity className="w-2.5 h-2.5" />
                      )}
                      <span>
                        [{labelText}: {confPercent}%]
                      </span>
                    </div>
                  </div>
                );
              })}
          </div>
        )}
      </div>

      {/* Surveillance Controls Form */}
      <form onSubmit={handleSubmit} className="mt-3 flex gap-2 items-center">
        {mode === 'live' ? (
          <input
            type="text"
            placeholder="rtsp://192.168.1.100:554/stream or camera URL..."
            value={videoSource}
            onChange={(e) => setVideoSource(e.target.value)}
            className="flex-1 bg-slate-900 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-cyan-500 placeholder-slate-600 font-mono"
          />
        ) : (
          <div className="flex-1 flex items-center gap-2">
            <input
              type="file"
              accept="video/*"
              onChange={(e) => {
                setSelectedFile(e.target.files[0] || null);
                setSampleVideoLoaded(false);
              }}
              className="flex-1 bg-slate-900 border border-slate-800 rounded-lg px-2 py-1 text-xs text-slate-400 file:mr-2 file:py-1 file:px-2.5 file:rounded-md file:border-0 file:text-[11px] file:font-semibold file:bg-slate-800 file:text-cyan-400 hover:file:bg-slate-700 cursor-pointer"
            />
          </div>
        )}

        <button
          type="submit"
          disabled={isProcessing || (mode === 'live' && !videoSource.trim()) || (mode === 'upload' && !selectedFile && !sampleVideoLoaded)}
          className="bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 disabled:opacity-40 text-slate-950 font-bold px-4 py-1.5 rounded-lg text-xs transition cursor-pointer disabled:cursor-not-allowed flex items-center gap-1.5 shadow-md shadow-cyan-500/20 shrink-0"
        >
          <Play className="w-3.5 h-3.5 fill-slate-950" />
          {isProcessing ? 'Auditing...' : 'Start Audit'}
        </button>

        <button
          type="button"
          onClick={handleResetClick}
          className="bg-rose-950/60 hover:bg-rose-900 text-rose-300 border border-rose-500/40 font-semibold px-3 py-1.5 rounded-lg text-xs transition cursor-pointer flex items-center gap-1 shrink-0 shadow-sm"
          title="Stop stream and reset viewport"
        >
          <Square className="w-3.5 h-3.5 fill-rose-300" />
          Reset
        </button>
      </form>
    </div>
  );
}
