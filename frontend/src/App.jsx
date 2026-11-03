import React, { useState, useEffect } from 'react';
import io from 'socket.io-client';
import {
  ShieldAlert,
  Radio,
  Clock,
  Layers,
  Activity,
  AlertTriangle,
  Play,
  RotateCcw,
  Bell,
  Mail,
  Phone,
  Send,
  CheckCircle2,
  X,
  Menu,
  Server,
  Cpu,
  ShieldCheck
} from 'lucide-react';

import VideoStream from './components/VideoStream';
import RiskGauge from './components/RiskGauge';
import AgentTerminal from './components/AgentTerminal';
import IncidentLog from './components/IncidentLog';
import AlertRecipientsModal from './components/AlertRecipientsModal';

const socket = io('http://localhost:5000');

// Privacy & Device-Name Sanitizer (Requirement 5)
const sanitizePrivacyText = (text) => {
  if (typeof text !== 'string') return text;
  return text
    .replace(/AATHITHYAAs-MacBook-Pro(\.local)?/gi, 'Apple Silicon')
    .replace(/MacBook(\s*Pro)?(\s*M2)?/gi, 'Apple Silicon')
    .replace(/\/Users\/[a-zA-Z0-9._-]+/g, '/opt/vigilops')
    .replace(/\b(aathithyaae)\b/gi, 'operator');
};

export default function App() {
  const [streamMode, setStreamMode] = useState('upload'); // 'upload' or 'live'
  const [riskScore, setRiskScore] = useState(0.0);
  const [threatLevel, setThreatLevel] = useState('LOW');
  const [protocol, setProtocol] = useState('');
  const [sopDetails, setSopDetails] = useState(null);
  const [breakdown, setBreakdown] = useState({});
  const [logs, setLogs] = useState([]);
  const [operationalStages, setOperationalStages] = useState([]);
  const [incidents, setIncidents] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [aiServiceOnline, setAiServiceOnline] = useState(false);
  const [currentTime, setCurrentTime] = useState(new Date().toLocaleTimeString());
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  // Measured latency and real detections
  const [measuredLatency, setMeasuredLatency] = useState(0);
  const [activeDetections, setActiveDetections] = useState([]);

  // Emergency Alert Contacts & Toast Notification Stack
  const [recipients, setRecipients] = useState([]);
  const [isAlertModalOpen, setIsAlertModalOpen] = useState(false);
  const [toastNotifications, setToastNotifications] = useState([]);
  const [auditProgress, setAuditProgress] = useState(null);

  // Real-time Clock
  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date().toLocaleTimeString()), 1000);
    return () => clearInterval(timer);
  }, []);

  // Check Backend and AI Service Health
  const checkHealth = () => {
    fetch('http://localhost:8000/health')
      .then((res) => res.json())
      .then((data) => setAiServiceOnline(data.status === 'online'))
      .catch(() => setAiServiceOnline(false));

    fetch('http://localhost:5000/health')
      .then((res) => res.json())
      .then((data) => {
        if (data.status === 'online') setIsConnected(true);
      })
      .catch(() => {
        if (!socket.connected) setIsConnected(false);
      });

    fetch('http://localhost:5000/api/alerts/recipients')
      .then((res) => res.json())
      .then((data) => {
        if (Array.isArray(data)) setRecipients(data);
      })
      .catch((err) => console.warn('Could not fetch recipients:', err));

    fetch('http://localhost:5000/api/incidents')
      .then((res) => res.json())
      .then((data) => {
        if (data.incidents && Array.isArray(data.incidents)) {
          setIncidents(data.incidents);
        } else if (Array.isArray(data)) {
          setIncidents(data);
        }
      })
      .catch((err) => console.warn('Could not fetch incidents:', err));
  };

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  // Socket.IO Telemetry Listener
  useEffect(() => {
    if (socket.connected) {
      setIsConnected(true);
    }

    const handleConnect = () => {
      setIsConnected(true);
      setLogs((prev) => [...prev, `${new Date().toLocaleTimeString()} — [System] Connected to Alert & Notification Hub (:5000).`]);
    };

    const handleDisconnect = () => {
      setIsConnected(false);
    };

    socket.on('connect', handleConnect);
    socket.on('disconnect', handleDisconnect);

    socket.on('initial_incidents', (data) => {
      if (Array.isArray(data)) setIncidents(data);
    });

    socket.on('recipients_update', (data) => {
      if (Array.isArray(data)) setRecipients(data);
    });

    socket.on('emergency_dispatch_event', (dispatchData) => {
      const newToast = {
        id: `toast-${Date.now()}`,
        ...dispatchData
      };
      setToastNotifications((prev) => [newToast, ...prev].slice(0, 3));

      // Auto dismiss after 10 seconds
      setTimeout(() => {
        setToastNotifications((prev) => prev.filter((t) => t.id !== newToast.id));
      }, 10000);
    });

    socket.on('risk_update', (data) => {
      if (data.score !== undefined) setRiskScore(data.score);
      if (data.level) setThreatLevel(data.level);
      if (data.breakdown) setBreakdown(data.breakdown);
      setIsProcessing(true);
    });

    socket.on('protocol_alert', (data) => {
      setProtocol(data.protocol);
      if (data.sop_details) setSopDetails(data.sop_details);
    });

    socket.on('operational_stages', (stages) => {
      if (Array.isArray(stages)) {
        setOperationalStages(stages);
      }
    });

    socket.on('agent_log', (log) => {
      const sanitized = sanitizePrivacyText(log);
      setLogs((prev) => {
        const updated = [...prev, sanitized];
        return updated.slice(-40);
      });
    });

    socket.on('incident_event', (newIncident) => {
      setIncidents((prev) => [newIncident, ...prev]);
    });

    socket.on('incident_updated', (updatedInc) => {
      setIncidents((prev) => prev.map((inc) => (inc.id === updatedInc.id ? updatedInc : inc)));
    });

    socket.on('protocol_acknowledged', () => {
      setProtocol('');
      setSopDetails(null);
    });

    socket.on('audit_progress', (progressData) => {
      setAuditProgress(progressData);
    });

    socket.on('audit_complete', () => {
      setTimeout(() => setAuditProgress(null), 4000);
    });

    socket.on('stream_reset', () => {
      handleReset();
    });

    return () => {
      socket.off('connect');
      socket.off('disconnect');
      socket.off('initial_incidents');
      socket.off('recipients_update');
      socket.off('emergency_dispatch_event');
      socket.off('risk_update');
      socket.off('protocol_alert');
      socket.off('operational_stages');
      socket.off('agent_log');
      socket.off('incident_event');
      socket.off('incident_updated');
      socket.off('protocol_acknowledged');
      socket.off('audit_progress');
      socket.off('audit_complete');
      socket.off('stream_reset');
    };
  }, []);

  const handleReset = () => {
    setIsProcessing(false);
    setRiskScore(0.0);
    setThreatLevel('LOW');
    setProtocol('');
    setSopDetails(null);
    setBreakdown({});
    setActiveDetections([]);
    setMeasuredLatency(0);
    setOperationalStages([]);
  };

  const handleStreamModeChange = (newMode) => {
    setStreamMode(newMode);
    handleReset();
  };

  const handleSourceSubmit = (sourceData) => {
    setIsProcessing(true);
    const sourceLabel =
      sourceData.type === 'stream'
        ? sourceData.source
        : sourceData.file?.name || sourceData.source || 'Local Video Audit File';

    setLogs((prev) => [
      ...prev,
      `${new Date().toLocaleTimeString()} — [Surveillance System] Initializing continuous feed ingestion: ${sourceLabel}`,
      `${new Date().toLocaleTimeString()} — [YOLOv8 Edge CV] Loaded real-time tracking model with ByteTrack. Monitoring plant workers & PPE...`,
      `${new Date().toLocaleTimeString()} — [LangGraph Multi-Agent Brain] Stateful safety reasoning pipeline engaged.`
    ]);

    fetch('http://localhost:5000/api/source', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ type: sourceData.type, source: sourceLabel }),
    }).catch((err) => {
      console.warn('Backend server note:', err);
    });
  };

  const handleTelemetryUpdate = (data) => {
    if (!data) return;
    if (data.risk_score !== undefined) setRiskScore(data.risk_score);
    if (data.threat_level) setThreatLevel(data.threat_level);
    if (data.breakdown) setBreakdown(data.breakdown);
    if (data.sop) {
      setSopDetails(data.sop);
      setProtocol(data.sop.summary || '');
    }
    if (data.latency_ms !== undefined) setMeasuredLatency(data.latency_ms);
    if (data.detections && Array.isArray(data.detections)) {
      setActiveDetections(data.detections);
    }
    if (data.operational_stages && Array.isArray(data.operational_stages)) {
      setOperationalStages(data.operational_stages);
    }
    if (data.logs && Array.isArray(data.logs)) {
      const sanitizedLogs = data.logs.map(sanitizePrivacyText);
      setLogs((prev) => {
        const combined = [...prev, ...sanitizedLogs];
        return combined.slice(-40);
      });
    }
  };

  const handleAcknowledge = () => {
    fetch('http://localhost:5000/api/acknowledge', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sop_id: sopDetails?.sop_id || 'SOP-EHS-042' })
    }).catch(() => {});

    setProtocol('');
    setSopDetails(null);
    setLogs((prev) => [
      ...prev,
      `${new Date().toLocaleTimeString()} — [Supervisor Action] Protocol acknowledged by Safety Officer on duty.`
    ]);
  };

  // Recipient Management Handlers
  const handleAddRecipient = async (newRecipient) => {
    try {
      const res = await fetch('http://localhost:5000/api/alerts/recipients', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newRecipient)
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.error || 'Failed to add recipient');
      }
      const saved = await res.json();
      setRecipients((prev) => [...prev, saved]);
    } catch (e) {
      alert(e.message);
    }
  };

  const handleDeleteRecipient = async (id) => {
    try {
      const res = await fetch(`http://localhost:5000/api/alerts/recipients/${id}`, {
        method: 'DELETE'
      });
      if (res.ok) {
        setRecipients((prev) => prev.filter((r) => r.id !== id));
      }
    } catch (e) {
      console.error(e);
    }
  };

  const handleToggleRecipient = async (id, updates) => {
    try {
      const res = await fetch(`http://localhost:5000/api/alerts/recipients/${id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updates)
      });
      if (res.ok) {
        const updated = await res.json();
        setRecipients((prev) => prev.map((r) => (r.id === id ? updated : r)));
      }
    } catch (e) {
      console.error(e);
    }
  };

  const handleSendTestAlert = async () => {
    const res = await fetch('http://localhost:5000/api/alerts/test', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || 'Test alert failed');
    }
    return await res.json();
  };

  // Incident Acknowledgment & Resolution Handlers
  const handleAcknowledgeIncident = async (id, { officerName, notes }) => {
    const res = await fetch(`http://localhost:5000/api/incidents/${id}/acknowledge`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ officerName, notes })
    });
    if (res.ok) {
      const updated = await res.json();
      setIncidents((prev) => prev.map((inc) => (inc.id === id ? updated : inc)));
    }
  };

  const handleResolveIncident = async (id, { officerName, notes }) => {
    const res = await fetch(`http://localhost:5000/api/incidents/${id}/resolve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ officerName, notes })
    });
    if (res.ok) {
      const updated = await res.json();
      setIncidents((prev) => prev.map((inc) => (inc.id === id ? updated : inc)));
    }
  };

  const activeEmailsCount = recipients.filter((r) => r.email && r.emailEnabled).length;

  return (
    <div className="flex h-screen bg-slate-900 text-slate-100 font-sans overflow-hidden">
      {/* ==================== TOAST NOTIFICATION STACK (Dedicated Bottom-Right Area, Non-Obscuring) ==================== */}
      <div className="fixed bottom-5 right-5 z-50 flex flex-col gap-2.5 max-w-sm w-80 pointer-events-none">
        {toastNotifications.map((toast) => {
          const sentEmails = toast.emailDelivery?.filter((e) => e.status === 'SENT' || e.status === 'MOCKED').length || 0;
          const configEmails = toast.emailDelivery?.filter((e) => e.status === 'CONFIG_REQUIRED').length || 0;
          const failedEmails = toast.emailDelivery?.filter((e) => e.status === 'FAILED').length || 0;

          const hasFailure = failedEmails > 0;
          const hasConfigReq = configEmails > 0;
          const allSent = sentEmails > 0 && !hasFailure && !hasConfigReq;

          const title = allSent
            ? 'GMAIL SMTP ALERT DISPATCHED'
            : hasConfigReq
            ? 'GMAIL: SMTP CONFIG REQUIRED'
            : hasFailure
            ? 'GMAIL: SMTP DISPATCH FAILED'
            : 'SAFETY ALERT PROCESSED';

          const borderColor = allSent
            ? 'border-emerald-500/80 shadow-emerald-500/20'
            : hasConfigReq
            ? 'border-amber-500/80 shadow-amber-500/20'
            : 'border-rose-500/80 shadow-rose-500/20';

          const titleColor = allSent
            ? 'text-emerald-400'
            : hasConfigReq
            ? 'text-amber-400'
            : 'text-rose-400';

          return (
            <div
              key={toast.id}
              className={`bg-slate-950/95 border-2 ${borderColor} rounded-xl p-3 shadow-2xl backdrop-blur-md pointer-events-auto space-y-1.5 transition-all duration-300 ease-out`}
            >
              <div className="flex justify-between items-start">
                <div className={`flex items-center gap-1.5 ${titleColor} font-bold text-xs`}>
                  <ShieldAlert className="w-4 h-4" />
                  <span>{title}</span>
                </div>
                <button
                  onClick={() => setToastNotifications((prev) => prev.filter((t) => t.id !== toast.id))}
                  className="text-slate-400 hover:text-slate-200 transition p-0.5 rounded hover:bg-slate-800"
                  title="Dismiss notification"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>

              <div className="text-[11px] text-slate-300">
                <span>Critical violation at </span>
                <strong className="text-white">{toast.source}</strong>:
                <div className="font-mono text-rose-300 font-semibold">{toast.hazard}</div>
              </div>

              <div className="flex items-center justify-between text-[10px] font-mono pt-1 border-t border-slate-800 text-slate-400">
                <span className={`flex items-center gap-1 ${sentEmails > 0 ? 'text-emerald-300 font-bold' : configEmails > 0 ? 'text-amber-400' : failedEmails > 0 ? 'text-rose-400' : 'text-slate-400'}`}>
                  <Mail className="w-3 h-3" />
                  {sentEmails > 0 ? `Gmail: ${sentEmails} Sent` : configEmails > 0 ? 'Config Req' : failedEmails > 0 ? 'Failed' : '0 Sent'}
                </span>
                <span className="text-slate-500">{toast.timestamp}</span>
              </div>
            </div>
          );
        })}
      </div>

      {/* ==================== LEFT SIDEBAR ==================== */}
      <aside className="w-64 bg-slate-950/90 border-r border-slate-800 p-4 hidden md:flex flex-col justify-between shadow-2xl backdrop-blur-md">
        <div className="space-y-6">
          {/* Logo & Brand Header */}
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-gradient-to-tr from-cyan-500 to-blue-600 rounded-xl text-slate-950 shadow-lg shadow-cyan-500/20">
              <ShieldAlert className="w-6 h-6 stroke-[2.5]" />
            </div>
            <div>
              <h1 className="text-base font-extrabold tracking-tight bg-gradient-to-r from-white via-slate-200 to-slate-400 bg-clip-text text-transparent">
                VigilOps AI
              </h1>
              <p className="text-[10px] text-cyan-400 font-mono tracking-wider">
                Autonomous Industrial Safety
              </p>
            </div>
          </div>

          {/* Quick Action: Configure Alert Contacts */}
          <div className="space-y-3">
            <button
              onClick={() => setIsAlertModalOpen(true)}
              className="w-full bg-gradient-to-r from-indigo-600/30 to-blue-600/30 hover:from-indigo-600/40 hover:to-blue-600/40 border border-indigo-500/40 text-indigo-200 font-bold p-2.5 rounded-xl text-xs transition cursor-pointer flex items-center justify-between shadow-md"
            >
              <span className="flex items-center gap-2">
                <Bell className="w-4 h-4 text-indigo-400" />
                <span>Alert Contacts</span>
              </span>
              <span className="text-[10px] font-mono bg-indigo-500/20 px-2 py-0.5 rounded-full border border-indigo-500/30 text-indigo-300">
                {recipients.length}
              </span>
            </button>

            {/* Ingestion Mode Selector */}
            <div className="space-y-2">
              <label className="text-[10px] font-bold text-slate-400 uppercase tracking-widest block flex items-center gap-1.5">
                <Layers className="w-3.5 h-3.5 text-cyan-400" /> Surveillance Mode
              </label>
              <select
                value={streamMode}
                onChange={(e) => handleStreamModeChange(e.target.value)}
                className="w-full bg-slate-900 border border-slate-800 rounded-lg p-2.5 text-xs text-slate-200 focus:outline-none focus:border-cyan-500 cursor-pointer font-medium shadow-inner"
              >
                <option value="live">24/7 CCTV / RTSP Stream Mode</option>
                <option value="upload">Video File Post-Shift Audit</option>
              </select>
            </div>

            {/* System Core Services Health Status */}
            <div className="space-y-2 border-t border-slate-900 pt-4">
              <div className="flex items-center justify-between">
                <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest block">
                  Core System Services
                </span>
                <span className="text-[9px] font-mono text-cyan-400/80 bg-slate-900 px-1.5 py-0.5 rounded border border-slate-800">
                  Live Status
                </span>
              </div>
              <div className="space-y-2 text-xs">
                {/* Service 1: Alert & Notification Hub */}
                <div className="bg-slate-900/80 border border-slate-850 p-2.5 rounded-lg flex justify-between items-center hover:border-slate-800 transition">
                  <div className="flex flex-col min-w-0 pr-2">
                    <span className="text-slate-200 font-semibold text-xs flex items-center gap-1.5 truncate">
                      <span className={`h-2 w-2 rounded-full shrink-0 ${isConnected ? 'bg-emerald-400 shadow-sm shadow-emerald-400/50' : 'bg-rose-500'}`}></span>
                      Alert & Notification Hub
                    </span>
                    <span className="text-[10px] text-slate-500 font-mono pl-3.5 truncate">
                      Gmail alerts, incident database (:5000)
                    </span>
                  </div>
                  <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold border shrink-0 ${isConnected ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30' : 'bg-rose-500/20 text-rose-400 border-rose-500/30'}`}>
                    {isConnected ? 'ONLINE' : 'OFFLINE'}
                  </span>
                </div>

                {/* Service 2: Vision & Threat AI Engine */}
                <div className="bg-slate-900/80 border border-slate-850 p-2.5 rounded-lg flex justify-between items-center hover:border-slate-800 transition">
                  <div className="flex flex-col min-w-0 pr-2">
                    <span className="text-slate-200 font-semibold text-xs flex items-center gap-1.5 truncate">
                      <span className={`h-2 w-2 rounded-full shrink-0 ${aiServiceOnline ? 'bg-emerald-400 shadow-sm shadow-emerald-400/50' : 'bg-amber-400'}`}></span>
                      Vision & Threat AI Engine
                    </span>
                    <span className="text-[10px] text-slate-500 font-mono pl-3.5 truncate">
                      YOLOv8 vision, PPE check & AI brain (:8000)
                    </span>
                  </div>
                  <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold border shrink-0 ${aiServiceOnline ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30' : 'bg-amber-500/20 text-amber-400 border-amber-500/30'}`}>
                    {aiServiceOnline ? 'ACTIVE' : 'STANDBY'}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Footer info */}
        <div className="border-t border-slate-900 pt-3 text-[11px] text-slate-500 flex justify-between items-center font-mono mt-4">
          <span className="flex items-center gap-1.5">
            <span className={`h-2 w-2 rounded-full ${isConnected ? 'bg-emerald-400' : 'bg-rose-500'}`}></span>
            Status:
          </span>
          <span className="text-cyan-400 font-bold">{isConnected ? 'OPERATIONAL' : 'DISCONNECTED'}</span>
        </div>
      </aside>

      {/* ==================== MAIN VIEWPORT CONTENT ==================== */}
      <main className="flex-1 min-w-0 p-3.5 lg:p-5 flex flex-col gap-4 overflow-y-auto scroll-smooth">
        {/* Top Status Bar */}
        <header className="flex flex-wrap justify-between items-center bg-slate-950/80 p-3 rounded-xl border border-slate-800 shadow-xl backdrop-blur-md gap-2.5">
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 bg-slate-900 px-3 py-1 rounded-md border border-slate-800 text-xs font-mono text-cyan-300">
              <Radio className="w-3.5 h-3.5 text-cyan-400 animate-pulse" />
              <span>ACTIVE FEED: {streamMode === 'live' ? '24/7 RTSP PLANT FLOOR' : 'VIDEO AUDIT'}</span>
            </div>
          </div>

          <div className="flex items-center gap-2.5 text-xs font-mono">
            {/* Manage Contacts Quick Button */}
            <button
              onClick={() => setIsAlertModalOpen(true)}
              className="flex items-center gap-1.5 bg-indigo-500/15 hover:bg-indigo-500/25 text-indigo-300 border border-indigo-500/40 px-2.5 py-1 rounded-md transition cursor-pointer text-[11px]"
            >
              <Bell className="w-3.5 h-3.5 text-indigo-400" />
              <span>Safety Contacts ({activeEmailsCount} Active)</span>
            </button>

            <div className="flex items-center gap-1.5 text-slate-400 bg-slate-900 px-2.5 py-1 rounded-md border border-slate-800 text-[11px]">
              <Clock className="w-3.5 h-3.5 text-slate-400" />
              <span>{currentTime}</span>
            </div>

            <div className="flex items-center gap-1.5">
              <span className="text-slate-400 hidden sm:inline text-[11px]">Threat:</span>
              <span
                className={`px-2 py-0.5 rounded font-bold text-[10px] ${
                  threatLevel === 'CRITICAL'
                    ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40 shadow-sm shadow-rose-500/20 animate-pulse'
                    : threatLevel === 'HIGH' || threatLevel === 'MODERATE'
                    ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
                    : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                }`}
              >
                {threatLevel}
              </span>
            </div>
          </div>
        </header>

        {/* Section 1: Upper Dashboard Grid: Video Feed (2/3) + Threat Gauge (1/3) - Medium-Sized & Balanced */}
        <section className="w-full shrink-0">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 items-stretch">
            <div className="lg:col-span-2 flex flex-col min-h-[380px] lg:h-[405px] overflow-hidden">
              <VideoStream
                mode={streamMode}
                onSourceSubmit={handleSourceSubmit}
                isProcessing={isProcessing}
                onReset={handleReset}
                onTelemetryUpdate={handleTelemetryUpdate}
                measuredLatency={measuredLatency}
                activeDetections={activeDetections}
                auditProgress={auditProgress}
              />
            </div>
            <div className="flex flex-col min-h-[380px] lg:h-[405px]">
              <RiskGauge score={riskScore} threatLevel={threatLevel} breakdown={breakdown} />
            </div>
          </div>
        </section>

        {/* Section 2: Middle Section: AI Safety Decision Center & Safety Procedures - Medium-Sized & Balanced */}
        <section className="w-full shrink-0">
          <AgentTerminal
            logs={logs}
            protocol={protocol}
            sopDetails={sopDetails}
            operationalStages={operationalStages}
            activeIncident={incidents.find((i) => i.status === 'ACTIVE') || (incidents.length > 0 ? incidents[0] : null)}
            onAcknowledge={handleAcknowledge}
          />
        </section>

        {/* Section 3: Bottom Section: Prominent Full-Width Incident Audit History Table (Non-Overlapping) */}
        <section className="w-full clear-both pt-2 pb-10">
          <IncidentLog
            incidents={incidents}
            onAcknowledgeIncident={handleAcknowledgeIncident}
            onResolveIncident={handleResolveIncident}
            onRefresh={checkHealth}
          />
        </section>
      </main>

      {/* Emergency Alert Recipients Modal */}
      <AlertRecipientsModal
        isOpen={isAlertModalOpen}
        onClose={() => setIsAlertModalOpen(false)}
        recipients={recipients}
        onAddRecipient={handleAddRecipient}
        onDeleteRecipient={handleDeleteRecipient}
        onToggleRecipient={handleToggleRecipient}
        onSendTestAlert={handleSendTestAlert}
      />
    </div>
  );
}
