import React, { useState, useRef, useEffect } from 'react';
import {
  ShieldAlert,
  CheckCircle2,
  FileText,
  AlertTriangle,
  Clock,
  UserCheck,
  BookOpen,
  ArrowRight,
  ShieldCheck,
  Check,
  ChevronDown,
  ChevronUp,
  AlertCircle,
  Users
} from 'lucide-react';

export default function AgentTerminal({
  logs = [],
  protocol = '',
  sopDetails = null,
  operationalStages = [],
  activeIncident = null,
  onAcknowledge
}) {
  const logContainerRef = useRef(null);
  const [isHovered, setIsHovered] = useState(false);
  const [isSupervisorActionOpen, setIsSupervisorActionOpen] = useState(false);

  useEffect(() => {
    // Only scroll within the log container itself, NEVER scrolling the window/page.
    if (logContainerRef.current && !isHovered) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logs, operationalStages, isHovered]);

  // Derived structured SOP fields
  const isRetrieved = Boolean(protocol || sopDetails);
  const retrievalStatus = isRetrieved
    ? (sopDetails?.retrieval_status || 'VERIFIED_FROM_CORPUS')
    : 'NONE_RETRIEVED';

  const sopId = sopDetails?.sop_id || (isRetrieved ? 'SOP-EHS-042' : 'STANDBY');
  const standardCode = sopDetails?.standard_code || (isRetrieved ? 'OSHA 1910.135' : '--');
  const regulatoryRef = sopDetails?.regulatory_ref || (isRetrieved ? `29 CFR ${standardCode}` : '--');
  const title = sopDetails?.title || (isRetrieved ? 'Personal Protective Equipment Compliance Protocol' : 'Safety Knowledge Base');
  
  // Requirement: Short, 1-sentence explanation of danger
  const whyDangerous = sopDetails?.why_dangerous || 'Working in an active industrial zone without required safety precautions increases the risk of severe injuries.';
  
  // Requirement: Max 3-4 short, clear immediate corrective actions
  const rawActions = sopDetails?.immediate_actions || sopDetails?.checklist || [
    'Stop the unsafe activity immediately.',
    'Equip the mandatory protective equipment.',
    'Notify the area safety officer before returning to work.'
  ];
  const immediateActions = rawActions.slice(0, 4);

  // Requirement: Max 3-4 short preventive measures
  const rawPreventive = sopDetails?.preventive_measures || [
    'Wear approved safety equipment inside the work area.',
    'Check protective gear daily for damage or wear.',
    'Follow marked safety zones and signs.'
  ];
  const preventiveMeasures = rawPreventive.slice(0, 4);

  // Requirement: Independent Responsible Personnel
  const responsiblePersonnel = sopDetails?.responsible_personnel || 'EHS Safety Officer & Site Supervisor';
  const sourceRef = sopDetails?.reference_source ? `${sopDetails.reference_source} (${sopDetails.reference_section || 'Standard 1910'})` : 'OSHA 29 CFR 1910 Corpus';

  // Associated incident ID
  const associatedIncidentId = sopDetails?.incident_id || activeIncident?.id || null;

  // Acknowledgement tracking - only claim acknowledgement if an actual verified record exists!
  const hasActualAcknowledgement = Boolean(
    (activeIncident && (activeIncident.status === 'ACKNOWLEDGED' || activeIncident.status === 'RESOLVED') && activeIncident.acknowledgedAt) ||
    (sopDetails?.acknowledgedAt && sopDetails?.acknowledgedBy)
  );

  const acknowledgementStatus = hasActualAcknowledgement
    ? 'ACKNOWLEDGED'
    : (retrievalStatus === 'NONE_RETRIEVED' ? 'NONE_RECORDED' : 'PENDING_VERIFICATION');

  const acknowledgedAt = hasActualAcknowledgement
    ? (activeIncident?.acknowledgedAt || sopDetails?.acknowledgedAt)
    : null;

  const acknowledgedBy = hasActualAcknowledgement
    ? (activeIncident?.acknowledgedBy || sopDetails?.acknowledgedBy || 'Safety Officer On Duty')
    : null;

  // Filter consecutive duplicate messages to maintain a clean timeline
  const filteredLogs = logs.filter((msg, idx, arr) => idx === 0 || msg !== arr[idx - 1]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 w-full items-stretch">
      {/* =========================================================================
          PANEL 1: AI SAFETY DECISION CENTER (Medium-Sized & Balanced)
          ========================================================================= */}
      <div className="bg-slate-950/95 rounded-xl border border-slate-800 p-3.5 flex flex-col font-sans text-xs shadow-xl backdrop-blur-md min-h-[380px] lg:h-[405px]">
        <div className="flex justify-between items-center pb-2 border-b border-slate-900 mb-2 shrink-0">
          <div className="flex items-center gap-2">
            <div className="p-1.5 bg-cyan-500/10 border border-cyan-500/30 rounded-lg text-cyan-400">
              <ShieldAlert className="w-3.5 h-3.5 text-cyan-400" />
            </div>
            <div>
              <h3 className="font-bold text-cyan-300 tracking-wider text-xs uppercase flex items-center gap-2">
                <span>AI Safety Decision Center</span>
                <span className="text-[9px] font-mono font-normal bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 px-2 py-0.5 rounded-full">
                  Autonomous Multi-Agent Brain
                </span>
              </h3>
            </div>
          </div>
          <span className={`text-[9px] font-mono px-2 py-0.5 rounded border transition-all ${isHovered ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40' : 'bg-slate-900 text-slate-400 border border-slate-800'}`}>
            {isHovered ? '📍 Holding Scroll' : 'Decision Timeline'}
          </span>
        </div>

        {/* Human-Readable Operational Decision Timeline - Fills remaining panel height with internal scroll */}
        <div 
          ref={logContainerRef}
          onMouseEnter={() => setIsHovered(true)}
          onMouseLeave={() => setIsHovered(false)}
          className="flex-1 min-h-0 overflow-y-auto space-y-1.5 pr-1 scrollbar-thin scrollbar-thumb-slate-800"
        >
          {filteredLogs.length === 0 ? (
            <div className="text-slate-500 italic flex flex-col items-center justify-center h-full py-8 text-center space-y-1.5">
              <Clock className="w-5 h-5 text-slate-700" />
              <span className="text-[11px]">Surveillance active — Multi-agent stages record here in real time.</span>
            </div>
          ) : (
            filteredLogs.map((logMsg, idx) => {
              const isCritical = logMsg.includes('CRITICAL') || logMsg.includes('EMERGENCY') || logMsg.includes('exceeds critical');
              const isWarning = logMsg.includes('WARNING') || logMsg.includes('MODERATE') || logMsg.includes('HIGH');
              const isVerified = logMsg.includes('Verified') || logMsg.includes('Matched') || logMsg.includes('SENT');

              const iconColor = isCritical
                ? 'text-rose-400'
                : isWarning
                ? 'text-amber-400'
                : isVerified
                ? 'text-emerald-400'
                : 'text-cyan-400';

              return (
                <div
                  key={idx}
                  className="bg-slate-900/60 border border-slate-850 hover:border-slate-800 p-2 rounded-lg transition-all flex items-start gap-2"
                >
                  <div className={`mt-0.5 p-1 rounded bg-slate-950 border border-slate-800 ${iconColor} shrink-0`}>
                    {isCritical ? (
                      <AlertTriangle className="w-2.5 h-2.5" />
                    ) : isVerified ? (
                      <CheckCircle2 className="w-2.5 h-2.5" />
                    ) : (
                      <ArrowRight className="w-2.5 h-2.5" />
                    )}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className={`text-[11px] leading-snug font-sans ${isCritical ? 'text-rose-200 font-semibold' : 'text-slate-300'}`}>
                      {logMsg}
                    </p>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>

      {/* =========================================================================
          PANEL 2: SAFETY PROCEDURES & RECOMMENDED ACTIONS (Medium-Sized & Balanced)
          ========================================================================= */}
      <div className="bg-slate-950/95 rounded-xl border border-slate-800 p-3.5 flex flex-col font-sans text-xs shadow-xl backdrop-blur-md min-h-[380px] lg:h-[405px] space-y-2">
        {/* Top Header */}
        <div className="flex justify-between items-center pb-2 border-b border-slate-900 shrink-0">
          <div className="flex items-center gap-2">
            <div className="p-1.5 bg-amber-500/10 border border-amber-500/30 rounded-lg text-amber-400">
              <FileText className="w-3.5 h-3.5 text-amber-400" />
            </div>
            <div>
              <h3 className="font-bold text-amber-300 tracking-wider text-xs uppercase flex items-center gap-2">
                <span>Safety Procedures & Recommended Actions</span>
              </h3>
            </div>
          </div>
          <span className={`text-[9px] font-mono px-2 py-0.5 rounded-full flex items-center gap-1 border ${
            retrievalStatus === 'VERIFIED_FROM_CORPUS'
              ? 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30'
              : 'text-amber-400 bg-amber-500/10 border-amber-500/30'
          }`}>
            <ShieldCheck className="w-3 h-3" />
            <span>{retrievalStatus === 'VERIFIED_FROM_CORPUS' ? 'OSHA VERIFIED' : 'STANDBY'}</span>
          </span>
        </div>

        {/* =====================================================================
            SUPERVISOR ACTION (Compact & Collapsible)
            ===================================================================== */}
        <div className="bg-slate-900/90 border border-slate-800 rounded-lg overflow-hidden transition-all shadow-sm shrink-0">
          {/* Collapsible Header Bar */}
          <div
            onClick={() => setIsSupervisorActionOpen(!isSupervisorActionOpen)}
            className="p-2 flex items-center justify-between cursor-pointer select-none bg-slate-900/60 hover:bg-slate-900/90 transition"
          >
            <div className="flex items-center gap-2 flex-wrap">
              <UserCheck className="w-3.5 h-3.5 text-cyan-400" />
              <span className="font-bold text-[11px] text-slate-200 uppercase tracking-wider">Supervisor Action</span>
              <span className="text-[10px] font-mono text-slate-400 bg-slate-950 px-2 py-0.5 rounded border border-slate-800">
                Incident: <strong className="text-cyan-300">{associatedIncidentId || 'None Active'}</strong>
              </span>
            </div>

            <div className="flex items-center gap-2">
              <span className={`text-[9px] font-mono font-bold px-2 py-0.5 rounded border ${
                retrievalStatus === 'VERIFIED_FROM_CORPUS'
                  ? 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30'
                  : 'bg-amber-500/15 text-amber-300 border-amber-500/30'
              }`}>
                {retrievalStatus === 'VERIFIED_FROM_CORPUS' ? 'VERIFIED' : 'NONE_RETRIEVED'}
              </span>

              <span className={`text-[9px] font-mono font-bold px-2 py-0.5 rounded border ${
                hasActualAcknowledgement
                  ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                  : retrievalStatus === 'NONE_RETRIEVED'
                  ? 'bg-amber-500/15 text-amber-300 border-amber-500/30'
                  : 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40'
              }`}>
                {hasActualAcknowledgement ? 'ACKNOWLEDGED' : (retrievalStatus === 'NONE_RETRIEVED' ? 'HUMAN VERIF REQ' : 'PENDING')}
              </span>

              <button
                type="button"
                className="text-slate-400 hover:text-slate-200 transition p-0.5"
                title={isSupervisorActionOpen ? 'Collapse' : 'Expand'}
              >
                {isSupervisorActionOpen ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
              </button>
            </div>
          </div>

          {/* Collapsible Content Body */}
          {isSupervisorActionOpen && (
            <div className="p-2.5 space-y-2 text-xs bg-slate-950/40 border-t border-slate-850 animate-fade-in">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-[11px]">
                {/* 1. Protocol Retrieval Status */}
                <div className="bg-slate-900/80 p-2 rounded border border-slate-800 space-y-0.5">
                  <span className="text-[9px] font-bold text-slate-400 uppercase tracking-wider block">
                    Protocol Retrieval Status
                  </span>
                  {retrievalStatus === 'VERIFIED_FROM_CORPUS' ? (
                    <div className="flex items-center gap-1.5 text-emerald-300 font-semibold font-mono text-[10px]">
                      <CheckCircle2 className="w-3 h-3 text-emerald-400 shrink-0" />
                      <span>Verified: {standardCode}</span>
                    </div>
                  ) : (
                    <div className="flex items-start gap-1.5 text-amber-300 text-[10px]">
                      <AlertCircle className="w-3 h-3 text-amber-400 shrink-0 mt-0.5" />
                      <div>
                        <div className="font-bold">No verified safety protocol retrieved</div>
                        <div className="text-slate-400 text-[9px]">Human verification required.</div>
                      </div>
                    </div>
                  )}
                </div>

                {/* 2. Safety Officer Acknowledgement Status */}
                <div className="bg-slate-900/80 p-2 rounded border border-slate-800 space-y-0.5">
                  <span className="text-[9px] font-bold text-slate-400 uppercase tracking-wider block">
                    Officer Acknowledgement
                  </span>
                  {hasActualAcknowledgement ? (
                    <div className="space-y-0.5">
                      <div className="flex items-center gap-1.5 text-emerald-300 font-semibold font-mono text-[10px]">
                        <ShieldCheck className="w-3 h-3 text-emerald-400 shrink-0" />
                        <span>Acknowledged by {acknowledgedBy}</span>
                      </div>
                      <div className="text-[9px] font-mono text-slate-400">
                        {acknowledgedAt ? new Date(acknowledgedAt).toLocaleTimeString() : ''}
                      </div>
                    </div>
                  ) : retrievalStatus === 'NONE_RETRIEVED' ? (
                    <div className="text-[10px] text-slate-400 font-mono">
                      <span className="text-amber-400 font-semibold">No record exists</span> (requires verified SOP)
                    </div>
                  ) : (
                    <div className="flex items-center gap-1.5 text-cyan-300 font-semibold font-mono text-[10px]">
                      <Clock className="w-3 h-3 text-cyan-400 shrink-0" />
                      <span>Pending review by officer</span>
                    </div>
                  )}
                </div>
              </div>

              {/* Action controls row */}
              <div className="flex items-center justify-between pt-1 border-t border-slate-850">
                <span className="text-[10px] font-mono text-slate-400">
                  Target: <strong className="text-slate-200">{associatedIncidentId || 'None'}</strong>
                </span>

                <div>
                  {isRetrieved && !hasActualAcknowledgement ? (
                    <button
                      type="button"
                      onClick={onAcknowledge}
                      className="bg-amber-500 hover:bg-amber-400 text-slate-950 font-bold px-3 py-1 rounded text-[10px] transition cursor-pointer flex items-center gap-1.5 shadow-sm"
                    >
                      <Check className="w-3 h-3 stroke-[3]" />
                      <span>Acknowledge Protocol</span>
                    </button>
                  ) : hasActualAcknowledgement ? (
                    <span className="text-[10px] font-mono font-semibold text-emerald-400 flex items-center gap-1">
                      <CheckCircle2 className="w-3 h-3 text-emerald-400" />
                      <span>Verified & Acknowledged</span>
                    </span>
                  ) : null}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Structured SOP Body Details (Content-Driven, Non-Clipping) */}
        <div className="flex-1 min-h-0 bg-slate-900/60 border border-slate-850 rounded-xl p-3 text-slate-300 overflow-y-auto space-y-3 scrollbar-thin scrollbar-thumb-slate-800">
          {isRetrieved ? (
            <div className="space-y-3">
              {/* Header: SOP & Regulation Reference */}
              <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 pb-2">
                <div className="flex items-center gap-2">
                  <span className="font-mono font-bold text-amber-300 bg-amber-500/15 border border-amber-500/30 px-2 py-0.5 rounded text-[10px]">
                    {sopId}
                  </span>
                  <span className="font-mono text-slate-200 font-semibold text-[11px]">
                    {title}
                  </span>
                </div>
                <span className="text-[9px] font-mono text-slate-400 bg-slate-950 border border-slate-800 px-2 py-0.5 rounded">
                  {regulatoryRef}
                </span>
              </div>

              {/* Requirement: WHY THIS IS DANGEROUS (One short sentence in prominent callout) */}
              <div className="bg-rose-950/40 border border-rose-500/30 p-2.5 rounded-lg text-[11px] text-rose-200">
                <div className="font-bold text-rose-300 mb-0.5 flex items-center gap-1.5 uppercase text-[10px] tracking-wider">
                  <AlertTriangle className="w-3 h-3 text-rose-400 shrink-0" />
                  <span>Why this is dangerous</span>
                </div>
                <p className="leading-snug m-0">{whyDangerous}</p>
              </div>

              {/* Requirement: IMMEDIATE CORRECTIVE ACTIONS (Max 3-4 short, clear instructions) */}
              <div className="space-y-1.5">
                <div className="text-[10px] font-bold uppercase tracking-wider text-amber-400 flex items-center gap-1.5">
                  <ShieldAlert className="w-3 h-3 text-amber-400" />
                  <span>Immediate Corrective Actions</span>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-1.5">
                  {immediateActions.map((action, idx) => (
                    <div
                      key={idx}
                      className="bg-slate-950/70 border border-slate-850 p-2 rounded-lg flex items-start gap-2 text-[11px] text-slate-200 leading-snug"
                    >
                      <span className="bg-amber-500/20 text-amber-300 font-mono font-bold text-[10px] px-1.5 py-0.5 rounded shrink-0">
                        {idx + 1}
                      </span>
                      <span>{action}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Requirement: DYNAMIC PREVENTIVE MEASURES & INDEPENDENT RESPONSIBLE PERSONNEL */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-2.5 pt-2 border-t border-slate-800">
                {/* PREVENTIVE MEASURES (2 columns, dynamic, non-truncated list) */}
                <div className="md:col-span-2 bg-slate-950/50 p-2.5 rounded-lg border border-slate-850 space-y-1.5">
                  <div className="text-[10px] font-bold uppercase tracking-wider text-cyan-400 flex items-center gap-1.5">
                    <BookOpen className="w-3 h-3 text-cyan-400" />
                    <span>Preventive Measures</span>
                  </div>
                  <ul className="space-y-1 text-[11px] text-slate-300">
                    {preventiveMeasures.map((measure, idx) => (
                      <li key={idx} className="flex items-start gap-1.5 leading-snug">
                        <span className="text-cyan-400 mt-1 shrink-0">•</span>
                        <span>{measure}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                {/* RESPONSIBLE PERSONNEL (1 column, independent designation) */}
                <div className="md:col-span-1 bg-slate-950/50 p-2.5 rounded-lg border border-slate-850 flex flex-col justify-between space-y-1.5">
                  <div>
                    <div className="text-[10px] font-bold uppercase tracking-wider text-indigo-400 flex items-center gap-1.5 mb-1">
                      <Users className="w-3 h-3 text-indigo-400" />
                      <span>Responsible Personnel</span>
                    </div>
                    <p className="text-[11px] text-slate-200 font-medium leading-snug">
                      {responsiblePersonnel}
                    </p>
                  </div>
                  <div className="text-[9px] font-mono text-slate-500 pt-1 border-t border-slate-800">
                    Role Verification Active
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="h-full flex flex-col items-center justify-center text-slate-500 text-center py-8 space-y-2">
              <FileText className="w-7 h-7 text-slate-700" />
              <span className="text-xs font-semibold text-slate-400">No Verified Safety Protocol Retrieved</span>
              <p className="text-[11px] text-slate-500 max-w-sm">
                No active hazard currently detected. If a workplace violation occurs, simple, action-oriented instructions and verified safety guidelines will populate here.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
