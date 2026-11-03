import React from 'react';
import { Activity, AlertOctagon, ShieldAlert, ShieldCheck, Clock, Navigation, AlertTriangle } from 'lucide-react';

export default function RiskGauge({ score = 0, threatLevel = 'LOW', breakdown = {} }) {
  const percentage = Math.min(Math.max(Math.round((score || 0) * 100), 0), 100);
  
  const isCritical = threatLevel === 'CRITICAL' || percentage > 75;
  const isModerate = threatLevel === 'MODERATE' || (percentage > 40 && percentage <= 75);

  const statusColor = isCritical
    ? 'text-rose-500'
    : isModerate
    ? 'text-amber-400'
    : 'text-emerald-400';

  const statusBg = isCritical
    ? 'bg-rose-500/20 text-rose-300 border-rose-500/40 shadow-rose-500/20'
    : isModerate
    ? 'bg-amber-500/20 text-amber-300 border-amber-500/40 shadow-amber-500/20'
    : 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40 shadow-emerald-500/20';

  const strokeColor = isCritical
    ? '#f43f5e'
    : isModerate
    ? '#fbbf24'
    : '#10b981';

  // SVG Gauge calculations
  const strokeDasharray = 251.2; // Circumference of r=40
  const strokeDashoffset = strokeDasharray - (strokeDasharray * percentage) / 100;

  return (
    <div className="flex flex-col justify-between h-full bg-slate-950/80 p-3.5 rounded-xl border border-slate-800 shadow-xl backdrop-blur-md">
      {/* Header */}
      <div className="flex justify-between items-start border-b border-slate-900 pb-2">
        <div>
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
            <Activity className="w-3.5 h-3.5 text-cyan-400" /> Predictive Threat Gauge
          </h3>
          <p className="text-[10px] text-slate-500 mt-0.5 font-mono">
            Scikit-learn RF Multi-Factor Risk Engine
          </p>
        </div>
        <span
          className={`px-2 py-0.5 rounded-md font-mono font-bold text-[10px] border shadow-sm transition-all duration-300 ${statusBg}`}
        >
          {threatLevel || (isCritical ? 'CRITICAL' : isModerate ? 'MODERATE' : 'LOW')}
        </span>
      </div>

      {/* SVG Arc Gauge Centerpiece (Medium & Compact) */}
      <div className="my-1.5 flex flex-col items-center justify-center relative">
        <div className="relative w-28 h-28 flex items-center justify-center">
          {/* Background SVG Circle */}
          <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
            <circle
              cx="50"
              cy="50"
              r="40"
              stroke="#1e293b"
              strokeWidth="10"
              fill="transparent"
            />
            <circle
              cx="50"
              cy="50"
              r="40"
              stroke={strokeColor}
              strokeWidth="10"
              fill="transparent"
              strokeDasharray={strokeDasharray}
              strokeDashoffset={strokeDashoffset}
              strokeLinecap="round"
              className="transition-all duration-700 ease-out"
            />
          </svg>

          {/* Center Gauge Percentage & Icon */}
          <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
            {isCritical ? (
              <AlertOctagon className="w-4 h-4 text-rose-500 animate-bounce mb-0.5" />
            ) : isModerate ? (
              <ShieldAlert className="w-4 h-4 text-amber-400 mb-0.5" />
            ) : (
              <ShieldCheck className="w-4 h-4 text-emerald-400 mb-0.5" />
            )}
            <span className={`text-2xl font-extrabold font-mono tracking-tight ${statusColor}`}>
              {percentage}%
            </span>
            <span className="text-[8px] font-mono uppercase text-slate-500 tracking-wider">
              Danger Score
            </span>
          </div>
        </div>
      </div>

      {/* Contributing Risk Factors Breakdown */}
      <div className="space-y-1.5 border-t border-slate-900 pt-2">
        <div className="text-[10px] uppercase font-bold text-slate-400 tracking-wider flex justify-between">
          <span>Risk Factor Breakdown</span>
          <span className="font-mono text-cyan-400">Multi-Vector</span>
        </div>

        {/* Factor 1: Violation Duration */}
        <div className="space-y-1">
          <div className="flex justify-between text-[11px] text-slate-400">
            <span className="flex items-center gap-1">
              <Clock className="w-3 h-3 text-slate-500" /> Violation Duration
            </span>
            <span className="font-mono font-semibold text-slate-200">
              {Number(breakdown.violation_duration_sec ?? 0).toFixed(1)}s
            </span>
          </div>
          <div className="w-full bg-slate-900 h-1.5 rounded-full overflow-hidden border border-slate-800">
            <div
              className="h-full bg-cyan-500 transition-all duration-500"
              style={{ width: `${Math.min(Math.round(((breakdown.violation_duration_sec ?? 0) / 12.0) * 100), 100)}%` }}
            />
          </div>
        </div>

        {/* Factor 2: Proximity to Hazard */}
        <div className="space-y-1">
          <div className="flex justify-between text-[11px] text-slate-400">
            <span className="flex items-center gap-1">
              <Navigation className="w-3 h-3 text-slate-500" /> Danger Zone Proximity
            </span>
            <span className="font-mono font-semibold text-slate-200">
              {Math.round(Number(breakdown.proximity_score ?? 0) * 100)}%
            </span>
          </div>
          <div className="w-full bg-slate-900 h-1.5 rounded-full overflow-hidden border border-slate-800">
            <div
              className={`h-full transition-all duration-500 ${isCritical ? 'bg-rose-500' : 'bg-amber-500'}`}
              style={{ width: `${Math.min(Math.round(Number(breakdown.proximity_score ?? 0) * 100), 100)}%` }}
            />
          </div>
        </div>

        {/* Factor 3: PPE Severity */}
        <div className="space-y-1">
          <div className="flex justify-between text-[11px] text-slate-400">
            <span className="flex items-center gap-1">
              <AlertTriangle className="w-3 h-3 text-slate-500" /> Hazard Severity Weight
            </span>
            <span className="font-mono font-semibold text-slate-200">
              {Math.round(Number(breakdown.severity_weight ?? (percentage > 0 ? 0.35 : 0)) * 100)}%
            </span>
          </div>
          <div className="w-full bg-slate-900 h-1.5 rounded-full overflow-hidden border border-slate-800">
            <div
              className="h-full bg-indigo-500 transition-all duration-500"
              style={{ width: `${Math.min(Math.round(Number(breakdown.severity_weight ?? (percentage > 0 ? 0.35 : 0)) * 100), 100)}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
