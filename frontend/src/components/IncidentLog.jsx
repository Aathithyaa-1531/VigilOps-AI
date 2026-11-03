import React, { useState } from 'react';
import {
  History,
  Camera,
  Search,
  Filter,
  CheckCircle2,
  Clock,
  AlertTriangle,
  Mail,
  Phone,
  Check,
  X,
  FileText,
  RotateCcw
} from 'lucide-react';

export default function IncidentLog({
  incidents = [],
  onAcknowledgeIncident,
  onResolveIncident,
  onRefresh
}) {
  const [searchTerm, setSearchTerm] = useState('');
  const [levelFilter, setLevelFilter] = useState('ALL');
  const [statusFilter, setStatusFilter] = useState('ALL');
  const [selectedIncident, setSelectedIncident] = useState(null);
  const [actionType, setActionType] = useState('ACKNOWLEDGE'); // 'ACKNOWLEDGE' or 'RESOLVE'
  const [officerName, setOfficerName] = useState('Safety Officer On Duty');
  const [actionNotes, setActionNotes] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Filter incidents locally
  const filteredIncidents = incidents.filter((inc) => {
    if (levelFilter !== 'ALL' && inc.level !== levelFilter) return false;
    if (statusFilter !== 'ALL' && inc.status !== statusFilter) return false;
    if (searchTerm.trim()) {
      const q = searchTerm.toLowerCase();
      const matchId = inc.id?.toLowerCase().includes(q);
      const matchSource = inc.source?.toLowerCase().includes(q);
      const matchHazard = inc.hazard?.toLowerCase().includes(q);
      const matchSop = inc.sop_id?.toLowerCase().includes(q);
      if (!matchId && !matchSource && !matchHazard && !matchSop) return false;
    }
    return true;
  });

  const handleOpenActionModal = (incident, type) => {
    setSelectedIncident(incident);
    setActionType(type);
    setActionNotes(
      type === 'ACKNOWLEDGE'
        ? 'Supervisor verified perimeter and acknowledged SOP.'
        : 'Hazard mitigated and sector cleared for routine operation.'
    );
  };

  const handleConfirmAction = async () => {
    if (!selectedIncident) return;
    setIsSubmitting(true);
    try {
      if (actionType === 'ACKNOWLEDGE' && onAcknowledgeIncident) {
        await onAcknowledgeIncident(selectedIncident.id, {
          officerName: officerName.trim() || 'Safety Officer',
          notes: actionNotes.trim()
        });
      } else if (actionType === 'RESOLVE' && onResolveIncident) {
        await onResolveIncident(selectedIncident.id, {
          officerName: officerName.trim() || 'Safety Officer',
          notes: actionNotes.trim()
        });
      }
      setSelectedIncident(null);
    } catch (err) {
      console.error(err);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="w-full bg-slate-950/90 rounded-xl border border-slate-800 p-3.5 lg:p-4 flex flex-col font-sans text-xs shadow-xl backdrop-blur-md">
      {/* Top Header & Search/Filter Controls Bar */}
      <div className="flex flex-wrap justify-between items-center pb-2.5 border-b border-slate-900 mb-2.5 gap-2.5">
        <div className="flex items-center gap-2">
          <History className="w-3.5 h-3.5 text-cyan-400" />
          <h3 className="font-bold text-slate-100 tracking-wider text-xs uppercase flex items-center gap-2">
            <span>Incident Audit History</span>
            <span className="text-[10px] font-mono font-normal bg-slate-900 text-slate-400 border border-slate-800 px-2 py-0.5 rounded-full">
              {filteredIncidents.length} of {incidents.length} Records (Accessible)
            </span>
          </h3>
        </div>

        {/* Search & Filters */}
        <div className="flex flex-wrap items-center gap-2">
          {/* Search Box */}
          <div className="relative">
            <Search className="w-3.5 h-3.5 text-slate-500 absolute left-2.5 top-2" />
            <input
              type="text"
              placeholder="Search ID, zone, hazard, SOP..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="bg-slate-900 border border-slate-800 rounded-lg pl-8 pr-3 py-1 text-[11px] text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500 w-48 font-mono"
            />
          </div>

          {/* Severity Filter */}
          <select
            value={levelFilter}
            onChange={(e) => setLevelFilter(e.target.value)}
            className="bg-slate-900 border border-slate-800 rounded-lg px-2.5 py-1 text-[11px] text-slate-300 focus:outline-none focus:border-cyan-500 cursor-pointer font-mono"
          >
            <option value="ALL">All Threat Levels</option>
            <option value="CRITICAL">Critical Only</option>
            <option value="MODERATE">Moderate Only</option>
            <option value="LOW">Low Only</option>
          </select>

          {/* Status Filter */}
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="bg-slate-900 border border-slate-800 rounded-lg px-2.5 py-1 text-[11px] text-slate-300 focus:outline-none focus:border-cyan-500 cursor-pointer font-mono"
          >
            <option value="ALL">All Statuses</option>
            <option value="ACTIVE">Active Alerts</option>
            <option value="ACKNOWLEDGED">Acknowledged</option>
            <option value="RESOLVED">Resolved</option>
          </select>

          {onRefresh && (
            <button
              onClick={onRefresh}
              className="p-1 bg-slate-900 hover:bg-slate-800 border border-slate-800 rounded-lg text-slate-400 hover:text-slate-200 transition"
              title="Refresh incident log"
            >
              <RotateCcw className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>

      {/* Main Table Viewport - Comfortable Medium Height with Non-Clipping Horizontal & Vertical Scrolling */}
      <div className="w-full overflow-x-auto max-h-[300px] overflow-y-auto rounded-lg border border-slate-900 bg-slate-950/40 scrollbar-thin scrollbar-thumb-slate-800">
        <table className="w-full text-left border-collapse min-w-[960px]">
          <thead className="sticky top-0 z-10 bg-slate-900/95 backdrop-blur-sm shadow-sm">
            <tr className="border-b border-slate-800 text-[10px] uppercase font-mono text-slate-400">
              <th className="p-2">Incident ID</th>
              <th className="p-2">Time</th>
              <th className="p-2">Zone / Camera</th>
              <th className="p-2">Hazard</th>
              <th className="p-2">Severity</th>
              <th className="p-2">Score</th>
              <th className="p-2">OSHA Code</th>
              <th className="p-2">Email Alert Dispatch</th>
              <th className="p-2">Status</th>
              <th className="p-2 text-right">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-900/80 font-mono text-[11px]">
            {filteredIncidents.length === 0 ? (
              <tr>
                <td colSpan="10" className="text-center py-8 text-slate-500 font-sans">
                  No incident records found matching current query or filters.
                </td>
              </tr>
            ) : (
              filteredIncidents.map((inc) => {
                const isCritical = inc.level === 'CRITICAL';
                const isAcknowledged = inc.status === 'ACKNOWLEDGED';
                const isResolved = inc.status === 'RESOLVED';

                // Exclusive Gmail SMTP notification status
                const emailStatus = inc.notification?.email?.status || (inc.notification?.email?.sentCount > 0 ? 'SENT' : (inc.notification?.email?.configRequired ? 'CONFIG_REQUIRED' : 'OFF'));
                const emailSent = emailStatus === 'SENT';
                const emailMocked = emailStatus === 'MOCKED';
                const emailFailed = emailStatus === 'FAILED';
                const emailConfigReq = emailStatus === 'CONFIG_REQUIRED';
                const emailSkipped = emailStatus === 'SKIPPED';
                const recipientEmail = inc.notification?.email?.emailAddress || inc.notification?.email?.recipient || 'Safety Administrator';
                const sendingTime = inc.notification?.email?.timestamp || inc.formattedTime || '--:--';
                const errorDetail = inc.notification?.email?.error || inc.notification?.email?.reason || '';
                const providerMsg = inc.notification?.email?.providerResponse || '';

                return (
                  <tr key={inc.id} className="hover:bg-slate-900/40 transition-colors text-slate-300">
                    <td className="p-2.5 font-bold text-cyan-400 whitespace-nowrap">{inc.id}</td>
                    <td className="p-2.5 text-slate-400 whitespace-nowrap">
                      {inc.formattedTime || (inc.timestamp ? new Date(inc.timestamp).toLocaleTimeString() : '--:--')}
                    </td>
                    <td className="p-2.5 text-slate-300 flex items-center gap-1.5 whitespace-nowrap">
                      <Camera className="w-3 h-3 text-slate-500 shrink-0" />
                      <span className="truncate max-w-[140px]" title={inc.source}>{inc.source}</span>
                    </td>
                    <td className="p-2.5 font-semibold text-rose-300 whitespace-nowrap">{inc.hazard}</td>
                    <td className="p-2.5 whitespace-nowrap">
                      <span
                        className={`px-2 py-0.5 rounded text-[9px] font-bold border ${
                          isCritical
                            ? 'bg-rose-500/20 text-rose-300 border-rose-500/40'
                            : inc.level === 'MODERATE'
                            ? 'bg-amber-500/20 text-amber-300 border-amber-500/40'
                            : 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                        }`}
                      >
                        {inc.level}
                      </span>
                    </td>
                    <td className="p-2.5 font-bold text-slate-200 whitespace-nowrap">
                      {Math.round(Number(inc.score || 0.85) * 100)}%
                    </td>
                    <td className="p-2.5 text-amber-300 font-semibold whitespace-nowrap">
                      {inc.standard_code || inc.sop_id || '29 CFR 1910'}
                    </td>
                    <td className="p-2.5 whitespace-nowrap">
                      <span
                        className={`px-2 py-0.5 rounded border inline-flex items-center gap-1.5 font-mono text-[9px] ${
                          emailSent
                            ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40 font-semibold'
                            : emailMocked
                            ? 'bg-purple-500/20 text-purple-300 border-purple-500/40 font-semibold'
                            : emailFailed
                            ? 'bg-rose-500/20 text-rose-300 border-rose-500/40 font-semibold'
                            : emailConfigReq
                            ? 'bg-amber-500/20 text-amber-300 border-amber-500/40'
                            : emailSkipped
                            ? 'bg-slate-800 text-slate-400 border-slate-700'
                            : 'bg-slate-800 text-slate-500 border-slate-700'
                        }`}
                        title={`Recipient: ${recipientEmail} | Time: ${sendingTime}${errorDetail ? ` | Error: ${errorDetail}` : ''}${providerMsg ? ` | Provider: ${providerMsg}` : ''}`}
                      >
                        <Mail className="w-2.5 h-2.5" />
                        <span>{emailSent ? 'SENT' : emailMocked ? 'MOCKED' : emailFailed ? 'FAILED' : emailConfigReq ? 'CONFIG REQ' : emailSkipped ? 'SKIPPED' : 'OFF'}</span>
                      </span>
                    </td>
                    <td className="p-2.5 whitespace-nowrap">
                      <span
                        className={`inline-flex items-center gap-1 text-[10px] px-2 py-0.5 rounded border font-sans font-medium ${
                          isResolved
                            ? 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30'
                            : isAcknowledged
                            ? 'text-cyan-300 bg-cyan-500/10 border-cyan-500/30'
                            : 'text-rose-300 bg-rose-500/10 border-rose-500/30 animate-pulse'
                        }`}
                      >
                        {isResolved ? (
                          <>
                            <CheckCircle2 className="w-3 h-3 text-emerald-400" /> Resolved
                          </>
                        ) : isAcknowledged ? (
                          <>
                            <Check className="w-3 h-3 text-cyan-400" /> Acknowledged
                          </>
                        ) : (
                          <>
                            <AlertTriangle className="w-3 h-3 text-rose-400" /> Active Alert
                          </>
                        )}
                      </span>
                    </td>
                    <td className="p-2.5 text-right whitespace-nowrap">
                      <div className="flex items-center justify-end gap-1.5 font-sans">
                        {!isAcknowledged && !isResolved && (
                          <button
                            type="button"
                            onClick={() => handleOpenActionModal(inc, 'ACKNOWLEDGE')}
                            className="bg-cyan-600 hover:bg-cyan-500 text-slate-950 font-bold px-2 py-1 rounded text-[10px] transition cursor-pointer shadow-sm"
                            title="Acknowledge alert as supervisor"
                          >
                            Acknowledge
                          </button>
                        )}
                        {!isResolved && (
                          <button
                            type="button"
                            onClick={() => handleOpenActionModal(inc, 'RESOLVE')}
                            className="bg-emerald-600 hover:bg-emerald-500 text-white font-bold px-2 py-1 rounded text-[10px] transition cursor-pointer shadow-sm"
                            title="Mark incident as resolved"
                          >
                            Resolve
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      {/* Incident Action Confirmation Modal */}
      {selectedIncident && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-fade-in">
          <div className="bg-slate-950 border border-slate-800 rounded-xl w-full max-w-md p-5 shadow-2xl space-y-4 font-sans">
            <div className="flex justify-between items-center border-b border-slate-850 pb-3">
              <h4 className="text-sm font-bold text-slate-100 flex items-center gap-2">
                <FileText className="w-4 h-4 text-cyan-400" />
                <span>
                  {actionType === 'ACKNOWLEDGE' ? 'Supervisor Acknowledgment' : 'Mark Incident Resolved'}
                </span>
              </h4>
              <button
                onClick={() => setSelectedIncident(null)}
                className="text-slate-400 hover:text-slate-200"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="space-y-2 text-xs">
              <div className="bg-slate-900/80 p-2.5 rounded-lg border border-slate-800 font-mono space-y-1">
                <div className="text-slate-400">Incident: <strong className="text-cyan-300">{selectedIncident.id}</strong></div>
                <div className="text-slate-400">Hazard: <strong className="text-rose-300">{selectedIncident.hazard}</strong></div>
                <div className="text-slate-400">Location: {selectedIncident.source}</div>
              </div>

              <div>
                <label className="text-[11px] font-semibold text-slate-400 block mb-1">
                  Supervisor / Officer Name:
                </label>
                <input
                  type="text"
                  value={officerName}
                  onChange={(e) => setOfficerName(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-cyan-500"
                />
              </div>

              <div>
                <label className="text-[11px] font-semibold text-slate-400 block mb-1">
                  Resolution / Log Notes:
                </label>
                <textarea
                  rows="3"
                  value={actionNotes}
                  onChange={(e) => setActionNotes(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-800 rounded-lg p-2.5 text-xs text-slate-200 focus:outline-none focus:border-cyan-500 resize-none font-mono"
                />
              </div>
            </div>

            <div className="flex justify-end gap-2 pt-2 border-t border-slate-850">
              <button
                type="button"
                onClick={() => setSelectedIncident(null)}
                className="bg-slate-900 hover:bg-slate-800 text-slate-300 px-3 py-1.5 rounded-lg text-xs transition"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleConfirmAction}
                disabled={isSubmitting}
                className={`font-bold px-4 py-1.5 rounded-lg text-xs transition text-white ${
                  actionType === 'ACKNOWLEDGE'
                    ? 'bg-cyan-600 hover:bg-cyan-500'
                    : 'bg-emerald-600 hover:bg-emerald-500'
                }`}
              >
                {isSubmitting ? 'Recording...' : `Confirm ${actionType === 'ACKNOWLEDGE' ? 'Acknowledgment' : 'Resolution'}`}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
