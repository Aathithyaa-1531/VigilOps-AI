import React, { useState } from 'react';
import {
  Bell,
  Mail,
  Plus,
  Trash2,
  X,
  CheckCircle2,
  AlertTriangle,
  Send,
  ShieldCheck,
  UserCheck,
  RefreshCw,
  ExternalLink,
  Info
} from 'lucide-react';

export default function AlertRecipientsModal({
  isOpen,
  onClose,
  recipients = [],
  onAddRecipient,
  onDeleteRecipient,
  onToggleRecipient,
  onSendTestAlert,
  lastDispatchResult
}) {
  const [name, setName] = useState('');
  const [role, setRole] = useState('EHS Safety Officer');
  const [email, setEmail] = useState('');
  const [emailEnabled, setEmailEnabled] = useState(true);
  const [errorMsg, setErrorMsg] = useState('');
  const [isSendingTest, setIsSendingTest] = useState(false);
  const [testResult, setTestResult] = useState(null);

  if (!isOpen) return null;

  const handleSubmit = (e) => {
    e.preventDefault();
    setErrorMsg('');

    if (!email.trim()) {
      setErrorMsg('Please specify a valid safety administrator email address.');
      return;
    }

    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email.trim())) {
      setErrorMsg('Please enter a valid email address (e.g., officer@plant.com).');
      return;
    }

    onAddRecipient({
      name: name.trim() || 'Safety Administrator',
      role,
      email: email.trim().toLowerCase(),
      emailEnabled
    });

    // Reset form
    setName('');
    setEmail('');
    setErrorMsg('');
  };

  const handleTestClick = async () => {
    setIsSendingTest(true);
    setTestResult(null);
    try {
      const result = await onSendTestAlert();
      setTestResult(result);
    } catch (e) {
      console.error(e);
      setTestResult({ error: e.message });
    } finally {
      setIsSendingTest(false);
    }
  };

  const activeEmailsCount = recipients.filter((r) => r.email && r.emailEnabled).length;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md animate-fade-in">
      <div className="bg-slate-950 border border-slate-800 rounded-2xl w-full max-w-2xl shadow-2xl overflow-hidden flex flex-col max-h-[90vh]">
        {/* Modal Header */}
        <div className="flex justify-between items-center px-6 py-4 border-b border-slate-850 bg-slate-900/50">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-indigo-500/10 border border-indigo-500/30 rounded-xl text-indigo-400">
              <Mail className="w-5 h-5 animate-pulse" />
            </div>
            <div>
              <h2 className="text-base font-bold text-slate-100 flex items-center gap-2">
                Gmail SMTP Emergency Alert Network
                <span className="text-[10px] font-mono font-bold bg-indigo-500/10 text-indigo-400 border border-indigo-500/30 px-2 py-0.5 rounded-full">
                  GMAIL SMTP EXCLUSIVE
                </span>
              </h2>
              <p className="text-xs text-slate-400">
                Configure safety administrators notified automatically via Gmail when a CRITICAL hazard is confirmed
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 text-slate-400 hover:text-slate-100 hover:bg-slate-800 rounded-lg transition"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-6 overflow-y-auto space-y-6">
          {/* Quick Stats Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-xl flex items-center gap-3">
              <div className="p-2 bg-cyan-500/10 rounded-lg text-cyan-400">
                <UserCheck className="w-5 h-5" />
              </div>
              <div>
                <span className="text-xs text-slate-400 font-medium">Registered Safety Officers</span>
                <div className="text-lg font-bold font-mono text-slate-100">{recipients.length} Personnel</div>
              </div>
            </div>

            <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-xl flex items-center gap-3">
              <div className="p-2 bg-indigo-500/10 rounded-lg text-indigo-400">
                <Mail className="w-5 h-5" />
              </div>
              <div>
                <span className="text-xs text-slate-400 font-medium">Active Gmail Alert Channels</span>
                <div className="text-lg font-bold font-mono text-indigo-300">{activeEmailsCount} Active</div>
              </div>
            </div>
          </div>

          {/* Form: Add Emergency Contact */}
          <div className="bg-slate-900/60 border border-slate-800 p-4 rounded-xl space-y-3">
            <h3 className="text-xs font-bold uppercase tracking-wider text-cyan-400 flex items-center gap-1.5">
              <Plus className="w-4 h-4" /> Add Safety Administrator
            </h3>

            {errorMsg && (
              <div className="p-2.5 bg-rose-500/10 border border-rose-500/30 text-rose-300 rounded-lg text-xs flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 shrink-0" />
                <span>{errorMsg}</span>
              </div>
            )}

            <form onSubmit={handleSubmit} className="space-y-3">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <div>
                  <label className="text-[11px] font-semibold text-slate-400 mb-1 block">
                    Full Name & Title
                  </label>
                  <input
                    type="text"
                    placeholder="e.g. Lead Safety Officer"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-cyan-500"
                  />
                </div>

                <div>
                  <label className="text-[11px] font-semibold text-slate-400 mb-1 block">
                    Department / Role
                  </label>
                  <select
                    value={role}
                    onChange={(e) => setRole(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-cyan-500"
                  >
                    <option value="EHS Safety Officer">EHS Safety Officer</option>
                    <option value="Floor Operations Supervisor">Floor Operations Supervisor</option>
                    <option value="Plant Director / Management">Plant Director / Management</option>
                    <option value="Site Medic & Emergency Response">Site Medic & Emergency Response</option>
                    <option value="Security Dispatch Control">Security Dispatch Control</option>
                  </select>
                </div>
              </div>

              <div>
                <label className="text-[11px] font-semibold text-slate-400 mb-1 block flex items-center gap-1">
                  <Mail className="w-3.5 h-3.5 text-indigo-400" /> Gmail / Email Address
                </label>
                <input
                  type="email"
                  placeholder="safety.officer@gmail.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-cyan-500 font-mono"
                />
              </div>

              <div className="flex flex-wrap items-center justify-between gap-3 pt-1 border-t border-slate-850">
                <label className="flex items-center gap-2 cursor-pointer text-slate-300 select-none text-xs">
                  <input
                    type="checkbox"
                    checked={emailEnabled}
                    onChange={(e) => setEmailEnabled(e.target.checked)}
                    className="rounded bg-slate-950 border-slate-700 text-indigo-500 focus:ring-0 cursor-pointer"
                  />
                  <span>Enable Email Notifications for this recipient</span>
                </label>

                <button
                  type="submit"
                  className="bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold px-4 py-1.5 rounded-lg text-xs transition cursor-pointer flex items-center gap-1.5 shadow-md shadow-cyan-500/20"
                >
                  <Plus className="w-4 h-4" />
                  Add Recipient
                </button>
              </div>
            </form>
          </div>

          {/* Configured Recipients Table */}
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400">
                Configured Personnel Matrix ({recipients.length})
              </h3>
              <span className="text-[11px] text-slate-500 font-mono">
                Persisted in Central Registry
              </span>
            </div>

            <div className="space-y-2 max-h-56 overflow-y-auto pr-1">
              {recipients.length === 0 ? (
                <div className="text-center py-8 text-slate-500 text-xs italic bg-slate-900/40 rounded-xl border border-slate-800">
                  No alert recipients configured yet. Add a safety administrator email above.
                </div>
              ) : (
                recipients.map((rcpt) => (
                  <div
                    key={rcpt.id}
                    className="bg-slate-900/80 border border-slate-800 p-3 rounded-xl flex items-center justify-between gap-4 hover:border-slate-700 transition"
                  >
                    <div className="space-y-1">
                      <div className="flex items-center gap-2">
                        <span className="font-semibold text-slate-100 text-xs">{rcpt.name}</span>
                        <span className="text-[10px] bg-slate-800 text-cyan-300 px-2 py-0.5 rounded border border-slate-700">
                          {rcpt.role}
                        </span>
                      </div>
                      <div className="flex items-center gap-2 text-[11px] font-mono text-slate-400">
                        <Mail className="w-3 h-3 text-indigo-400" />
                        <span>{rcpt.email}</span>
                      </div>
                    </div>

                    <div className="flex items-center gap-2">
                      {/* Email Toggle */}
                      <button
                        type="button"
                        onClick={() => onToggleRecipient(rcpt.id, { emailEnabled: !rcpt.emailEnabled })}
                        className={`px-2 py-0.5 rounded text-[10px] font-mono font-semibold border transition cursor-pointer ${
                          rcpt.emailEnabled
                            ? 'bg-indigo-500/20 text-indigo-300 border-indigo-500/40'
                            : 'bg-slate-800 text-slate-500 border-slate-700'
                        }`}
                      >
                        Email: {rcpt.emailEnabled ? 'ACTIVE' : 'MUTED'}
                      </button>

                      {/* Delete */}
                      <button
                        type="button"
                        onClick={() => onDeleteRecipient(rcpt.id)}
                        className="p-1.5 text-slate-500 hover:text-rose-400 hover:bg-rose-500/10 rounded-lg transition"
                        title="Remove contact"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Test Alert Dispatch Bar (Requirement 6: Provide a Send Test Email button) */}
          <div className="bg-indigo-950/30 border border-indigo-500/30 p-4 rounded-xl space-y-3">
            <div className="flex flex-wrap justify-between items-center gap-3">
              <div>
                <h4 className="text-xs font-bold text-indigo-300 flex items-center gap-1.5">
                  <Send className="w-3.5 h-3.5 text-indigo-400" /> Verify Gmail SMTP Dispatch Pipeline
                </h4>
                <p className="text-[11px] text-slate-400 mt-0.5">
                  Send a simulated test verification email to {activeEmailsCount} active safety administrator(s).
                </p>
              </div>

              <button
                type="button"
                onClick={handleTestClick}
                disabled={isSendingTest || activeEmailsCount === 0}
                className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-bold px-4 py-2 rounded-lg text-xs transition cursor-pointer flex items-center gap-2 shadow-lg shadow-indigo-600/30 shrink-0"
              >
                {isSendingTest ? (
                  <>
                    <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                    <span>Dispatching via Gmail...</span>
                  </>
                ) : (
                  <>
                    <Send className="w-3.5 h-3.5" />
                    <span>Send Test Email</span>
                  </>
                )}
              </button>
            </div>

            {/* Test Result Display */}
            {testResult && (
              <div className="bg-slate-950 p-3 rounded-lg border border-slate-800 text-xs font-mono space-y-2 mt-2">
                <div className="text-slate-300 font-bold flex items-center gap-2">
                  <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                  <span className="text-indigo-300">
                    [TEST EMAIL DRILL] Incident Reference: {testResult.result?.incidentId || 'DRILL-01'}
                  </span>
                </div>
                <div className="text-[10px] text-amber-400/90 font-sans italic">
                  Note: Clearly distinguished as a TEST drill alert. Gmail SMTP acceptance confirms message queueing (SENT).
                </div>

                {/* Email Delivery Details */}
                {testResult.result?.emailDelivery && testResult.result.emailDelivery.length > 0 && (
                  <div className="space-y-1 pt-1 border-t border-slate-900">
                    <span className="text-[10px] text-indigo-400 font-bold uppercase">Gmail SMTP Dispatch Result:</span>
                    {testResult.result.emailDelivery.map((ed, i) => (
                      <div key={i} className="text-[11px] text-slate-300 pl-2 flex flex-wrap items-center gap-2">
                        <span>{ed.recipient} ({ed.email}):</span>
                        <span className={`px-2 py-0.5 rounded-full font-bold text-[10px] border ${
                          ed.status === 'SENT'
                            ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/30'
                            : ed.status === 'MOCKED'
                            ? 'bg-purple-500/20 text-purple-300 border-purple-500/30'
                            : ed.status === 'CONFIG_REQUIRED'
                            ? 'bg-amber-500/20 text-amber-300 border-amber-500/30'
                            : 'bg-rose-500/20 text-rose-300 border-rose-500/30'
                        }`}>
                          {ed.status}
                        </span>
                        {ed.messageId && <span className="text-slate-400 text-[10px]">ID: {ed.messageId}</span>}
                        {ed.timestamp && <span className="text-slate-500 text-[10px]">Time: {ed.timestamp}</span>}
                        {ed.reason && <span className="text-rose-400 text-[10px]">({ed.reason})</span>}
                        {ed.providerResponse && <span className="text-slate-400 text-[10px]">({ed.providerResponse})</span>}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Modal Footer */}
        <div className="px-6 py-3 border-t border-slate-850 bg-slate-900/50 flex justify-between items-center text-xs text-slate-400">
          <span className="flex items-center gap-1.5 font-mono text-[11px]">
            <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
            VigilOps AI — Gmail SMTP Exclusive Safety Alerting Active
          </span>
          <button
            onClick={onClose}
            className="bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold px-4 py-1.5 rounded-lg text-xs transition cursor-pointer"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
