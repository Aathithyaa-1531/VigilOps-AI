require('dotenv').config();

const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const cors = require('cors');

const incidentManager = require('./incidentManager');
const alertService = require('./alertService');

const app = express();
app.use(cors({ origin: '*', methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'] }));
app.use(express.json({ limit: '20mb' }));
app.use(express.urlencoded({ extended: true, limit: '20mb' }));

const server = http.createServer(app);
const io = new Server(server, {
    cors: { origin: '*', methods: ['GET', 'POST'] },
});

const ALERT_COOLDOWN_SECONDS = Number(process.env.ALERT_COOLDOWN_SECONDS) || 60;
const FASTAPI_URL = process.env.FASTAPI_URL || 'http://localhost:8000';

// Global Privacy & Device-Name Sanitizer (Requirement 5)
function sanitizePrivacyData(val) {
    if (typeof val === 'string') {
        return val
            .replace(/AATHITHYAAs-MacBook-Pro(\.local)?/gi, 'Apple Silicon (Edge Node)')
            .replace(/MacBook(\s*Pro)?(\s*M2)?/gi, 'Apple Silicon')
            .replace(/\/Users\/[a-zA-Z0-9._-]+/g, '/opt/vigilops')
            .replace(/\b(aathithyaae)\b/gi, 'operator');
    }
    if (Array.isArray(val)) {
        return val.map(sanitizePrivacyData);
    }
    if (val !== null && typeof val === 'object') {
        const cleaned = {};
        for (const [k, v] of Object.entries(val)) {
            cleaned[k] = sanitizePrivacyData(v);
        }
        return cleaned;
    }
    return val;
}

const originalEmit = io.emit.bind(io);
io.emit = (event, ...args) => {
    const sanitizedArgs = args.map(sanitizePrivacyData);
    return originalEmit(event, ...sanitizedArgs);
};

// Optional API Key middleware for administrative operations
const requireAdminKey = (req, res, next) => {
    const adminKey = process.env.ADMIN_API_KEY;
    if (!adminKey) return next(); // If not configured, allow local access

    const provided = req.headers['x-api-key'] || req.query.apiKey;
    if (provided && provided === adminKey) {
        return next();
    }
    // Allow local development convenience unless explicit header is passed incorrectly
    if (req.ip === '127.0.0.1' || req.ip === '::1' || req.ip === '::ffff:127.0.0.1') {
        return next();
    }
    return res.status(401).json({ error: 'Unauthorized: Valid X-API-KEY header required' });
};

io.on('connection', (socket) => {
    console.log(`🔌 [Matrix Gateway] Client connected: ${socket.id}`);
    
    // Send current incident history and recipients upon connection
    const { incidents } = incidentManager.getIncidents({ limit: 50 });
    socket.emit('initial_incidents', incidents);
    socket.emit('recipients_update', alertService.getRecipients());

    socket.on('disconnect', () => {
        console.log(`🔌 [Matrix Gateway] Client disconnected: ${socket.id}`);
    });
});

// ==================== TELEMETRY METRICS INGESTION ====================
app.post('/api/metrics', async (req, res) => {
    const { source, score, level, protocol, sop_details, logs, hazards, breakdown } = req.body;
    const timestamp = new Date().toLocaleTimeString();

    // 1. Broadcast real-time telemetry to React Dashboard
    io.emit('risk_update', { score, level, breakdown, timestamp });

    if (req.body.operational_stages && Array.isArray(req.body.operational_stages)) {
        io.emit('operational_stages', req.body.operational_stages);
    }

    if (protocol || sop_details) {
        io.emit('protocol_alert', { protocol, sop_details });
    }

    if (logs && Array.isArray(logs)) {
        logs.forEach(log => io.emit('agent_log', log));
    }

    // 2. Incident recording on HIGH or CRITICAL threat level
    const isCritical = level === 'CRITICAL' || Number(score) >= (Number(process.env.CRITICAL_THRESHOLD) || 0.75);
    const isHighOrAbove = isCritical || level === 'HIGH' || Number(score) >= 0.60;

    if (isHighOrAbove) {
        const primaryHazard = hazards && hazards.length > 0 ? hazards[0].label : 'UNSAFE_CONDITION';
        const sourceName = source || 'SURV_CAM_01';

        // Check deduplication cooldown
        const isDuplicate = incidentManager.isDuplicate(
            sourceName,
            primaryHazard,
            level,
            ALERT_COOLDOWN_SECONDS
        );

        if (isDuplicate) {
            console.log(`⏳ [Cooldown] Duplicate hazard '${primaryHazard}' at '${sourceName}' suppressed within ${ALERT_COOLDOWN_SECONDS}s window.`);
            io.emit('agent_log', `[Deduplication Engine] Suppressed duplicate alert for '${primaryHazard}' (active cooldown).`);
            return res.status(200).json({ status: 'deduplicated', reason: 'cooldown_active' });
        }

        // Create persistent incident record
        const incident = incidentManager.createIncident({
            source: sourceName,
            hazard: primaryHazard,
            level: isCritical ? 'CRITICAL' : 'HIGH',
            score: score || (isCritical ? 0.88 : 0.65),
            sop_id: sop_details?.sop_id || 'SOP-EHS-042',
            standard_code: sop_details?.standard_code || '29 CFR 1910.132',
            hazardDescription: protocol || `${primaryHazard} violation detected in active production sector.`,
            breakdown
        });

        io.emit('incident_event', incident);
        io.emit('agent_log', `🚨 [Incident Recorded] Logged ${incident.id} (${incident.level}) in compliance registry.`);

        // Dispatch emergency Gmail SMTP alert for HIGH and CRITICAL severity (Requirement B.2)
        const shouldSendEmail = isCritical || incident.level === 'HIGH';
        if (shouldSendEmail) {
            alertService.dispatchAlert({
                incidentId: incident.id,
                source: incident.source,
                score: incident.score,
                level: incident.level,
                hazard: incident.hazard,
                protocol,
                sop_details,
                isTest: false // Genuine safety alert - never a test email!
            }).then((dispatchResult) => {
                const emailSent = dispatchResult.emailDelivery.some(e => e.status === 'SENT');
                const emailMocked = dispatchResult.emailDelivery.some(e => e.status === 'MOCKED');
                const emailFailed = dispatchResult.emailDelivery.some(e => e.status === 'FAILED');
                const emailConfigReq = dispatchResult.emailDelivery.some(e => e.status === 'CONFIG_REQUIRED');
                const emailSkipped = dispatchResult.emailDelivery.some(e => e.status === 'SKIPPED');
                const emailStatus = emailSent ? 'SENT' : (emailMocked ? 'MOCKED' : (emailFailed ? 'FAILED' : (emailSkipped ? 'SKIPPED' : (emailConfigReq ? 'CONFIG_REQUIRED' : 'OFF'))));

                const firstDelivery = dispatchResult.emailDelivery[0] || {};

                // Update incident notification status (Gmail-exclusive)
                const updatedInc = incidentManager.updateNotificationStatus(incident.id, {
                    email: {
                        status: emailStatus,
                        recipient: firstDelivery.recipient || null,
                        emailAddress: firstDelivery.email || null,
                        messageId: firstDelivery.messageId || null,
                        providerResponse: firstDelivery.providerResponse || null,
                        error: firstDelivery.reason || null,
                        sentCount: dispatchResult.emailDelivery.filter(e => e.status === 'SENT' || e.status === 'MOCKED').length,
                        failedCount: dispatchResult.emailDelivery.filter(e => e.status === 'FAILED').length,
                        configRequired: emailConfigReq,
                        timestamp: dispatchResult.timestamp,
                        deliveries: dispatchResult.emailDelivery
                    }
                });

                if (updatedInc) {
                    io.emit('incident_updated', updatedInc);
                }

                // Emit live dispatch event to UI
                io.emit('emergency_dispatch_event', {
                    incidentId: incident.id,
                    hazard: incident.hazard,
                    source: incident.source,
                    emailDelivery: dispatchResult.emailDelivery,
                    timestamp: dispatchResult.timestamp
                });

                io.emit('agent_log', `📧 [Gmail SMTP Alert] ${incident.id}: ${emailStatus} (${firstDelivery.email || 'No Recipient'})`);
            }).catch((dispatchErr) => {
                console.error('❌ Gmail alert dispatch error:', dispatchErr.message);
                incidentManager.updateNotificationStatus(incident.id, {
                    email: {
                        status: 'FAILED',
                        error: dispatchErr.message
                    }
                });
            });
        }
    }

    return res.status(200).json({ status: 'broadcast_complete' });
});

// ==================== INCIDENT AUDIT & MANAGEMENT API ====================
app.get('/api/incidents', (req, res) => {
    const { level, hazard, status, search, limit, offset } = req.query;
    const result = incidentManager.getIncidents({ level, hazard, status, search, limit, offset });
    return res.status(200).json(result);
});

// Acknowledge incident
app.post('/api/incidents/:id/acknowledge', (req, res) => {
    try {
        const { officerName, notes } = req.body;
        const updated = incidentManager.acknowledgeIncident(req.params.id, {
            officerName: officerName || 'Safety Officer On Duty',
            notes: notes || 'Supervisor verified perimeter and acknowledged SOP.'
        });

        io.emit('incident_updated', updated);
        io.emit('agent_log', `[Supervisor Action] Incident ${updated.id} acknowledged by ${updated.acknowledgedBy}.`);
        return res.status(200).json(updated);
    } catch (err) {
        return res.status(404).json({ error: err.message });
    }
});

// Resolve incident
app.post('/api/incidents/:id/resolve', (req, res) => {
    try {
        const { officerName, notes } = req.body;
        const updated = incidentManager.resolveIncident(req.params.id, {
            officerName: officerName || 'Safety Officer On Duty',
            notes: notes || 'Hazard mitigated and sector cleared for routine operation.'
        });

        io.emit('incident_updated', updated);
        io.emit('agent_log', `[Supervisor Action] Incident ${updated.id} resolved by ${updated.acknowledgedBy}.`);
        return res.status(200).json(updated);
    } catch (err) {
        return res.status(404).json({ error: err.message });
    }
});

// Archive incident
app.delete('/api/incidents/:id', requireAdminKey, (req, res) => {
    const success = incidentManager.deleteIncident(req.params.id);
    if (success) {
        io.emit('agent_log', `[Audit Registry] Incident ${req.params.id} archived.`);
        return res.status(200).json({ status: 'archived', id: req.params.id });
    }
    return res.status(404).json({ error: 'Incident not found' });
});

// ==================== EMERGENCY CONTACTS & CONFIG STATUS API ====================
app.get('/api/alerts/config-status', (req, res) => {
    return res.status(200).json(alertService.getConfigStatus());
});

app.get('/api/alerts/recipients', (req, res) => {
    return res.status(200).json(alertService.getRecipients());
});

app.post('/api/alerts/recipients', requireAdminKey, (req, res) => {
    try {
        const newRec = alertService.addRecipient(req.body);
        io.emit('recipients_update', alertService.getRecipients());
        io.emit('agent_log', `[Personnel Registry] Added safety contact: ${newRec.name} (${newRec.email})`);
        return res.status(201).json(newRec);
    } catch (err) {
        return res.status(400).json({ error: err.message });
    }
});

app.put('/api/alerts/recipients/:id', requireAdminKey, (req, res) => {
    try {
        const updated = alertService.updateRecipient(req.params.id, req.body);
        io.emit('recipients_update', alertService.getRecipients());
        return res.status(200).json(updated);
    } catch (err) {
        return res.status(404).json({ error: err.message });
    }
});

app.delete('/api/alerts/recipients/:id', requireAdminKey, (req, res) => {
    const success = alertService.deleteRecipient(req.params.id);
    if (success) {
        io.emit('recipients_update', alertService.getRecipients());
        io.emit('agent_log', `[Personnel Registry] Removed contact: ${req.params.id}`);
        return res.status(200).json({ status: 'deleted' });
    }
    return res.status(404).json({ error: 'Recipient not found' });
});

// ==================== GMAIL SMTP TEST DRILL API ====================
// Send Test Email endpoint (Requirement 6: Provide a Send Test Email button)
app.post('/api/alerts/test', async (req, res) => {
    try {
        const isMock = Boolean(req.body && req.body.mock);
        // Register this test drill as an incident in incidentManager
        const drillIncident = incidentManager.createIncident({
            source: 'CONTROL ROOM CONSOLE - DRILL',
            hazard: 'EMERGENCY_DRILL_SYSTEM_CHECK',
            level: 'CRITICAL',
            score: 0.99,
            sop_id: 'DRILL-VERIFY-01',
            standard_code: '29 CFR 1910.38',
            hazardDescription: 'System verification drill testing Gmail SMTP emergency email dispatch.'
        });
        io.emit('incident_event', drillIncident);

        const result = await alertService.dispatchAlert({
            incidentId: drillIncident.id,
            source: drillIncident.source,
            score: drillIncident.score,
            level: drillIncident.level,
            hazard: drillIncident.hazard,
            protocol: 'Verification drill testing Gmail SMTP dispatch to safety administrator.',
            sop_details: {
                sop_id: 'DRILL-VERIFY-01',
                standard_code: '29 CFR 1910.38',
                summary: 'Emergency action drill testing Gmail SMTP mail gateway.',
                emergency_action: 'TEST ONLY: Confirm test alert email receipt in safety supervisor inbox.'
            },
            isTest: true,
            isMock: isMock
        });

        // Compute and update delivery status
        const emailSent = result.emailDelivery.some(e => e.status === 'SENT');
        const emailMocked = result.emailDelivery.some(e => e.status === 'MOCKED');
        const emailFailed = result.emailDelivery.some(e => e.status === 'FAILED');
        const emailConfigReq = result.emailDelivery.some(e => e.status === 'CONFIG_REQUIRED');
        const emailStatus = emailSent ? 'SENT' : (emailMocked ? 'MOCKED' : (emailFailed ? 'FAILED' : (emailConfigReq ? 'CONFIG_REQUIRED' : 'OFF')));

        const firstDelivery = result.emailDelivery[0] || {};

        const updatedDrill = incidentManager.updateNotificationStatus(drillIncident.id, {
            email: {
                status: emailStatus,
                recipient: firstDelivery.recipient || null,
                emailAddress: firstDelivery.email || null,
                messageId: firstDelivery.messageId || null,
                providerResponse: firstDelivery.providerResponse || null,
                error: firstDelivery.reason || null,
                sentCount: result.emailDelivery.filter(e => e.status === 'SENT' || e.status === 'MOCKED').length,
                failedCount: result.emailDelivery.filter(e => e.status === 'FAILED').length,
                configRequired: emailConfigReq,
                timestamp: result.timestamp,
                deliveries: result.emailDelivery
            }
        });
        if (updatedDrill) {
            io.emit('incident_updated', updatedDrill);
        }

        io.emit('emergency_dispatch_event', {
            incidentId: drillIncident.id,
            hazard: 'EMERGENCY_DRILL_SYSTEM_CHECK',
            source: 'Control Room Console',
            emailDelivery: result.emailDelivery,
            timestamp: result.timestamp
        });

        const sentCount = result.emailDelivery.filter(e => e.status === 'SENT' || e.status === 'MOCKED').length;
        io.emit('agent_log', `⚡ [Test Email Dispatched] Gmail alert status: ${emailStatus} (${firstDelivery.email || 'No email'})`);
        return res.status(200).json({ status: 'test_email_dispatched', result });
    } catch (err) {
        return res.status(500).json({ error: err.message });
    }
});

// Direct single-target verification test endpoint (Email only)
app.post('/api/alerts/test-direct', async (req, res) => {
    try {
        const { email, name, isMock } = req.body;
        const testIncidentId = `DRILL-DIRECT-${Date.now().toString().slice(-4)}`;
        
        const directRecipient = {
            id: 'rcpt-direct-test',
            name: name || 'Testing Administrator',
            role: 'Lead Safety Officer',
            email: email || '',
            emailEnabled: Boolean(email)
        };

        const originalRecipients = [...alertService.getRecipients()];
        alertService.recipients = [directRecipient];

        const result = await alertService.dispatchAlert({
            incidentId: testIncidentId,
            source: 'EMERGENCY DISPATCH VERIFICATION CONSOLE',
            score: 0.95,
            level: 'CRITICAL',
            hazard: 'EMERGENCY_DRILL_SYSTEM_CHECK',
            protocol: 'Verification drill testing direct safety administrator email dispatch via Gmail SMTP.',
            sop_details: {
                sop_id: 'DRILL-DIRECT-01',
                standard_code: '29 CFR 1910.38',
                summary: 'Direct recipient verification testing email deliverability via Gmail SMTP.',
                emergency_action: 'TEST ONLY: Confirm email delivery in supervisor inbox.'
            },
            isTest: true,
            isMock: Boolean(isMock)
        });

        // Restore original recipients
        alertService.recipients = originalRecipients;

        return res.status(200).json({ status: 'direct_test_completed', result });
    } catch (err) {
        return res.status(500).json({ error: err.message });
    }
});

// Stream reset endpoint
app.post('/api/stream/reset', async (req, res) => {
    try {
        // Reset local cooldowns
        incidentManager.cooldownMap.clear();

        // Forward reset to FastAPI
        await fetch(`${FASTAPI_URL}/api/stream/reset`, { method: 'POST' }).catch(() => {});

        io.emit('stream_reset', { timestamp: new Date().toLocaleTimeString() });
        io.emit('agent_log', `[System] Surveillance viewport and tracking cache reset.`);
        return res.status(200).json({ status: 'reset_complete' });
    } catch (err) {
        return res.status(500).json({ error: err.message });
    }
});

// Forwarding camera stream ingestion to Python FastAPI microservice
app.post('/api/source', async (req, res) => {
    const { type, source } = req.body;
    const sourceLabel = source || 'Uploaded Video Audit File';
    console.log(`📡 Ingestion request received: [${type}] ${sourceLabel}`);

    io.emit('agent_log', `[System] Ingestion initiated: [${type.toUpperCase()}] ${sourceLabel}`);

    try {
        const response = await fetch(`${FASTAPI_URL}/api/start-stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ source: sourceLabel, mode: type })
        });
        const data = await response.json();
        return res.status(200).json({ status: 'forwarded_to_fastapi', data });
    } catch (err) {
        console.warn('ℹ️ Python FastAPI microservice offline or unreachable.');
        return res.status(200).json({ status: 'simulation_fallback', message: 'FastAPI service offline' });
    }
});

// Forward audit progress from Python video_auditor to React Dashboard via Socket.IO
app.post('/api/audit/progress', (req, res) => {
    io.emit('audit_progress', req.body);
    return res.status(200).json({ status: 'broadcasted' });
});

// Trigger high-performance video audit on uploaded or server video
app.post('/api/audit/start', async (req, res) => {
    const { videoPath, filename, inference_target } = req.body;
    const targetLabel = inference_target === 'cloud_gpu' ? 'Cloud GPU (Remote)' : 'Apple Silicon (MPS)';
    io.emit('agent_log', `[Video Audit] Initializing ${targetLabel} parallel audit for: ${filename || 'Selected Video'}`);

    try {
        const response = await fetch(`${FASTAPI_URL}/api/audit-video`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ video_path: videoPath, filename, inference_target: inference_target || 'local_m2' })
        });
        const data = await response.json();
        io.emit('agent_log', `[Video Audit] Audit complete: ${data.processed_frames} frames in ${data.elapsed_seconds}s (${data.processing_speed_fps} FPS, ${data.incidents?.length || 0} incidents found)`);
        io.emit('audit_complete', data);
        return res.status(200).json(data);
    } catch (err) {
        return res.status(500).json({ error: err.message });
    }
});

// Acknowledge SOP Alert shortcut
app.post('/api/acknowledge', (req, res) => {
    const { sop_id } = req.body;
    io.emit('agent_log', `[Supervisor Action] Protocol ${sop_id || 'SOP-EHS-042'} acknowledged by Safety Officer on duty.`);
    io.emit('protocol_acknowledged', { sop_id });
    return res.status(200).json({ status: 'acknowledged' });
});

// Health check endpoint
app.get('/health', (req, res) => {
    return res.status(200).json({
        status: 'online',
        service: 'VigilOps Matrix Telemetry Gateway',
        recipients_count: alertService.getRecipients().length,
        incidents_count: incidentManager.getIncidents({ limit: 1 }).total
    });
});

const PORT = process.env.PORT || 5000;
server.listen(PORT, () => {
    console.log(`🚀 VigilOps AI Matrix Telemetry Gateway active on http://localhost:${PORT}`);
});