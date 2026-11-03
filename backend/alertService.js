/**
 * VigilOps AI – Gmail SMTP Exclusive Notification Service
 * Dispatches verified emergency notifications exclusively via Gmail SMTP (Nodemailer / STARTTLS Port 587).
 *
 * Strict Delivery States (Gmail SMTP):
 * - CONFIG_REQUIRED : Gmail credentials (SMTP_USER, SMTP_PASS) missing in backend/.env.
 * - SENT            : Gmail SMTP server accepted the message for delivery (250 OK).
 * - FAILED          : SMTP rejected the message, auth failed, or connection failed.
 * - MOCKED          : Test notification only; no real email was sent (development / automated tests).
 * - SKIPPED         : Duplicate alert prevented for the same incident record.
 *
 * NOTE: Gmail SMTP acceptance is strictly recorded as SENT, NEVER as DELIVERED.
 * SMS functionality is completely removed and disabled under all circumstances.
 */

const fs = require('fs');
const path = require('path');
const nodemailer = require('nodemailer');

const DATA_DIR = path.join(__dirname, 'data');
const RECIPIENTS_FILE = path.join(DATA_DIR, 'recipients.json');
const DISPATCH_LOG_FILE = path.join(DATA_DIR, 'dispatch_history.json');

if (!fs.existsSync(DATA_DIR)) {
    fs.mkdirSync(DATA_DIR, { recursive: true });
}

function isValidEmail(email) {
    if (!email) return false;
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(String(email).trim());
}

const DEFAULT_RECIPIENTS = [
    {
        id: 'rcpt-admin-001',
        name: 'Lead Safety Officer',
        role: 'Site EHS Director',
        email: 'safety.director@industrial-facility.com',
        emailEnabled: true,
        createdAt: new Date().toISOString()
    }
];

class AlertService {
    constructor() {
        this.recipients = this.loadRecipients();
        this.dispatchHistory = this.loadDispatchHistory();
        this.mailTransporter = null;
        this.mockTransporter = null;
        this.initProviders();
    }

    loadRecipients() {
        try {
            if (fs.existsSync(RECIPIENTS_FILE)) {
                const data = fs.readFileSync(RECIPIENTS_FILE, 'utf8');
                const parsed = JSON.parse(data);
                if (Array.isArray(parsed) && parsed.length > 0) {
                    return parsed.map(r => ({
                        id: r.id,
                        name: r.name,
                        role: r.role || 'Safety Officer',
                        email: r.email,
                        emailEnabled: r.emailEnabled !== false,
                        createdAt: r.createdAt || new Date().toISOString()
                    }));
                }
            }
        } catch (e) {
            console.error('⚠️ Error reading recipients file:', e.message);
        }
        this.saveRecipients(DEFAULT_RECIPIENTS);
        return DEFAULT_RECIPIENTS;
    }

    saveRecipients(recipients) {
        try {
            fs.writeFileSync(RECIPIENTS_FILE, JSON.stringify(recipients, null, 2), 'utf8');
            this.recipients = recipients;
        } catch (e) {
            console.error('⚠️ Error saving recipients file:', e.message);
        }
    }

    loadDispatchHistory() {
        try {
            if (fs.existsSync(DISPATCH_LOG_FILE)) {
                const data = fs.readFileSync(DISPATCH_LOG_FILE, 'utf8');
                return JSON.parse(data);
            }
        } catch (e) {
            console.error('⚠️ Error loading dispatch history:', e.message);
        }
        return [];
    }

    logDispatch(record) {
        this.dispatchHistory.unshift(record);
        if (this.dispatchHistory.length > 100) this.dispatchHistory.pop();
        try {
            fs.writeFileSync(DISPATCH_LOG_FILE, JSON.stringify(this.dispatchHistory, null, 2), 'utf8');
        } catch (e) {
            console.error('⚠️ Error saving dispatch log:', e.message);
        }
    }

    /**
     * Initializes Gmail SMTP provider dynamically from backend/.env configuration.
     */
    async initProviders(force = false) {
        if (process.env.NODE_ENV === 'test' && !force) {
            return;
        }

        require('dotenv').config({ override: true });

        const emailEnabled = process.env.EMAIL_ENABLED !== 'false';
        if (!emailEnabled) {
            console.log('ℹ️ [AlertService] EMAIL_ENABLED is false. Email dispatch disabled.');
            this.mailTransporter = null;
            return;
        }

        const host = process.env.SMTP_HOST || 'smtp.gmail.com';
        const user = process.env.SMTP_USER;
        const pass = process.env.SMTP_PASSWORD || process.env.SMTP_PASS;
        const port = Number(process.env.SMTP_PORT) || 587;
        const secure = process.env.SMTP_SECURE === 'true';

        if (user && pass) {
            try {
                const cleanPass = pass.trim().replace(/\s+/g, ''); // strip spaces in Gmail app password
                const transportConfig = {
                    host: host.trim(),
                    port: port,
                    secure: secure,
                    auth: {
                        user: user.trim(),
                        pass: cleanPass
                    },
                    connectionTimeout: 10000,
                    greetingTimeout: 10000,
                    socketTimeout: 15000
                };

                this.mailTransporter = nodemailer.createTransport(transportConfig);
                console.log(`📧 [AlertService] Gmail SMTP Transporter initialized (${user.trim()} via ${host}:${port}, Secure: ${secure}).`);
            } catch (err) {
                console.error('❌ [AlertService] Failed to initialize Gmail SMTP transporter:', err.message);
                this.mailTransporter = null;
            }
        } else {
            this.mailTransporter = null;
        }
    }

    /**
     * Allows setting a mocked transporter for development or unit testing without live network.
     */
    setMockTransporter(transporter) {
        this.mockTransporter = transporter;
    }

    getRecipients() {
        return this.recipients;
    }

    /**
     * Validates provider configuration health without ever exposing credentials.
     */
    getConfigStatus() {
        if (process.env.NODE_ENV !== 'test') {
            require('dotenv').config({ override: true });
        }
        const hasSmtpPass = Boolean(process.env.SMTP_PASSWORD || process.env.SMTP_PASS);
        const hasSmtpUser = Boolean(process.env.SMTP_USER);
        const smtpHost = process.env.SMTP_HOST || 'smtp.gmail.com';
        const smtpPort = Number(process.env.SMTP_PORT) || 587;
        const smtpSecure = process.env.SMTP_SECURE === 'true';
        const emailEnabled = process.env.EMAIL_ENABLED !== 'false';

        const isConfigured = hasSmtpPass && hasSmtpUser && emailEnabled;

        return {
            email: {
                provider: 'gmail',
                host: smtpHost,
                port: smtpPort,
                secure: smtpSecure,
                configured: isConfigured,
                user: process.env.SMTP_USER || null,
                status: isConfigured ? 'CONFIGURED' : 'CONFIG_REQUIRED',
                emailEnabled: emailEnabled
            },
            recipientsCount: this.recipients.filter(r => r.email && r.emailEnabled).length,
            smsEnabled: false,
            readyForLiveDrill: isConfigured
        };
    }

    addRecipient({ name, role, email, emailEnabled = true }) {
        if (!email) {
            throw new Error('A valid safety administrator email address is required.');
        }

        if (!isValidEmail(email)) {
            throw new Error(`Invalid email address format: '${email}'`);
        }

        const newRecipient = {
            id: `rcpt-${Date.now()}-${Math.floor(Math.random() * 1000)}`,
            name: name ? String(name).trim() : 'Safety Administrator',
            role: role ? String(role).trim() : 'EHS Specialist',
            email: String(email).trim().toLowerCase(),
            emailEnabled: Boolean(emailEnabled),
            createdAt: new Date().toISOString()
        };

        const existing = this.recipients.find(r => r.email.toLowerCase() === newRecipient.email);
        if (existing) {
            throw new Error(`Recipient with email '${newRecipient.email}' already exists in registry.`);
        }

        this.recipients.push(newRecipient);
        this.saveRecipients(this.recipients);
        return newRecipient;
    }

    updateRecipient(id, updates = {}) {
        const rcpt = this.recipients.find(r => r.id === id);
        if (!rcpt) throw new Error(`Recipient with ID '${id}' not found.`);

        if (updates.email && !isValidEmail(updates.email)) {
            throw new Error(`Invalid email address format: '${updates.email}'`);
        }

        if (updates.name !== undefined) rcpt.name = updates.name.trim();
        if (updates.role !== undefined) rcpt.role = updates.role.trim();
        if (updates.email !== undefined) rcpt.email = updates.email.trim().toLowerCase();
        if (updates.emailEnabled !== undefined) rcpt.emailEnabled = Boolean(updates.emailEnabled);

        this.saveRecipients(this.recipients);
        return rcpt;
    }

    deleteRecipient(id) {
        const initialLen = this.recipients.length;
        this.recipients = this.recipients.filter(r => r.id !== id);
        if (this.recipients.length < initialLen) {
            this.saveRecipients(this.recipients);
            return true;
        }
        return false;
    }

    /**
     * Primary Email Dispatch Pipeline (Exclusive Gmail SMTP)
     */
    async dispatchAlert({
        incidentId,
        source = 'SURV_CAM_01',
        score = 0.85,
        level = 'CRITICAL',
        hazard = 'SAFETY_HAZARD',
        protocol = '',
        sop_details = {},
        location = '',
        isTest = false,
        isMock = false
    }) {
        const timestamp = new Date();
        const formattedTime = timestamp.toLocaleTimeString();
        const isoTimestamp = timestamp.toISOString();
        const dashboardUrl = process.env.DASHBOARD_URL || 'http://localhost:5173';
        const facilityLocation = location || source || 'Industrial Production Sector';

        const emailTargets = this.recipients.filter(r => r.email && r.emailEnabled);

        const dispatchRecord = {
            id: `DISPATCH-${Date.now()}`,
            incidentId: incidentId || `INC-AUTO-${Date.now()}`,
            timestamp: formattedTime,
            isoTimestamp: isoTimestamp,
            level,
            hazard,
            source,
            location: facilityLocation,
            score: Math.round(Number(score) * 100),
            isTest,
            emailDelivery: [],
            smsDelivery: [], // Always empty; SMS is disabled
            overallStatus: 'COMPLETED'
        };

        // Re-read .env to catch dynamic user key updates unless in test mode
        if (process.env.NODE_ENV !== 'test') {
            await this.initProviders();
        }

        console.log(`🚨 [AlertService] Dispatching Gmail Alert for ${hazard} at ${source} (Incident: ${dispatchRecord.incidentId})...`);

        // Check if email dispatch is globally enabled
        const emailEnabled = process.env.EMAIL_ENABLED !== 'false';
        if (!emailEnabled) {
            console.log('ℹ️ [AlertService] EMAIL_ENABLED is set to false in .env. Skipping dispatch.');
            for (const target of emailTargets) {
                dispatchRecord.emailDelivery.push({
                    recipient: target.name,
                    email: target.email,
                    status: 'SKIPPED',
                    reason: 'EMAIL_ENABLED is set to false in backend/.env',
                    timestamp: formattedTime
                });
            }
            this.logDispatch(dispatchRecord);
            return dispatchRecord;
        }

        // ==================== EMAIL CONTENT FORMATTING ====================
        // Requirement B.3: Distinct Subject Lines for Genuine vs Test Alerts
        // For genuine critical incidents: [VigilOps AI] CRITICAL SAFETY ALERT – Incident ID: {incident_id}
        // For genuine high incidents:     [VigilOps AI] HIGH PRIORITY SAFETY ALERT – Incident ID: {incident_id}
        // For manual test drill:          [TEST EMAIL] [VigilOps AI] SAFETY DRILL – Incident ID: {incident_id}
        const emailSubject = isTest
            ? `[TEST EMAIL] [VigilOps AI] SAFETY DRILL – Incident ID: ${dispatchRecord.incidentId}`
            : (level === 'HIGH'
                ? `[VigilOps AI] HIGH PRIORITY SAFETY ALERT – Incident ID: ${dispatchRecord.incidentId}`
                : `[VigilOps AI] CRITICAL SAFETY ALERT – Incident ID: ${dispatchRecord.incidentId}`);

        const ppeViolationDesc = hazard.includes('HELMET')
            ? 'Safety Helmet / Hard Hat Non-Compliance (OSHA 1910.135)'
            : hazard.includes('VEST')
            ? 'High-Visibility Retroreflective Vest Non-Compliance (ANSI/ISEA 107)'
            : `${hazard} Safety Protocol Non-Compliance`;

        const recommendedAction = sop_details?.emergency_action || 'Evacuate danger perimeter and contact area safety supervisor immediately.';
        const standardCitation = `${sop_details?.standard_code || '29 CFR 1910.135'} (${sop_details?.sop_id || 'SOP-EHS-042'})`;
        const sopProtocol = protocol || sop_details?.summary || 'Halt sector activity immediately and equip mandatory protective safety equipment.';

        // Readable plain-text fallback (Requirement B.3 - No TEST wording in real alerts)
        const plainTextBody = `
===================================================================
${emailSubject}
${isTest ? '*** SYSTEM VERIFICATION DRILL – TEST EMAIL ONLY ***' : (level === 'HIGH' ? '*** HIGH PRIORITY EHS SAFETY VIOLATION DETECTED ***' : '*** CRITICAL EHS SAFETY VIOLATION DETECTED ***')}
===================================================================

Incident ID:            ${dispatchRecord.incidentId}
Detected Hazard:        ${hazard}
Detected PPE Violation: ${ppeViolationDesc}
Risk Level:             ${level}
Threat Score:           ${Math.round(Number(score) * 100)}%
Detection Time:         ${formattedTime} (${isoTimestamp})
Camera / Source:        ${source}
Facility Location:      ${facilityLocation}
OSHA Standard:          ${standardCitation}

Standard Operating Protocol (SOP):
${sopProtocol}

RECOMMENDED SAFETY ACTION:
${recommendedAction}

Incident Command Reference:
${dashboardUrl}

-------------------------------------------------------------------
Dispatched autonomously by VigilOps AI Matrix Telemetry Gateway via Gmail SMTP.
===================================================================
`.trim();

        // Professional HTML email template (Requirement B.3 - No TEST wording in real alerts)
        const headerBg = isTest ? '#0369a1' : (level === 'HIGH' ? '#92400e' : '#991b1b');
        const headerBorder = isTest ? '#38bdf8' : (level === 'HIGH' ? '#f59e0b' : '#ef4444');
        const badgeLabel = isTest ? 'VERIFICATION DRILL TEST EMAIL' : (level === 'HIGH' ? 'AUTONOMOUS EHS HIGH PRIORITY ALERT' : 'AUTONOMOUS EHS CRITICAL ALERT');
        const headerTitle = isTest ? 'VigilOps AI — Test Emergency Drill' : (level === 'HIGH' ? 'VigilOps AI — High Priority Safety Alert' : 'VigilOps AI — Critical Safety Violation');

        const emailHtml = `
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>${emailSubject}</title>
            </head>
            <body style="margin: 0; padding: 20px; background-color: #020617; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; color: #f1f5f9;">
                <div style="max-width: 650px; margin: auto; background-color: #0b0f19; border: 1px solid #334155; border-radius: 12px; overflow: hidden; box-shadow: 0 10px 25px rgba(0,0,0,0.5);">
                    <!-- Header -->
                    <div style="background-color: ${headerBg}; padding: 18px 24px; border-bottom: 2px solid ${headerBorder};">
                        <span style="display: inline-block; background-color: rgba(0,0,0,0.3); color: #ffffff; padding: 3px 8px; border-radius: 4px; font-size: 10px; font-weight: 800; letter-spacing: 0.5px; text-transform: uppercase;">
                            ${badgeLabel}
                        </span>
                        <h1 style="color: #ffffff; margin: 8px 0 2px 0; font-size: 20px; font-weight: 700; letter-spacing: -0.5px;">
                            ${headerTitle}
                        </h1>
                        <p style="color: rgba(255,255,255,0.85); font-size: 12px; margin: 0; font-family: monospace;">
                            Incident ID: <strong>${dispatchRecord.incidentId}</strong> | Detected: ${formattedTime}
                        </p>
                    </div>

                    <!-- Body Content -->
                    <div style="padding: 24px;">
                        <!-- Primary Metrics Table -->
                        <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px; font-size: 13px;">
                            <tr style="border-bottom: 1px solid #1e293b;">
                                <td style="padding: 8px 0; color: #94a3b8; width: 40%;">Incident Identifier:</td>
                                <td style="padding: 8px 0; color: #38bdf8; font-weight: bold; font-family: monospace;">${dispatchRecord.incidentId}</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #1e293b;">
                                <td style="padding: 8px 0; color: #94a3b8;">Detected Hazard:</td>
                                <td style="padding: 8px 0; color: #f87171; font-weight: bold;">${hazard}</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #1e293b;">
                                <td style="padding: 8px 0; color: #94a3b8;">Detected PPE Violation:</td>
                                <td style="padding: 8px 0; color: #fbbf24; font-weight: bold;">${ppeViolationDesc}</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #1e293b;">
                                <td style="padding: 8px 0; color: #94a3b8;">Risk Level:</td>
                                <td style="padding: 8px 0;">
                                    <span style="background-color: ${level === 'CRITICAL' ? '#450a0a' : '#451a03'}; color: ${level === 'CRITICAL' ? '#fca5a5' : '#fcd34d'}; padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 11px; border: 1px solid ${level === 'CRITICAL' ? '#991b1b' : '#b45309'};">
                                        ${level}
                                    </span>
                                </td>
                            </tr>
                            <tr style="border-bottom: 1px solid #1e293b;">
                                <td style="padding: 8px 0; color: #94a3b8;">Threat Score:</td>
                                <td style="padding: 8px 0; color: #f8fafc; font-weight: bold; font-family: monospace;">${Math.round(Number(score) * 100)}%</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #1e293b;">
                                <td style="padding: 8px 0; color: #94a3b8;">Detection Timestamp:</td>
                                <td style="padding: 8px 0; color: #cbd5e1; font-family: monospace;">${formattedTime} (${isoTimestamp})</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #1e293b;">
                                <td style="padding: 8px 0; color: #94a3b8;">Camera / Video Source:</td>
                                <td style="padding: 8px 0; color: #f8fafc; font-weight: bold;">${source}</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #1e293b;">
                                <td style="padding: 8px 0; color: #94a3b8;">Facility Location:</td>
                                <td style="padding: 8px 0; color: #e2e8f0;">${facilityLocation}</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #1e293b;">
                                <td style="padding: 8px 0; color: #94a3b8;">OSHA Standard Citation:</td>
                                <td style="padding: 8px 0; color: #fbbf24; font-weight: bold;">${standardCitation}</td>
                            </tr>
                        </table>

                        <!-- Standard Operating Protocol (SOP) -->
                        <div style="background-color: #1e1b4b; border-left: 4px solid #818cf8; padding: 12px 16px; border-radius: 6px; margin-bottom: 16px;">
                            <strong style="color: #c7d2fe; display: block; margin-bottom: 4px; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px;">Standard Operating Protocol:</strong>
                            <p style="color: #e0e7ff; margin: 0; font-size: 12px; line-height: 1.5;">${sopProtocol}</p>
                        </div>

                        <!-- Mandatory Recommended Action -->
                        <div style="background-color: #450a0a; border: 1px solid #991b1b; padding: 14px 16px; border-radius: 6px; margin-bottom: 22px;">
                            <strong style="color: #fca5a5; font-size: 13px; display: flex; align-items: center; gap: 6px;">
                                🚨 Recommended Safety Action:
                            </strong>
                            <p style="color: #fecaca; margin: 6px 0 0 0; font-size: 13px; line-height: 1.4; font-weight: 500;">
                                ${recommendedAction}
                            </p>
                        </div>

                        <!-- Incident Dashboard Reference Button -->
                        <div style="text-align: center; margin: 24px 0 16px 0;">
                            <a href="${dashboardUrl}" style="background-color: #0284c7; color: #ffffff; padding: 10px 24px; border-radius: 6px; text-decoration: none; font-weight: bold; font-size: 13px; display: inline-block; letter-spacing: 0.3px;">
                                Open Incident Command Console &rarr;
                            </a>
                        </div>
                    </div>

                    <!-- Footer -->
                    <div style="background-color: #0f172a; border-top: 1px solid #1e293b; padding: 14px 24px; font-size: 11px; color: #64748b; text-align: center;">
                        VigilOps AI Matrix Telemetry Gateway — Powered by Gmail SMTP (STARTTLS Port 587).<br>
                        This alert was generated automatically by computer vision & temporal risk models.
                    </div>
                </div>
            </body>
            </html>
        `;

        // ==================== DISPATCH LOOP OVER RECIPIENTS ====================
        for (const target of emailTargets) {
            // Requirement 7: Prevent duplicate incident notification
            if (!isTest && incidentId) {
                const alreadyDispatched = this.dispatchHistory.some(d =>
                    d.incidentId === incidentId &&
                    d.emailDelivery &&
                    d.emailDelivery.some(e => (e.status === 'SENT' || e.status === 'MOCKED') && e.email.toLowerCase() === target.email.toLowerCase())
                );
                if (alreadyDispatched) {
                    console.log(`ℹ️ [AlertService] Duplicate email alert prevented: Email already sent to ${target.email} for incident ${incidentId}.`);
                    dispatchRecord.emailDelivery.push({
                        recipient: target.name,
                        email: target.email,
                        status: 'SKIPPED',
                        reason: `Duplicate alert prevented: Email already dispatched to ${target.email} for incident ${incidentId}`,
                        timestamp: formattedTime
                    });
                    continue;
                }
            }

            // Check if mocked mode is requested (for testing without live network)
            if (isMock || this.mockTransporter) {
                const activeMock = this.mockTransporter || {
                    sendMail: async () => ({
                        messageId: `mocked-${Date.now()}@vigilops.ai`,
                        response: '250 OK Mocked SMTP Acceptance'
                    })
                };

                try {
                    const mockInfo = await activeMock.sendMail({
                        from: process.env.EMAIL_FROM || process.env.SMTP_USER || 'alerts@vigilops.ai',
                        to: target.email,
                        subject: emailSubject,
                        text: plainTextBody,
                        html: emailHtml
                    });

                    dispatchRecord.emailDelivery.push({
                        recipient: target.name,
                        email: target.email,
                        status: 'MOCKED',
                        messageId: mockInfo.messageId,
                        providerResponse: mockInfo.response || '250 OK Mocked SMTP Acceptance',
                        timestamp: formattedTime,
                        isoTimestamp: isoTimestamp,
                        isMock: true
                    });
                    console.log(`🧪 [Email: MOCKED] ${target.email} (MessageId: ${mockInfo.messageId}) - Mocked test notification.`);
                } catch (mockErr) {
                    dispatchRecord.emailDelivery.push({
                        recipient: target.name,
                        email: target.email,
                        status: 'FAILED',
                        code: mockErr.code || 'MOCK_ERROR',
                        reason: mockErr.message,
                        timestamp: formattedTime
                    });
                }
                continue;
            }

            // Real Gmail SMTP Dispatch
            if (!this.mailTransporter) {
                dispatchRecord.emailDelivery.push({
                    recipient: target.name,
                    email: target.email,
                    status: 'CONFIG_REQUIRED',
                    reason: 'Gmail SMTP credentials (SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS) not configured in backend/.env',
                    timestamp: formattedTime
                });
                console.warn(`⚠️ [Email: CONFIG_REQUIRED] ${target.email} - Gmail SMTP credentials missing in backend/.env.`);
            } else {
                try {
                    const sender = (process.env.EMAIL_FROM || process.env.SMTP_USER || 'alerts@vigilops.ai').trim();
                    const info = await this.mailTransporter.sendMail({
                        from: sender,
                        to: target.email,
                        subject: emailSubject,
                        text: plainTextBody,
                        html: emailHtml
                    });

                    // Requirement 5: Status must be SENT (never DELIVERED). Gmail acceptance only confirms queueing.
                    dispatchRecord.emailDelivery.push({
                        recipient: target.name,
                        email: target.email,
                        status: 'SENT',
                        messageId: info.messageId,
                        accepted: info.accepted,
                        providerResponse: info.response || '250 OK Message accepted by Gmail SMTP server',
                        timestamp: formattedTime,
                        isoTimestamp: isoTimestamp
                    });
                    console.log(`✅ [Email: SENT] MessageId ${info.messageId} -> ${target.email} | Provider: ${info.response || '250 OK'}`);
                } catch (err) {
                    dispatchRecord.emailDelivery.push({
                        recipient: target.name,
                        email: target.email,
                        status: 'FAILED',
                        code: err.code || 'SMTP_ERROR',
                        reason: err.message,
                        timestamp: formattedTime,
                        isoTimestamp: isoTimestamp
                    });
                    console.error(`❌ [Email: FAILED] ${target.email}:`, err.message);
                }
            }
        }

        this.logDispatch(dispatchRecord);
        return dispatchRecord;
    }
}

const alertService = new AlertService();
module.exports = alertService;
