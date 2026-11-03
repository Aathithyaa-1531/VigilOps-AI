/**
 * VigilOps AI - Persistent Incident Management & Audit Engine
 * Manages incident records, deduplication, lifecycle states, and notification tracking.
 */

const fs = require('fs');
const path = require('path');

const DATA_DIR = path.join(__dirname, 'data');
const INCIDENTS_FILE = path.join(DATA_DIR, 'incidents.json');

if (!fs.existsSync(DATA_DIR)) {
    fs.mkdirSync(DATA_DIR, { recursive: true });
}

// Default initial safety records if storage file is completely empty
const SEED_INCIDENTS = [
    {
        id: 'INC-2026-001',
        timestamp: new Date(Date.now() - 3600000 * 2).toISOString(),
        formattedTime: '08:45:10',
        source: 'CAM_01 - Crane Hoist Sector A',
        hazard: 'NO_HELMET',
        hazardDescription: 'Unhelmeted Worker in Overhead Hoist Perimeter',
        level: 'CRITICAL',
        score: 0.88,
        sop_id: 'SOP-EHS-042',
        standard_code: '29 CFR 1910.135',
        status: 'ACKNOWLEDGED',
        acknowledgedBy: 'Alex Vance (Lead EHS Director)',
        acknowledgedAt: new Date(Date.now() - 3600000 * 1.8).toISOString(),
        notes: 'Worker briefed and issued Type II Class E hard hat before re-entry.',
        notification: {
            email: { status: 'SENT', recipients: ['alex.vance@industrial-safety.com'], timestamp: '08:45:12' }
        }
    },
    {
        id: 'INC-2026-002',
        timestamp: new Date(Date.now() - 3600000 * 1).toISOString(),
        formattedTime: '09:30:25',
        source: 'CAM_04 - Forklift Loading Bay 4',
        hazard: 'FORKLIFT_HAZARD',
        hazardDescription: 'Pedestrian Crossing within 2.5m Turning Radius',
        level: 'MODERATE',
        score: 0.64,
        sop_id: 'SOP-EHS-077',
        standard_code: '29 CFR 1910.178(m)',
        status: 'RESOLVED',
        acknowledgedBy: 'Marcus Brody (Floor Supervisor)',
        acknowledgedAt: new Date(Date.now() - 3600000 * 0.9).toISOString(),
        resolvedAt: new Date(Date.now() - 3600000 * 0.7).toISOString(),
        notes: 'Pedestrian redirected to designated green safety walkway.',
        notification: {
            email: { status: 'SKIPPED', reason: 'Severity below critical threshold' }
        }
    }
];

class IncidentManager {
    constructor() {
        this.incidents = this.loadIncidents();
        this.cooldownMap = new Map(); // tracks last detection per `source:hazard`
    }

    loadIncidents() {
        try {
            if (fs.existsSync(INCIDENTS_FILE)) {
                const data = fs.readFileSync(INCIDENTS_FILE, 'utf8');
                const parsed = JSON.parse(data);
                if (Array.isArray(parsed) && parsed.length > 0) {
                    return parsed;
                }
            }
        } catch (err) {
            console.error('⚠️ Error reading incidents file:', err.message);
        }
        this.saveIncidents(SEED_INCIDENTS);
        return SEED_INCIDENTS;
    }

    saveIncidents(incidents) {
        try {
            fs.writeFileSync(INCIDENTS_FILE, JSON.stringify(incidents, null, 2), 'utf8');
            this.incidents = incidents;
        } catch (err) {
            console.error('⚠️ Error writing incidents file:', err.message);
        }
    }

    /**
     * Checks if a detection is a duplicate within the active cooldown period
     */
    isDuplicate(source, hazard, level, cooldownSeconds = 60) {
        const key = `${source}:${hazard}`;
        const lastEntry = this.cooldownMap.get(key);
        if (!lastEntry) return false;

        const timeDiffSec = (Date.now() - lastEntry.time) / 1000;
        
        // If within cooldown period and level hasn't escalated to HIGH or CRITICAL
        if (timeDiffSec < cooldownSeconds) {
            if (lastEntry.level !== 'CRITICAL' && level === 'CRITICAL') {
                return false; // Escalation to CRITICAL triggers new alert
            }
            if (lastEntry.level !== 'HIGH' && lastEntry.level !== 'CRITICAL' && level === 'HIGH') {
                return false; // Escalation to HIGH triggers new alert
            }
            return true;
        }
        return false;
    }

    registerDetection(source, hazard, level) {
        const key = `${source}:${hazard}`;
        this.cooldownMap.set(key, { time: Date.now(), level });
    }

    /**
     * Creates and records a validated incident
     */
    createIncident({
        source,
        hazard,
        level = 'CRITICAL',
        score = 0.85,
        sop_id = 'SOP-EHS-042',
        standard_code = '29 CFR 1910.132',
        hazardDescription = '',
        breakdown = {}
    }) {
        const timestamp = new Date();
        const nextNum = this.incidents.length + 1;
        const id = `INC-${timestamp.getFullYear()}-${String(nextNum).padStart(3, '0')}`;

        const newIncident = {
            id,
            timestamp: timestamp.toISOString(),
            formattedTime: timestamp.toLocaleTimeString(),
            source: source || 'SURV_CAM_01',
            hazard: hazard || 'UNSAFE_CONDITION',
            hazardDescription: hazardDescription || `${hazard} Violation Detected`,
            level,
            score: Math.round(Number(score) * 100) / 100,
            sop_id,
            standard_code,
            status: 'ACTIVE',
            acknowledgedBy: null,
            acknowledgedAt: null,
            resolvedAt: null,
            notes: '',
            breakdown,
            notification: {
                email: { status: 'QUEUED' }
            }
        };

        this.incidents.unshift(newIncident);
        if (this.incidents.length > 200) {
            this.incidents.pop(); // Keep rolling window of 200 incidents
        }
        this.saveIncidents(this.incidents);
        this.registerDetection(source, hazard, level);

        return newIncident;
    }

    updateNotificationStatus(incidentId, notificationResults) {
        const inc = this.incidents.find(i => i.id === incidentId);
        if (!inc) return null;

        inc.notification = {
            ...inc.notification,
            ...notificationResults,
            updatedAt: new Date().toISOString()
        };
        this.saveIncidents(this.incidents);
        return inc;
    }

    acknowledgeIncident(incidentId, { officerName = 'Safety Officer', notes = '' }) {
        const inc = this.incidents.find(i => i.id === incidentId);
        if (!inc) throw new Error(`Incident ${incidentId} not found`);

        inc.status = 'ACKNOWLEDGED';
        inc.acknowledgedBy = officerName;
        inc.acknowledgedAt = new Date().toISOString();
        if (notes) inc.notes = notes;

        this.saveIncidents(this.incidents);
        return inc;
    }

    resolveIncident(incidentId, { officerName = 'Safety Officer', notes = '' }) {
        const inc = this.incidents.find(i => i.id === incidentId);
        if (!inc) throw new Error(`Incident ${incidentId} not found`);

        inc.status = 'RESOLVED';
        inc.resolvedAt = new Date().toISOString();
        if (!inc.acknowledgedBy) inc.acknowledgedBy = officerName;
        if (notes) inc.notes = notes;

        this.saveIncidents(this.incidents);
        return inc;
    }

    deleteIncident(incidentId) {
        const initialLen = this.incidents.length;
        this.incidents = this.incidents.filter(i => i.id !== incidentId);
        if (this.incidents.length !== initialLen) {
            this.saveIncidents(this.incidents);
            return true;
        }
        return false;
    }

    getIncidents({ level, hazard, status, search, limit = 50, offset = 0 } = {}) {
        let filtered = [...this.incidents];

        if (level && level !== 'ALL') {
            filtered = filtered.filter(i => i.level === level.toUpperCase());
        }

        if (hazard && hazard !== 'ALL') {
            filtered = filtered.filter(i => i.hazard.toUpperCase().includes(hazard.toUpperCase()));
        }

        if (status && status !== 'ALL') {
            filtered = filtered.filter(i => i.status === status.toUpperCase());
        }

        if (search && search.trim()) {
            const query = search.trim().toLowerCase();
            filtered = filtered.filter(i =>
                i.id.toLowerCase().includes(query) ||
                i.source.toLowerCase().includes(query) ||
                i.hazard.toLowerCase().includes(query) ||
                (i.sop_id && i.sop_id.toLowerCase().includes(query)) ||
                (i.notes && i.notes.toLowerCase().includes(query))
            );
        }

        const total = filtered.length;
        const paginated = filtered.slice(Number(offset), Number(offset) + Number(limit));

        return {
            total,
            incidents: paginated
        };
    }
}

const incidentManager = new IncidentManager();
module.exports = incidentManager;
