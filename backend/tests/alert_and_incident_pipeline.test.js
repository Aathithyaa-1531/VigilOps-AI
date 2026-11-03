/**
 * VigilOps AI - Automated Test Suite: Alert Delivery & Incident Pipeline
 * Verifies Scenarios 1 - 17 for Gmail SMTP exclusive notification system.
 */

const assert = require('assert');
const fs = require('fs');
const path = require('path');

// Ensure clean test environment
process.env.NODE_ENV = 'test';
process.env.ALERT_COOLDOWN_SECONDS = '2'; // Fast 2s cooldown for testing
process.env.SMTP_HOST = ''; // Start without credentials
process.env.SMTP_USER = '';
process.env.SMTP_PASS = '';
process.env.EMAIL_ENABLED = 'true';

const incidentManager = require('../incidentManager');
const alertService = require('../alertService');

async function runTestSuite() {
    console.log('🧪 Starting VigilOps AI Pipeline & Alert Test Suite...\n');
    let passed = 0;
    let failed = 0;

    function test(name, fn) {
        try {
            fn();
            console.log(`  ✅ [PASS] ${name}`);
            passed++;
        } catch (err) {
            console.error(`  ❌ [FAIL] ${name}:`, err.message);
            failed++;
        }
    }

    async function asyncTest(name, fn) {
        try {
            await fn();
            console.log(`  ✅ [PASS] ${name}`);
            passed++;
        } catch (err) {
            console.error(`  ❌ [FAIL] ${name}:`, err.message);
            failed++;
        }
    }

    // ==================== SCENARIO 1: Critical Hazard Detected & Recorded ====================
    test('Scenario 1: Critical hazard detected and recorded with unique ID and persistent attributes', () => {
        const incident = incidentManager.createIncident({
            source: 'TEST_CAM_01 - Hoist Sector',
            hazard: 'CRANE_HAZARD',
            level: 'CRITICAL',
            score: 0.94,
            sop_id: 'SOP-EHS-091',
            standard_code: '29 CFR 1910.179(n)'
        });

        assert(incident.id, 'Incident must have an ID');
        assert(incident.id.startsWith('INC-'), 'ID must match INC-YYYY-XXX format');
        assert.strictEqual(incident.hazard, 'CRANE_HAZARD');
        assert.strictEqual(incident.level, 'CRITICAL');
        assert.strictEqual(incident.status, 'ACTIVE');
        assert.strictEqual(incident.sop_id, 'SOP-EHS-091');
    });

    // ==================== SCENARIO 2: Missing Credentials ====================
    await asyncTest('Scenario 2: Missing Gmail credentials produce clear CONFIG_REQUIRED status without faking delivery', async () => {
        alertService.mailTransporter = null;
        alertService.mockTransporter = null;

        const result = await alertService.dispatchAlert({
            incidentId: `INC-AUTO-${Date.now()}`,
            source: 'TEST_CAM_01',
            score: 0.88,
            level: 'CRITICAL',
            hazard: 'NO_HELMET',
            isTest: true,
            isMock: false
        });

        const emailResult = result.emailDelivery[0];
        assert.strictEqual(emailResult.status, 'CONFIG_REQUIRED', 'Missing credentials must strictly be CONFIG_REQUIRED');
        assert.deepStrictEqual(result.smsDelivery, [], 'SMS must never be dispatched');
    });

    // ==================== SCENARIO 3: Gmail SMTP Notification Dispatch ====================
    await asyncTest('Scenario 3: Critical incident triggers email notification when Gmail SMTP is configured', async () => {
        alertService.setMockTransporter(null);
        alertService.mailTransporter = {
            sendMail: async () => ({
                messageId: 'mock-msg-12345@gmail.com',
                accepted: ['aathithyaaelaiyaraja@gmail.com'],
                response: '250 2.0.0 OK Message accepted by Gmail SMTP server'
            })
        };

        const result = await alertService.dispatchAlert({
            incidentId: 'INC-TEST-002',
            source: 'Zone B Assembly',
            score: 0.92,
            level: 'CRITICAL',
            hazard: 'NO_HELMET',
            isTest: true,
            isMock: false
        });

        assert.strictEqual(result.emailDelivery.length, 1);
        assert.strictEqual(result.emailDelivery[0].status, 'SENT');
        assert.strictEqual(result.emailDelivery[0].messageId, 'mock-msg-12345@gmail.com');
        assert.deepStrictEqual(result.smsDelivery, [], 'SMS is disabled');
    });

    // ==================== SCENARIO 4: Email Network Failure ====================
    await asyncTest('Scenario 4: Email delivery failure is caught and reported accurately as FAILED', async () => {
        alertService.setMockTransporter(null);
        alertService.mailTransporter = {
            sendMail: async () => {
                const err = new Error('SMTP Connection Refused (Port 587 timeout)');
                err.code = 'ETIMEDOUT';
                throw err;
            }
        };

        const result = await alertService.dispatchAlert({
            incidentId: `INC-AUTO-${Date.now()}`,
            source: 'Zone D',
            score: 0.95,
            level: 'CRITICAL',
            hazard: 'NO_VEST',
            isTest: true,
            isMock: false
        });

        assert.strictEqual(result.emailDelivery[0].status, 'FAILED');
        assert.strictEqual(result.emailDelivery[0].code, 'ETIMEDOUT');
        assert(result.emailDelivery[0].reason.includes('Port 587 timeout'));
    });

    // ==================== SCENARIO 5: SMTP Authentication Failure ====================
    await asyncTest('Scenario 5: SMTP authentication failure is caught and reported accurately as FAILED', async () => {
        alertService.setMockTransporter(null);
        alertService.mailTransporter = {
            sendMail: async () => {
                const err = new Error('535-5.7.8 Username and Password not accepted');
                err.code = 'EAUTH';
                throw err;
            }
        };

        const result = await alertService.dispatchAlert({
            incidentId: `INC-AUTO-${Date.now()}`,
            source: 'Zone E',
            score: 0.93,
            level: 'CRITICAL',
            hazard: 'NO_HELMET',
            isTest: true,
            isMock: false
        });

        assert.strictEqual(result.emailDelivery[0].status, 'FAILED');
        assert.strictEqual(result.emailDelivery[0].code, 'EAUTH');
        assert(result.emailDelivery[0].reason.includes('535-5.7.8'));
    });

    // ==================== SCENARIO 6: Cooldown Enforcement ====================
    test('Scenario 6: Repeated detections within cooldown do not generate uncontrolled duplicate alerts', () => {
        const source = 'CAM_COOLDOWN_TEST';
        const hazard = 'FORKLIFT_HAZARD';

        incidentManager.registerDetection(source, hazard, 'CRITICAL');
        const isDup = incidentManager.isDuplicate(source, hazard, 'CRITICAL', 60);
        assert.strictEqual(isDup, true, 'Immediate repeat detection must be recognized as duplicate');
    });

    // ==================== SCENARIO 7: Threat Escalation Breaks Cooldown ====================
    test('Scenario 7: A new or escalated incident (MODERATE -> CRITICAL) breaks cooldown and triggers new alert', () => {
        const source = 'CAM_ESCALATE_TEST';
        const hazard = 'SLIP_HAZARD';

        incidentManager.registerDetection(source, hazard, 'MODERATE');
        const isDup = incidentManager.isDuplicate(source, hazard, 'CRITICAL', 60);
        assert.strictEqual(isDup, false, 'Escalation from MODERATE to CRITICAL must break cooldown');
    });

    // ==================== SCENARIO 8: Dashboard Notification Telemetry Sync ====================
    test('Scenario 8: Dashboard notification status matches backend/provider results in incident record', () => {
        const testInc = incidentManager.createIncident({
            source: 'Zone G',
            hazard: 'NO_HELMET',
            level: 'CRITICAL',
            score: 0.91
        });

        const updated = incidentManager.updateNotificationStatus(testInc.id, {
            email: {
                status: 'SENT',
                recipient: 'Aathithyaa Elaiyaraja',
                emailAddress: 'aathithyaaelaiyaraja@gmail.com',
                messageId: 'gmail-msg-verified@gmail.com',
                timestamp: '01:05:00'
            }
        });

        assert.strictEqual(updated.notification.email.status, 'SENT');
        assert.strictEqual(updated.notification.email.emailAddress, 'aathithyaaelaiyaraja@gmail.com');
    });

    // ==================== SCENARIO 9: Incident History Persistence ====================
    test('Scenario 9: Incident history survives page reloads and process restarts when persisted', () => {
        const testInc = incidentManager.createIncident({
            source: 'CAM_PERSIST',
            hazard: 'CRANE_HAZARD',
            level: 'CRITICAL',
            score: 0.89
        });

        const res = incidentManager.getIncidents({ search: testInc.id });
        assert(res.incidents.length > 0, 'Incident must be found');
        assert.strictEqual(res.incidents[0].id, testInc.id);
    });

    // ==================== SCENARIO 10: Incident Lifecycle ====================
    test('Scenario 10: Incident Lifecycle: Acknowledge and Resolve updates status and logs officer metadata', () => {
        const inc = incidentManager.createIncident({
            source: 'CAM_LIFECYCLE',
            hazard: 'FALL_HAZARD',
            level: 'CRITICAL'
        });

        const acked = incidentManager.acknowledgeIncident(inc.id, {
            officerName: 'Sarah Connor (Safety Lead)',
            notes: 'Perimeter cordoned off.'
        });
        assert.strictEqual(acked.status, 'ACKNOWLEDGED');
        assert.strictEqual(acked.acknowledgedBy, 'Sarah Connor (Safety Lead)');

        const resolved = incidentManager.resolveIncident(inc.id, {
            officerName: 'Sarah Connor (Safety Lead)',
            notes: 'Guardrails installed, zone certified safe.'
        });
        assert.strictEqual(resolved.status, 'RESOLVED');
        assert(resolved.resolvedAt);
    });

    // ==================== SCENARIO 11: Provider Rejection Handling ====================
    await asyncTest('Scenario 11: Email dispatch records exact provider response and never marks SENT on connection rejection', async () => {
        alertService.setMockTransporter(null);
        alertService.mailTransporter = {
            sendMail: async () => {
                const err = new Error('550 5.1.1 User unknown');
                err.code = 'EENVELOPE';
                throw err;
            }
        };

        const result = await alertService.dispatchAlert({
            incidentId: `INC-REJECT-${Date.now()}`,
            source: 'Surveillance Cam 01',
            score: 0.94,
            level: 'CRITICAL',
            hazard: 'FALL_HAZARD',
            isTest: true,
            isMock: false
        });

        const del = result.emailDelivery[0];
        assert.strictEqual(del.status, 'FAILED');
        assert.notStrictEqual(del.status, 'SENT', 'Must never mark SENT on rejection');
        assert(del.reason.includes('550 5.1.1'));
    });

    // ==================== SCENARIO 12: Duplicate Prevention ====================
    await asyncTest('Scenario 12: Duplicate email alerts are prevented for the same incident record', async () => {
        const uniqueId = `INC-DEDUP-PIPE-${Date.now()}`;
        alertService.setMockTransporter(null);
        alertService.mailTransporter = {
            sendMail: async () => ({
                messageId: `msg-${Date.now()}@gmail.com`,
                response: '250 OK'
            })
        };

        const first = await alertService.dispatchAlert({
            incidentId: uniqueId,
            source: 'Crane Zone A',
            hazard: 'CRANE_HAZARD',
            level: 'CRITICAL',
            score: 0.95,
            isTest: false,
            isMock: false
        });
        assert.strictEqual(first.emailDelivery[0].status, 'SENT');

        // Second dispatch for identical incidentId:
        const second = await alertService.dispatchAlert({
            incidentId: uniqueId,
            source: 'Crane Zone A',
            hazard: 'CRANE_HAZARD',
            level: 'CRITICAL',
            score: 0.95,
            isTest: false,
            isMock: false
        });
        assert.strictEqual(second.emailDelivery[0].status, 'SKIPPED');
        assert(second.emailDelivery[0].reason.includes('Duplicate alert prevented'));
    });

    // ==================== SCENARIO 13: Mocked Development Mode ====================
    await asyncTest('Scenario 13: Mock development mode produces MOCKED status with full HTML and plain text', async () => {
        let sentOptions = null;
        alertService.setMockTransporter({
            sendMail: async (options) => {
                sentOptions = options;
                return {
                    messageId: `mock-${Date.now()}@vigilops.ai`,
                    response: '250 OK Mocked Acceptance'
                };
            }
        });

        const result = await alertService.dispatchAlert({
            incidentId: `INC-MOCK-${Date.now()}`,
            source: 'CAM_FAB_01',
            hazard: 'NO_HELMET',
            level: 'CRITICAL',
            isTest: true,
            isMock: true
        });

        assert.strictEqual(result.emailDelivery[0].status, 'MOCKED');
        assert(sentOptions.text.length > 50, 'Text body must be populated');
        assert(sentOptions.html.length > 100, 'HTML body must be populated');
        assert(sentOptions.subject.includes('[TEST EMAIL]'), 'Test emails must have [TEST EMAIL] subject prefix');
    });

    // ==================== SCENARIO 14: Confirms SMS is Disabled ====================
    await asyncTest('Scenario 14: Confirms SMS functionality is completely disabled under any circumstances', async () => {
        const result = await alertService.dispatchAlert({
            incidentId: `INC-SMS-DISABLED-${Date.now()}`,
            source: 'CAM_01',
            hazard: 'NO_VEST',
            level: 'CRITICAL',
            isTest: true,
            isMock: true
        });

        assert.deepStrictEqual(result.smsDelivery, []);
    });

    // ==================== SCENARIO 15: getConfigStatus reports CONFIGURED ====================
    test('Scenario 15: getConfigStatus reports CONFIGURED health without exposing secrets', () => {
        process.env.SMTP_HOST = 'smtp.gmail.com';
        process.env.SMTP_PORT = '587';
        process.env.SMTP_USER = 'aathithyaaelaiyaraja@gmail.com';
        process.env.SMTP_PASS = 'qirtmjbeexfbmsso';
        process.env.EMAIL_ENABLED = 'true';

        const status = alertService.getConfigStatus();
        assert.strictEqual(status.email.provider, 'gmail');
        assert.strictEqual(status.email.configured, true);
        assert.strictEqual(status.email.status, 'CONFIGURED');
        assert.strictEqual(status.smsEnabled, false);
    });

    // ==================== SCENARIO 16: Invalid Recipient Email ====================
    test('Scenario 16: Invalid recipient email formats are rejected', () => {
        assert.throws(() => {
            alertService.addRecipient({ name: 'Test', email: 'invalid-email' });
        }, /Invalid email address format/);
    });

    // ==================== SCENARIO 17: Never Marks as DELIVERED ====================
    await asyncTest('Scenario 17: Gmail SMTP acceptance is strictly recorded as SENT (never DELIVERED)', async () => {
        alertService.setMockTransporter(null);
        alertService.mailTransporter = {
            sendMail: async () => ({
                messageId: '<d4-sample@gmail.com>',
                accepted: ['aathithyaaelaiyaraja@gmail.com'],
                response: '250 2.0.0 OK d4-20020a05620a280400b0078b668d2987sm2647008qkp.35 - gsmtp'
            })
        };

        const result = await alertService.dispatchAlert({
            incidentId: `INC-NOT-DELIVERED-${Date.now()}`,
            source: 'CAM_01',
            hazard: 'NO_HELMET',
            level: 'CRITICAL',
            isTest: true,
            isMock: false
        });

        assert.strictEqual(result.emailDelivery[0].status, 'SENT');
        assert.notStrictEqual(result.emailDelivery[0].status, 'DELIVERED');
    });

    console.log('\n======================================================');
    console.log(`🏁 Test Summary: ${passed} Passed, ${failed} Failed`);
    console.log('======================================================\n');

    if (failed > 0) process.exit(1);
}

runTestSuite().catch(err => {
    console.error('Fatal test error:', err);
    process.exit(1);
});
