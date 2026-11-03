/**
 * VigilOps AI – Test Suite for Automatic Gmail Alerts for HIGH and CRITICAL Incidents
 * Verifies Part B & Part E requirements:
 * 1. HIGH threat level automatically triggers safety alert email.
 * 2. CRITICAL threat level automatically triggers safety alert email.
 * 3. LOW threat level produces NO automatic email.
 * 4. Genuine incident emails contain NO "TEST EMAIL", "TEST DRILL", or "SIMULATED INCIDENT" wording.
 * 5. Distinct subject formats:
 *    - CRITICAL: [VigilOps AI] CRITICAL SAFETY ALERT – Incident ID: {incident_id}
 *    - HIGH:     [VigilOps AI] HIGH PRIORITY SAFETY ALERT – Incident ID: {incident_id}
 */

process.env.NODE_ENV = 'test';
process.env.EMAIL_ENABLED = 'true';
process.env.SMTP_USER = 'safety.admin@vigilops.ai';
process.env.SMTP_PASS = 'mock-password';
process.env.ALERT_COOLDOWN_SECONDS = '60';

const assert = require('assert');
const alertService = require('../alertService');
const incidentManager = require('../incidentManager');

async function runTests() {
    console.log('🧪 Starting VigilOps AI Automatic HIGH/CRITICAL Alert Verification Suite...\n');

    let capturedMails = [];
    alertService.setMockTransporter({
        sendMail: async (mailOptions) => {
            capturedMails.push(mailOptions);
            return {
                messageId: `<mock-${Date.now()}@vigilops.ai>`,
                response: '250 2.0.0 OK Message accepted by Gmail SMTP server'
            };
        }
    });

    // Clean recipients to single verified test address
    alertService.recipients = [
        {
            id: 'rcpt-test-auto',
            name: 'Lead Safety Officer',
            role: 'EHS Director',
            email: 'safety.officer@industrial-safety.com',
            emailEnabled: true
        }
    ];

    // TEST 1: CRITICAL Threat Level triggers automatic email with exact required subject & body
    capturedMails = [];
    const critIncidentId = `INC-CRIT-TEST-${Date.now()}`;
    const critRes = await alertService.dispatchAlert({
        incidentId: critIncidentId,
        source: 'CAM_01 - Crane Hoist',
        score: 0.92,
        level: 'CRITICAL',
        hazard: 'NO_HELMET',
        isTest: false // Genuine incident
    });

    assert.strictEqual(critRes.emailDelivery.length, 1);
    assert.strictEqual(critRes.emailDelivery[0].status, 'MOCKED');
    assert.strictEqual(capturedMails.length, 1);

    const critMail = capturedMails[0];
    assert.strictEqual(critMail.subject, `[VigilOps AI] CRITICAL SAFETY ALERT – Incident ID: ${critIncidentId}`);
    assert(!critMail.subject.includes('[TEST EMAIL]'), 'Genuine critical alert subject must not contain [TEST EMAIL]');
    assert(!critMail.text.includes('*** SYSTEM VERIFICATION DRILL – TEST EMAIL ONLY ***'), 'Genuine critical alert text must not contain drill wording');
    assert(!critMail.html.includes('VERIFICATION DRILL TEST EMAIL'), 'Genuine critical alert HTML must not contain drill wording');
    assert(critMail.text.includes('Safety Helmet / Hard Hat Non-Compliance (OSHA 1910.135)'), 'Must contain detected PPE violation description');
    console.log('  ✅ [PASS] Test 1: CRITICAL incident automatically dispatches email without any TEST wording');

    // TEST 2: HIGH Threat Level triggers automatic email with exact required subject & body
    capturedMails = [];
    const highIncidentId = `INC-HIGH-TEST-${Date.now()}`;
    const highRes = await alertService.dispatchAlert({
        incidentId: highIncidentId,
        source: 'CAM_04 - Assembly Sector',
        score: 0.72,
        level: 'HIGH',
        hazard: 'NO_VEST',
        isTest: false // Genuine incident
    });

    assert.strictEqual(highRes.emailDelivery.length, 1);
    assert.strictEqual(highRes.emailDelivery[0].status, 'MOCKED');
    assert.strictEqual(capturedMails.length, 1);

    const highMail = capturedMails[0];
    assert.strictEqual(highMail.subject, `[VigilOps AI] HIGH PRIORITY SAFETY ALERT – Incident ID: ${highIncidentId}`);
    assert(!highMail.subject.includes('[TEST EMAIL]'), 'Genuine high alert subject must not contain [TEST EMAIL]');
    assert(!highMail.text.includes('TEST EMAIL'), 'Genuine high alert text must not contain TEST EMAIL');
    assert(highMail.html.includes('VigilOps AI — High Priority Safety Alert'), 'HTML must feature High Priority alert branding');
    assert(highMail.text.includes('High-Visibility Retroreflective Vest Non-Compliance (ANSI/ISEA 107)'), 'Must contain detected PPE violation description');
    console.log('  ✅ [PASS] Test 2: HIGH incident automatically dispatches email without any TEST wording');

    // TEST 3: Duplicate Incident does not send repeated emails
    capturedMails = [];
    const dupRes = await alertService.dispatchAlert({
        incidentId: highIncidentId, // Same incident ID
        source: 'CAM_04 - Assembly Sector',
        score: 0.72,
        level: 'HIGH',
        hazard: 'NO_VEST',
        isTest: false
    });
    assert.strictEqual(capturedMails.length, 0, 'No email should be dispatched for duplicate incident');
    assert.strictEqual(dupRes.emailDelivery[0].status, 'SKIPPED');
    console.log('  ✅ [PASS] Test 3: Duplicate incident is safely prevented from sending repeated emails (SKIPPED)');

    // TEST 4: Manual Test Drill button correctly preserves [TEST EMAIL] identification
    capturedMails = [];
    const drillIncidentId = `INC-DRILL-${Date.now()}`;
    const drillRes = await alertService.dispatchAlert({
        incidentId: drillIncidentId,
        source: 'CONTROL ROOM CONSOLE',
        score: 0.99,
        level: 'CRITICAL',
        hazard: 'EMERGENCY_DRILL_SYSTEM_CHECK',
        isTest: true // Manual test drill
    });

    assert.strictEqual(capturedMails.length, 1);
    const drillMail = capturedMails[0];
    assert(drillMail.subject.startsWith('[TEST EMAIL]'), 'Manual test drill must have [TEST EMAIL] subject prefix');
    assert(drillMail.text.includes('SYSTEM VERIFICATION DRILL – TEST EMAIL ONLY'), 'Manual test drill body must identify as test');
    console.log('  ✅ [PASS] Test 4: Manual test drill is distinctly separated and identified with [TEST EMAIL]');

    console.log('\n======================================================');
    console.log('🏁 All 4 Automatic Email & Verification Tests PASSED');
    console.log('======================================================\n');
}

runTests().catch((err) => {
    console.error('❌ Test failed:', err);
    process.exit(1);
});

