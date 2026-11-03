/**
 * VigilOps AI – Gmail SMTP Exclusive Notification Test Suite
 * Tests all requirements from "VigilOps AI – Gmail-Only Notification Integration":
 * 1. Missing SMTP credentials -> CONFIG_REQUIRED
 * 2. Successful SMTP authentication -> CONFIGURED & config health check
 * 3. Email sending -> SENT/MOCKED with HTML + Plain text fallback & all required fields
 * 4. SMTP authentication failure / rejection -> FAILED with error code
 * 5. Invalid recipient handling -> Rejects invalid emails
 * 6. Duplicate incident notification prevention -> SKIPPED
 * 7. Dashboard synchronization -> Incident record status update
 * 8. Zero SMS dispatches under any circumstances -> smsDelivery is []
 * 9. SMTP acceptance is never labeled DELIVERED
 */

process.env.NODE_ENV = 'test';

const assert = require('assert');
const alertService = require('../alertService');
const incidentManager = require('../incidentManager');

let passedTests = 0;
let failedTests = 0;

function runTest(testName, testFn) {
    try {
        testFn();
        console.log(`  ✅ [PASS] ${testName}`);
        passedTests++;
    } catch (err) {
        console.error(`  ❌ [FAIL] ${testName}: ${err.message}`);
        failedTests++;
    }
}

async function runAsyncTest(testName, testFn) {
    try {
        await testFn();
        console.log(`  ✅ [PASS] ${testName}`);
        passedTests++;
    } catch (err) {
        console.error(`  ❌ [FAIL] ${testName}: ${err.message}`);
        failedTests++;
    }
}

async function runTestSuite() {
    console.log('🧪 Starting VigilOps AI Gmail-Only Notification Test Suite...\n');

    // =========================================================================
    // TEST 1: Missing SMTP credentials produces CONFIG_REQUIRED
    // =========================================================================
    await runAsyncTest('Test 1: Missing SMTP credentials produce CONFIG_REQUIRED', async () => {
        alertService.mailTransporter = null;
        alertService.mockTransporter = null;

        const result = await alertService.dispatchAlert({
            incidentId: `INC-GMAIL-CFG-${Date.now()}`,
            source: 'CAM_01 - Fabrication Bay',
            hazard: 'NO_HELMET',
            level: 'CRITICAL',
            score: 0.92,
            isTest: true,
            isMock: false
        });

        assert(result.emailDelivery.length > 0, 'Should have email delivery records');
        const delivery = result.emailDelivery[0];
        assert.strictEqual(delivery.status, 'CONFIG_REQUIRED', 'Missing credentials must produce CONFIG_REQUIRED');
        assert(delivery.reason.includes('credentials') || delivery.reason.includes('SMTP'), 'Should provide clear reason');
    });

    // =========================================================================
    // TEST 2: Successful SMTP configuration reports CONFIGURED & zero credentials
    // =========================================================================
    runTest('Test 2: getConfigStatus reports CONFIGURED health without disclosing secrets', () => {
        process.env.SMTP_HOST = 'smtp.gmail.com';
        process.env.SMTP_PORT = '587';
        process.env.SMTP_SECURE = 'false';
        process.env.SMTP_USER = 'aathithyaaelaiyaraja@gmail.com';
        process.env.SMTP_PASS = 'qirtmjbeexfbmsso';
        process.env.EMAIL_ENABLED = 'true';

        const config = alertService.getConfigStatus();
        assert.strictEqual(config.email.provider, 'gmail');
        assert.strictEqual(config.email.host, 'smtp.gmail.com');
        assert.strictEqual(config.email.port, 587);
        assert.strictEqual(config.email.secure, false);
        assert.strictEqual(config.email.status, 'CONFIGURED');
        assert.strictEqual(config.email.configured, true);
        assert.strictEqual(config.smsEnabled, false, 'SMS must be disabled');
        assert(!('SMTP_PASS' in config.email), 'Never disclose SMTP_PASS');
        assert(!('pass' in config.email), 'Never disclose pass');
    });

    // =========================================================================
    // TEST 3: Email sending in mocked development mode with required content
    // =========================================================================
    await runAsyncTest('Test 3: Mocked email sending formats all 9 required fields + text fallback', async () => {
        let capturedMail = null;
        alertService.setMockTransporter({
            sendMail: async (options) => {
                capturedMail = options;
                return {
                    messageId: `gmail-mock-${Date.now()}@gmail.com`,
                    response: '250 2.0.0 OK Message queued for delivery'
                };
            }
        });

        const testIncId = `INC-GMAIL-REQ4-${Date.now()}`;
        const result = await alertService.dispatchAlert({
            incidentId: testIncId,
            source: 'CAM_02 - Overhead Crane Gantry',
            location: 'Building 4 Assembly Bay',
            hazard: 'CRANE_HAZARD',
            level: 'CRITICAL',
            score: 0.96,
            protocol: 'Halt gantry hoist traverse and clear drop zone immediately.',
            sop_details: {
                sop_id: 'SOP-EHS-091',
                standard_code: '29 CFR 1910.179',
                summary: 'Standard operating protocol for overhead crane hoist safety.',
                emergency_action: 'Trigger emergency crane stop button and establish 5-meter exclusion radius.'
            },
            isTest: false,
            isMock: true
        });

        const delivery = result.emailDelivery[0];
        assert.strictEqual(delivery.status, 'MOCKED', 'Mock dispatch must produce MOCKED status');
        assert(delivery.messageId.startsWith('gmail-mock-'), 'Must record messageId');

        // Requirement 4: Subject verification
        assert.strictEqual(capturedMail.subject, `[VigilOps AI] CRITICAL SAFETY ALERT – Incident ID: ${testIncId}`);

        // Requirement 4: Body content verification
        const text = capturedMail.text;
        const html = capturedMail.html;
        assert(text, 'Must provide readable plain-text fallback');
        assert(html, 'Must provide professional HTML template');

        assert(text.includes(testIncId), 'Text must contain Incident ID');
        assert(text.includes('CRANE_HAZARD'), 'Text must contain Detected hazard');
        assert(text.includes('CRITICAL'), 'Text must contain Risk level');
        assert(text.includes('96%'), 'Text must contain Threat score');
        assert(text.includes('CAM_02 - Overhead Crane Gantry'), 'Text must contain Camera / source');
        assert(text.includes('Building 4 Assembly Bay'), 'Text must contain Location');
        assert(text.includes('Trigger emergency crane stop button'), 'Text must contain Recommended action');
        assert(text.includes('http://localhost:5173'), 'Text must contain Dashboard reference');

        assert(html.includes(testIncId), 'HTML must contain Incident ID');
        assert(html.includes('CRANE_HAZARD'), 'HTML must contain Detected hazard');
        assert(html.includes('Trigger emergency crane stop button'), 'HTML must contain Recommended action');
    });

    // =========================================================================
    // TEST 4: SMTP authentication failure or network rejection produces FAILED
    // =========================================================================
    await runAsyncTest('Test 4: SMTP authentication rejection produces FAILED status with error code', async () => {
        alertService.setMockTransporter({
            sendMail: async () => {
                const err = new Error('535-5.7.8 Username and Password not accepted');
                err.code = 'EAUTH';
                throw err;
            }
        });

        const result = await alertService.dispatchAlert({
            incidentId: `INC-AUTH-ERR-${Date.now()}`,
            source: 'CAM_01',
            hazard: 'NO_HELMET',
            level: 'CRITICAL',
            score: 0.88,
            isTest: true,
            isMock: true
        });

        const delivery = result.emailDelivery[0];
        assert.strictEqual(delivery.status, 'FAILED', 'Authentication rejection must produce FAILED');
        assert.strictEqual(delivery.code, 'EAUTH');
        assert(delivery.reason.includes('535-5.7.8'), 'Must capture exact error message');
    });

    // =========================================================================
    // TEST 5: Invalid recipient handling rejects invalid email formats
    // =========================================================================
    runTest('Test 5: addRecipient rejects invalid email formats', () => {
        assert.throws(() => {
            alertService.addRecipient({ name: 'Bad User', email: 'not-an-email' });
        }, /Invalid email address format/);

        assert.throws(() => {
            alertService.addRecipient({ name: 'Empty User', email: '' });
        }, /valid safety administrator email/);
    });

    // =========================================================================
    // TEST 6: Duplicate incident notification prevention produces SKIPPED
    // =========================================================================
    await runAsyncTest('Test 6: Duplicate incident notification is prevented and marked SKIPPED', async () => {
        const uniqueIncId = `INC-DEDUP-TEST-${Date.now()}`;
        alertService.setMockTransporter({
            sendMail: async () => ({
                messageId: `msg-${Date.now()}@gmail.com`,
                response: '250 OK'
            })
        });

        // First dispatch: Succeeds
        const first = await alertService.dispatchAlert({
            incidentId: uniqueIncId,
            source: 'CAM_03 - Loading Bay',
            hazard: 'FORKLIFT_HAZARD',
            level: 'CRITICAL',
            score: 0.90,
            isTest: false,
            isMock: true
        });
        assert.strictEqual(first.emailDelivery[0].status, 'MOCKED');

        // Second dispatch for SAME incidentId: Must be prevented (SKIPPED)
        const second = await alertService.dispatchAlert({
            incidentId: uniqueIncId,
            source: 'CAM_03 - Loading Bay',
            hazard: 'FORKLIFT_HAZARD',
            level: 'CRITICAL',
            score: 0.90,
            isTest: false,
            isMock: true
        });
        assert.strictEqual(second.emailDelivery[0].status, 'SKIPPED', 'Duplicate alert must be marked SKIPPED');
        assert(second.emailDelivery[0].reason.includes('Duplicate alert prevented'));
    });

    // =========================================================================
    // TEST 7: Dashboard synchronization updates incident record
    // =========================================================================
    runTest('Test 7: Incident record stores email notification state in incidentManager', () => {
        const incident = incidentManager.createIncident({
            source: 'CAM_TEST_SYNC',
            hazard: 'FALL_HAZARD',
            level: 'CRITICAL',
            score: 0.95
        });

        const updated = incidentManager.updateNotificationStatus(incident.id, {
            email: {
                status: 'SENT',
                recipient: 'Safety Officer',
                emailAddress: 'safety@plant.com',
                messageId: 'mock-msg-123@gmail.com',
                timestamp: '01:00:00'
            }
        });

        assert.strictEqual(updated.notification.email.status, 'SENT');
        assert.strictEqual(updated.notification.email.emailAddress, 'safety@plant.com');
        assert.strictEqual(updated.notification.email.messageId, 'mock-msg-123@gmail.com');
    });

    // =========================================================================
    // TEST 8: Zero SMS functionality under any circumstances
    // =========================================================================
    await runAsyncTest('Test 8: Confirms SMS is never dispatched under any circumstances', async () => {
        alertService.setMockTransporter({
            sendMail: async () => ({ messageId: 'msg@gmail.com', response: '250 OK' })
        });

        const result = await alertService.dispatchAlert({
            incidentId: `INC-NO-SMS-${Date.now()}`,
            source: 'CAM_01',
            hazard: 'NO_HELMET',
            level: 'CRITICAL',
            isTest: false,
            isMock: true
        });

        assert.deepStrictEqual(result.smsDelivery, [], 'smsDelivery must strictly be an empty array');
        assert(!('smsProvider' in alertService) || alertService.smsProvider === undefined, 'smsProvider must not exist');
    });

    // =========================================================================
    // TEST 9: Strictly never labels SMTP acceptance as DELIVERED
    // =========================================================================
    await runAsyncTest('Test 9: SMTP acceptance is strictly recorded as SENT (never DELIVERED)', async () => {
        let sentOptions = null;
        alertService.setMockTransporter(null); // Real transporter interface simulation
        alertService.mailTransporter = {
            sendMail: async (options) => {
                sentOptions = options;
                return {
                    messageId: '<d4-sample@gmail.com>',
                    accepted: ['aathithyaaelaiyaraja@gmail.com'],
                    response: '250 2.0.0 OK d4-20020a05620a280400b0078b668d2987sm2647008qkp.35 - gsmtp'
                };
            }
        };

        const result = await alertService.dispatchAlert({
            incidentId: `INC-STATUS-TEST-${Date.now()}`,
            source: 'CAM_01',
            hazard: 'NO_HELMET',
            level: 'CRITICAL',
            isTest: false,
            isMock: false
        });

        const delivery = result.emailDelivery[0];
        assert.strictEqual(delivery.status, 'SENT', 'Must be SENT, not DELIVERED');
        assert.notStrictEqual(delivery.status, 'DELIVERED', 'Gmail SMTP acceptance does not confirm recipient inbox delivery');
        assert(delivery.providerResponse.includes('250 2.0.0 OK'));
    });

    console.log('\n======================================================');
    console.log(`🏁 Gmail-Only Notification Test Summary: ${passedTests} Passed, ${failedTests} Failed`);
    console.log('======================================================\n');

    if (failedTests > 0) {
        process.exit(1);
    }
}

runTestSuite().catch(err => {
    console.error('Fatal test error:', err);
    process.exit(1);
});
