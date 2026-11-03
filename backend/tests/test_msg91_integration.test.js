/**
 * VigilOps AI - MSG91 SMS Gateway Integration Test Suite
 * Validates MSG91 v5 Flow API dispatching, delivery-report webhooks,
 * state machine transitions (CONFIG_REQUIRED, SENT, DELIVERED, FAILED, SKIPPED),
 * and duplicate alert prevention.
 */

const assert = require('assert');
const EventEmitter = require('events');
const http = require('http');
const https = require('https');

process.env.NODE_ENV = 'test';
process.env.SMS_PROVIDER = 'msg91';

const alertService = require('../alertService');
const incidentManager = require('../incidentManager');

async function runMSG91Suite() {
    console.log('🧪 Starting MSG91 SMS Gateway Dedicated Test Suite...\n');
    let passed = 0;
    let failed = 0;

    async function test(name, fn) {
        try {
            await fn();
            console.log(`  ✅ [PASS] ${name}`);
            passed++;
        } catch (err) {
            console.error(`  ❌ [FAIL] ${name}:`, err.message);
            failed++;
        }
    }

    // ==================== TEST 1: Missing Credentials Guard ====================
    await test('Test 1: Unconfigured MSG91 credentials produce CONFIG_REQUIRED without mock delivery', async () => {
        delete process.env.MSG91_AUTH_KEY;
        delete process.env.MSG91_TEMPLATE_ID;
        alertService.smsProvider = 'msg91';

        const dispatch = await alertService.dispatchAlert({
            incidentId: 'INC-MSG91-GUARD-' + Date.now(),
            hazard: 'NO_HELMET',
            source: 'Fabrication Bay 1',
            score: 0.93,
            isTest: true
        });

        assert(dispatch.smsDelivery.length > 0, 'Must record SMS attempts');
        const entry = dispatch.smsDelivery[0];
        assert.strictEqual(entry.provider, 'msg91');
        assert.strictEqual(entry.status, 'CONFIG_REQUIRED');
        assert(entry.reason.includes('MSG91 credentials'), 'Reason must specify missing MSG91 credentials');
    });

    // ==================== TEST 2: Successful API Submission (SENT status) ====================
    await test('Test 2: MSG91 v5 Flow API dispatches payload and marks SENT (never DELIVERED prematurely)', async () => {
        process.env.MSG91_AUTH_KEY = 'valid_mock_auth_key_12345';
        process.env.MSG91_TEMPLATE_ID = 'flow_template_ehs_001';
        process.env.MSG91_SENDER_ID = 'VIGOPS';
        process.env.MSG91_ROUTE = '4';
        alertService.smsProvider = 'msg91';

        let capturedHeaders = null;
        let capturedPayload = null;
        const testRequestId = 'msg91-req-id-' + Date.now();

        const originalRequest = https.request;
        https.request = (opts, cb) => {
            capturedHeaders = opts.headers;
            const req = new EventEmitter();
            req.write = (chunk) => {
                try { capturedPayload = JSON.parse(chunk); } catch (e) { capturedPayload = chunk; }
            };
            req.end = () => {
                const res = new EventEmitter();
                res.statusCode = 200;
                process.nextTick(() => {
                    cb(res);
                    res.emit('data', JSON.stringify({
                        type: 'success',
                        message: testRequestId
                    }));
                    res.emit('end');
                });
            };
            return req;
        };

        const targetIncident = 'INC-MSG91-API-' + Date.now();
        const dispatch = await alertService.dispatchAlert({
            incidentId: targetIncident,
            hazard: 'CRANE_HAZARD',
            source: 'Overhead Gantry 2',
            score: 0.97,
            isTest: true
        });

        https.request = originalRequest;

        // Verify HTTP Headers
        assert.strictEqual(capturedHeaders['authkey'], 'valid_mock_auth_key_12345');
        assert.strictEqual(capturedHeaders['content-type'], 'application/json');

        // Verify Payload
        assert.strictEqual(capturedPayload.template_id, 'flow_template_ehs_001');
        assert.strictEqual(capturedPayload.sender, 'VIGOPS');
        assert.strictEqual(capturedPayload.route, '4');
        assert.strictEqual(capturedPayload.recipients[0].mobiles, '916379864357');

        // Verify Delivery State: MUST be SENT, NOT DELIVERED
        const entry = dispatch.smsDelivery[0];
        assert.strictEqual(entry.status, 'SENT', 'Must be SENT on API acceptance');
        assert.strictEqual(entry.provider, 'msg91');
        assert.strictEqual(entry.requestId, testRequestId);
        assert.strictEqual(entry.messageId, testRequestId);
    });

    // ==================== TEST 3: Webhook Transitions SENT -> DELIVERED ====================
    await test('Test 3: MSG91 carrier delivery webhook transitions record to DELIVERED', async () => {
        const testReqId = 'msg91-webhook-deliv-' + Date.now();
        const testInc = 'INC-MSG91-DELIV-' + Date.now();

        // Seed a SENT record in dispatch history
        alertService.dispatchHistory.unshift({
            id: 'DISPATCH-TEST-DELIV',
            incidentId: testInc,
            timestamp: '11:45:00 PM',
            smsDelivery: [
                {
                    recipient: 'Aathithyaa Elaiyaraja',
                    phone: '+916379864357',
                    provider: 'msg91',
                    status: 'SENT',
                    requestId: testReqId,
                    messageId: testReqId
                }
            ]
        });

        // Trigger webhook delivery report
        const result = alertService.updateDeliveryStatus(testReqId, 'DELIVERED', {
            desc: 'DELIVERED',
            status: '1',
            number: '916379864357',
            deliveryTime: '2026-11-02 23:45:10'
        });

        assert(result, 'updateDeliveryStatus must find and update the record');
        assert.strictEqual(result.mappedStatus, 'DELIVERED');
        assert.strictEqual(result.entry.status, 'DELIVERED');
        assert.strictEqual(result.entry.desc, 'DELIVERED');
    });

    // ==================== TEST 4: Webhook Transitions SENT -> FAILED ====================
    await test('Test 4: MSG91 carrier failure webhook (e.g. NDNC / Blocked) transitions record to FAILED', async () => {
        const testReqId = 'msg91-webhook-fail-' + Date.now();
        const testInc = 'INC-MSG91-FAIL-' + Date.now();

        // Seed a SENT record
        alertService.dispatchHistory.unshift({
            id: 'DISPATCH-TEST-FAIL',
            incidentId: testInc,
            timestamp: '11:46:00 PM',
            smsDelivery: [
                {
                    recipient: 'Aathithyaa Elaiyaraja',
                    phone: '+916379864357',
                    provider: 'msg91',
                    status: 'SENT',
                    requestId: testReqId,
                    messageId: testReqId
                }
            ]
        });

        const result = alertService.updateDeliveryStatus(testReqId, 'FAILED', {
            desc: 'NDNC',
            status: '9',
            errorCode: 'NDNC_FILTERED'
        });

        assert(result, 'updateDeliveryStatus must find and update the record');
        assert.strictEqual(result.mappedStatus, 'FAILED');
        assert.strictEqual(result.entry.status, 'FAILED');
        assert.strictEqual(result.entry.errorCode, 'NDNC_FILTERED');
    });

    // ==================== TEST 5: Duplicate SMS Prevention ====================
    await test('Test 5: Duplicate SMS alerts are prevented for the same incident record', async () => {
        process.env.MSG91_AUTH_KEY = 'mock_key';
        process.env.MSG91_TEMPLATE_ID = 'mock_tpl';
        alertService.smsProvider = 'msg91';

        let callCount = 0;
        const originalRequest = https.request;
        https.request = (opts, cb) => {
            callCount++;
            const req = new EventEmitter();
            req.write = () => {};
            req.end = () => {
                const res = new EventEmitter();
                res.statusCode = 200;
                process.nextTick(() => {
                    cb(res);
                    res.emit('data', JSON.stringify({ type: 'success', message: 'req-dedup-call' }));
                    res.emit('end');
                });
            };
            return req;
        };

        const incidentId = 'INC-MSG91-DEDUP-' + Date.now();

        // Call 1: First dispatch
        const dispatch1 = await alertService.dispatchAlert({
            incidentId,
            hazard: 'FALL_HAZARD',
            source: 'Tower 4',
            score: 0.94,
            isTest: false
        });
        assert.strictEqual(dispatch1.smsDelivery[0].status, 'SENT');
        assert.strictEqual(callCount, 1, 'First dispatch must call MSG91 API');

        // Call 2: Duplicate dispatch for same incident
        const dispatch2 = await alertService.dispatchAlert({
            incidentId,
            hazard: 'FALL_HAZARD',
            source: 'Tower 4',
            score: 0.96,
            isTest: false
        });

        https.request = originalRequest;

        assert.strictEqual(callCount, 1, 'Duplicate dispatch must be blocked without calling provider');
        assert.strictEqual(dispatch2.smsDelivery[0].status, 'SKIPPED');
        assert(dispatch2.smsDelivery[0].reason.includes('Duplicate alert prevented'));
    });

    // ==================== TEST 6: MSG91 Provider Error Mapping ====================
    await test('Test 6: MSG91 provider rejection (e.g. Code 400 Template Error) maps accurately to FAILED', async () => {
        process.env.MSG91_AUTH_KEY = 'auth_key_123';
        process.env.MSG91_TEMPLATE_ID = 'bad_template';
        alertService.smsProvider = 'msg91';

        const originalRequest = https.request;
        https.request = (opts, cb) => {
            const req = new EventEmitter();
            req.write = () => {};
            req.end = () => {
                const res = new EventEmitter();
                res.statusCode = 400;
                process.nextTick(() => {
                    cb(res);
                    res.emit('data', JSON.stringify({
                        type: 'error',
                        message: 'Template ID in your API request is missing or incorrect.',
                        code: 400
                    }));
                    res.emit('end');
                });
            };
            return req;
        };

        const dispatch = await alertService.dispatchAlert({
            incidentId: 'INC-MSG91-ERR-' + Date.now(),
            hazard: 'NO_HELMET',
            source: 'Zone 1',
            score: 0.90,
            isTest: true
        });

        https.request = originalRequest;

        const entry = dispatch.smsDelivery[0];
        assert.strictEqual(entry.status, 'FAILED');
        assert.strictEqual(entry.errorCode, 400);
        assert(entry.reason.includes('Template ID in your API request is missing or incorrect'));
    });

    // ==================== TEST 7: Config Status Validator Security ====================
    await test('Test 7: getConfigStatus reports provider health without ever disclosing secrets', async () => {
        process.env.MSG91_AUTH_KEY = 'secret_private_key_do_not_leak';
        process.env.MSG91_TEMPLATE_ID = 'template_public_id';
        process.env.MSG91_SENDER_ID = 'VIGOPS';
        process.env.MSG91_ROUTE = '4';

        const config = alertService.getConfigStatus();
        const jsonStr = JSON.stringify(config);

        // Security assertion: AUTH_KEY MUST NEVER APPEAR IN OUTPUT
        assert(!jsonStr.includes('secret_private_key_do_not_leak'), 'AUTH_KEY must NOT be exposed in config status');
        assert.strictEqual(config.sms.activeProvider, 'msg91');
        assert.strictEqual(config.sms.msg91.configured, true);
        assert.strictEqual(config.sms.msg91.senderId, 'VIGOPS');
        assert.strictEqual(config.sms.msg91.route, '4');
        assert.strictEqual(config.sms.msg91.status, 'CONFIGURED');
    });

    console.log(`\n======================================================`);
    console.log(`🏁 MSG91 Test Summary: ${passed} Passed, ${failed} Failed`);
    console.log(`======================================================\n`);

    if (failed > 0) process.exit(1);
}

runMSG91Suite();

