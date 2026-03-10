let allRuns = [];
let lastTestCount = -1;
let lastCompletedTestsCount = -1;
let currentViewMode = 'training';

function changeViewMode(mode) {
    currentViewMode = mode;
    lastTestCount = -1;
    lastCompletedTestsCount = -1;
    refreshLogs();
}

function formatVal(val, key) {

    if (val === 'N/A' || val === undefined) return 'N/A';

    // Round decimals to two digits
    if (typeof val === 'number' && val % 1 !== 0) {
        return val.toFixed(2);
    }

    // Format 'parallelism_type'
    if (key === 'parallelism_type' && typeof val === 'string') {
        let formatted = val.replace(/_/g, ' ').replace(/parallel/g, 'parallelism');;
        return formatted.charAt(0).toUpperCase() + formatted.slice(1);
    }

    return val;
}

async function updateSavedOptions() {
    const modelType = document.getElementById('parallelism-input').value;
    const savedSelect = document.getElementById('saved');

    savedSelect.innerHTML = '<option value="no">Train a new model and then test</option>';

    if (!modelType) return;

    try {
        const res = await fetch(`/api/archives/${modelType}`);
        const runs = await res.json();

        if (runs && runs.length > 0) {
            const group = document.createElement('optgroup');
            group.label = "Load saved model from:";

            runs.forEach(run => {
                const opt = document.createElement('option');
                opt.value = run.path;
                opt.textContent = `${run.timestamp}`;
                group.appendChild(opt);
            });
            savedSelect.appendChild(group);
        }
    } catch (err) {
        console.error("Error fetching archives:", err);
    }
}

document.addEventListener("DOMContentLoaded", () => {
    const pSelect = document.getElementById('parallelism-input');
    const mSelect = document.getElementById('model-input');
    if (pSelect) pSelect.addEventListener('change', updateSavedOptions);
    if (mSelect) mSelect.addEventListener('change', updateSavedOptions);

    const btnClearLatest = document.getElementById('btn-clear-latest');
    const btnClearHistory = document.getElementById('btn-clear-history');
    if (btnClearLatest) btnClearLatest.addEventListener('click', () => clearLog('latest'));
    if (btnClearHistory) btnClearHistory.addEventListener('click', () => clearLog('history'));
});

function updateTestRow(selectElem, runIndex) {
    const testIndex = selectElem.value;
    const test = allRuns[runIndex].tests[testIndex];
    document.getElementById(`test-acc-${runIndex}`).textContent = `${formatVal(test.accuracy)}%`;
    document.getElementById(`test-time-${runIndex}`).textContent = `${formatVal(test.test_time)}s`;

    const inf_tp = test.inference_throughput;
    document.getElementById(`test-tp-${runIndex}`).textContent = formatVal(inf_tp);

    const lat = test.inference_latency_ms;
    document.getElementById(`test-lat-${runIndex}`).textContent = lat !== 'N/A' ? `${formatVal(lat)}ms` : lat;
}


async function refreshLogs() {
    try {
        const [latestRes, historyRes] = await Promise.all([
            fetch('/api/latest'),
            fetch('/api/history')
        ]);

        const latest = await latestRes.json();
        allRuns = await historyRes.json();

        // Update latest metrics
        if (latest.has_data) {
            document.querySelectorAll('[data-key]').forEach(el => {
                const key = el.getAttribute('data-key');
                let val = latest[key];
                if (val !== undefined) el.textContent = formatVal(val, key);
            });

            document.querySelectorAll('#latest-card .metric').forEach(el => el.style.display = 'none');
            document.querySelectorAll(`#latest-card .${currentViewMode}-metric`).forEach(el => el.style.display = 'flex');

            const tsMetric = document.querySelector('[data-key="timestamp"]');
            if (tsMetric) tsMetric.parentElement.style.display = 'flex';

            document.getElementById('latest-card').style.display = 'block';
            document.getElementById('latest-loader').style.display = 'none';
        }

        const trainBody = document.getElementById('train-body');
        const testBody = document.getElementById('test-body');
        const historyTables = document.getElementById('history-tables');
        const historyLoader = document.getElementById('history-loader');

        const currentTestCount = allRuns.reduce((acc, run) => acc + (run.tests ? run.tests.length : 0), 0);

        const currentCompletedTests = allRuns.reduce((acc, run) => {
            return acc + (run.tests ? run.tests.filter(t => t.accuracy !== undefined && t.accuracy !== 'N/A').length : 0);
        }, 0);

        // Vi uppdaterar bara listan om antalet körningar har ändrats,
        // annars tappar man markeringen (den blå färgen) när man klickat på en rad.
        if (trainBody && testBody && allRuns && (
            trainBody.children.length !== allRuns.length ||
            currentTestCount !== lastTestCount ||
            currentCompletedTests !== lastCompletedTestsCount
        )) {

            if (allRuns.length > 0) {
                if (historyTables) historyTables.style.display = 'grid';
                if (historyLoader) historyLoader.style.display = 'none';
            }

            const testTable = document.getElementById('test-history-table');
            if (testTable) testTable.style.display = (currentViewMode === 'training') ? 'table' : 'none';

            let theadHtml = '';
            let tbodyHtml = '';

            if (currentViewMode === 'training') {
                theadHtml = `
                    <tr><th>Timestamp</th><th>Parallelism</th><th>World Size</th><th>Epochs</th><th>Train Time</th><th>Throughput</th><th>Latency</th></tr>`;
                tbodyHtml = allRuns.map((run) => `
                    <tr>
                        <td>${run.timestamp}</td>
                        <td>${formatVal(run.parallelism_type, 'parallelism_type')}</td>
                        <td>${run.world_size || 'N/A'}</td>
                        <td>${run.epochs}</td>
                        <td>${formatVal(run.training_time)}s</td>
                        <td>${formatVal(run.throughput)}</td>
                        <td>${run.latency_per_batch_ms !== 'N/A' ? formatVal(run.latency_per_batch_ms) + 'ms' : 'N/A'}</td>
                    </tr>`).join('');
            } else if (currentViewMode === 'cpu') {
                theadHtml = `<tr><th>Timestamp</th><th>Parallelism</th><th>Avg CPU Usage</th></tr>`;
                tbodyHtml = allRuns.map((run) => `
                    <tr>
                        <td>${run.timestamp}</td>
                        <td>${formatVal(run.parallelism_type, 'parallelism_type')}</td>
                        <td>${run.overall_avg_cpu_usage !== undefined ? formatVal(run.overall_avg_cpu_usage) + '%' : 'N/A'}</td>
                    </tr>`).join('');
            } else if (currentViewMode === 'mem') {
                theadHtml = `<tr><th>Timestamp</th><th>Parallelism</th><th>Avg Memory %</th><th>Memory Used (B)</th></tr>`;
                tbodyHtml = allRuns.map((run) => `
                    <tr>
                        <td>${run.timestamp}</td>
                        <td>${formatVal(run.parallelism_type, 'parallelism_type')}</td>
                        <td>${run.overall_avg_memory_percent !== undefined ? formatVal(run.overall_avg_memory_percent) + '%' : 'N/A'}</td>
                        <td>${run.overall_avg_memory_used !== undefined ? run.overall_avg_memory_used : 'N/A'}</td>
                    </tr>`).join('');
            } else if (currentViewMode === 'net') {
                theadHtml = `<tr><th>Timestamp</th><th>Parallelism</th><th>Avg Network (B/s)</th><th>Sent (B)</th><th>Recv (B)</th></tr>`;
                tbodyHtml = allRuns.map((run) => `
                    <tr>
                        <td>${run.timestamp}</td>
                        <td>${formatVal(run.parallelism_type, 'parallelism_type')}</td>
                        <td>${run.overall_avg_bytes_per_second !== undefined ? run.overall_avg_bytes_per_second : 'N/A'}</td>
                        <td>${run.total_bytes_sent_during_run !== undefined ? run.total_bytes_sent_during_run : 'N/A'}</td>
                        <td>${run.total_bytes_recv_during_run !== undefined ? run.total_bytes_recv_during_run : 'N/A'}</td>
                    </tr>`).join('');
            }

            document.querySelector('#main-history-table thead').innerHTML = theadHtml;
            trainBody.innerHTML = tbodyHtml;

            if (currentViewMode === 'training') {
                testBody.innerHTML = allRuns.map((run, i) => {
                    if (!run.tests || run.tests.length === 0) {
                        return `<tr><td colspan="5" style="text-align:center;">No tests available</td></tr>`;
                    }
                    const selectOpts = run.tests.map((t, j) => {
                        const timeStr = t.timestamp ? t.timestamp : `Test ${j + 1}`;
                        return `<option value="${j}">${timeStr}</option>`;
                    }).join('');

                    const firstTest = run.tests[0];
                    const lat = firstTest.inference_latency_ms;
                    const inf_tp = firstTest.inference_throughput;

                    return `
                        <tr>
                            <td>
                                <select onchange="updateTestRow(this, ${i})" style="width: 100%; padding: 4px; background: #1a1d24; color: white; border: 1px solid white; border-radius: 4px;">
                                    ${selectOpts}
                                </select>
                            </td>
                            <td id="test-acc-${i}" style="font-weight: bold; color: #28a745;">${formatVal(firstTest.accuracy)}%</td>
                            <td id="test-time-${i}">${formatVal(firstTest.test_time)}s</td>
                            <td id="test-tp-${i}">${formatVal(inf_tp)}</td>
                            <td id="test-lat-${i}">${lat !== 'N/A' ? formatVal(lat) + 'ms' : lat}</td>
                        </tr>`;
                }).join('');
            }

            lastTestCount = currentTestCount;
            lastCompletedTestsCount = currentCompletedTests;
        }

    } catch (err) { }
}

async function clearLog(target) {
    await fetch('/api/logs/clear', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ target })
    });

    if (target === 'latest') {
        document.getElementById('latest-card').style.display = 'none';
        document.getElementById('latest-loader').style.display = 'block';
    } else if (target === 'history') {
        document.getElementById('history-tables').style.display = 'none';
        document.getElementById('history-loader').style.display = 'block';
    }

    refreshLogs();
}

refreshLogs();