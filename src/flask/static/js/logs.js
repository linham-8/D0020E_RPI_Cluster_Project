let allRuns = [];
let lastTestCount = -1;

function formatVal(val, key) {
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
    const modelType = document.getElementById('parallelism').value;
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
    const pSelect = document.getElementById('parallelism');
    const mSelect = document.getElementById('model');
    if (pSelect) pSelect.addEventListener('change', updateSavedOptions);
    if (mSelect) mSelect.addEventListener('change', updateSavedOptions);
});

function updateTestRow(selectElem, runIndex) {
    const testIndex = selectElem.value;
    const test = allRuns[runIndex].tests[testIndex];
    document.getElementById(`test-acc-${runIndex}`).textContent = `${formatVal(test.accuracy)}%`;
    document.getElementById(`test-time-${runIndex}`).textContent = `${formatVal(test.test_time)}s`;
    document.getElementById(`test-lat-${runIndex}`).textContent = `${formatVal(test.inference_latency_ms || test.latency_per_batch)}ms`;
}

function showTestsForRun(runIndex) {
    const run = allRuns[runIndex];
    const tbody = document.getElementById('test-body');
    tbody.innerHTML = '';

    const rows = document.querySelectorAll('#train-body tr');
    rows.forEach((r, idx) => {
        if (idx === runIndex) r.classList.add('selected-row');
        else r.classList.remove('selected-row');
    });

    if (!run.tests || run.tests.length === 0) {
        tbody.innerHTML = '<tr><td colspan="4">No tests found for this run.</td></tr>';
        return;
    }

    run.tests.forEach(test => {
        const timePart = test.timestamp.split(' ')[1];
        tbody.innerHTML += `
            <tr>
                <td>${timePart}</td>
                <td style="font-weight: bold; color: #28a745;">${formatVal(test.accuracy)}%</td>
                <td>${formatVal(test.test_time)}</td>
                <td>${formatVal(test.inference_latency_ms)}</td>
            </tr>`;
    });
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

                if (key === 'time_val') val = latest.training_time !== "N/A" ? latest.training_time : latest.test_time;
                if (key === 'throughput') val = latest.throughput !== "N/A" ? latest.throughput : latest.inference_throughput;

                if (latest[key] !== undefined) {
                    el.textContent = formatVal(latest[key], key);
                }
            });

            document.getElementById('latest-card').style.display = 'block';
            document.getElementById('latest-loader').style.display = 'none';
        }

        const trainBody = document.getElementById('train-body');
        const testBody = document.getElementById('test-body');
        const historyTables = document.getElementById('history-tables');
        const historyLoader = document.getElementById('history-loader');

        const currentTestCount = allRuns.reduce((acc, run) => acc + (run.tests ? run.tests.length : 0), 0);

        // Vi uppdaterar bara listan om antalet körningar har ändrats, 
        // annars tappar man markeringen (den blå färgen) när man klickat på en rad.
        if (trainBody && testBody && allRuns && (trainBody.children.length !== allRuns.length || currentTestCount !== lastTestCount)) {

            if (allRuns.length > 0) {
                if (historyTables) historyTables.style.display = 'grid';
                if (historyLoader) historyLoader.style.display = 'none';
            }

            trainBody.innerHTML = allRuns.map((run) => `
                <tr>
                    <td>${run.timestamp}</td>
                    <td>${formatVal(run.parallelism_type, 'parallelism_type')}</td>
                    <td>${run.world_size || 'N/A'}</td>
                    <td>${run.epochs}</td>
                    <td>${formatVal(run.training_time)}s</td> <td>${formatVal(run.throughput)}</td>
                </tr>`).join('');
            
            testBody.innerHTML = allRuns.map((run, i) => {
                if (!run.tests || run.tests.length === 0) {
                    return `<tr><td colspan="4">No tests available</td></tr>`;
                }
                const selectOpts = run.tests.map((t, j) => `<option value="${j}">${t.timestamp.split(' ')[1] || t.timestamp}</option>`).join('');
                const firstTest = run.tests[0];
                return `
                    <tr>
                        <td><select onchange="updateTestRow(this, ${i})">${selectOpts}</select></td>
                        <td id="test-acc-${i}" style="font-weight: bold; color: #28a745;">${formatVal(firstTest.accuracy)}%</td>
                        <td id="test-time-${i}">${formatVal(firstTest.test_time)}s</td> <td id="test-lat-${i}">${formatVal(firstTest.inference_latency_ms || firstTest.latency_per_batch)}ms</td> </tr>`;
            }).join('');

            lastTestCount = currentTestCount;
        }

    } catch (err) { }
}

refreshLogs();
setInterval(refreshLogs, 5000);