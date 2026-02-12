let allRuns = [];

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
    
    savedSelect.innerHTML = '<option value="no">New Training Run (Default)</option>';
    
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
                opt.textContent = `${run.timestamp} (Ep: ${run.epochs})`;
                group.appendChild(opt);
            });
            savedSelect.appendChild(group);
        }
    } catch (err) {
        console.error("Error fetching archives:", err);
    }
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
        
        // Vi uppdaterar bara listan om antalet körningar har ändrats, 
        // annars tappar man markeringen (den blå färgen) när man klickat på en rad.
        if (trainBody && allRuns && trainBody.children.length !== allRuns.length) {
            trainBody.innerHTML = allRuns.map((run, index) => `
                <tr onclick="showTestsForRun(${index})" style="cursor: pointer;">
                    <td>${run.timestamp}</td>
                    <td>${formatVal(run.parallelism_type, 'parallelism_type')}</td>
                    <td>${run.epochs}</td>
                    <td>${formatVal(run.training_time)}</td>
                    <td>${formatVal(run.throughput)}</td>
                </tr>`).join('');
        }

    } catch (err) {
        console.error("Dashboard update failed:", err);
    }
}

refreshLogs();
setInterval(refreshLogs, 5000);