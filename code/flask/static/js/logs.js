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

async function refreshLogs() {
    try {
        const [latestRes, historyRes] = await Promise.all([
            fetch('/api/latest'),
            fetch('/api/history')
        ]);

        const latest = await latestRes.json();
        const history = await historyRes.json();

        // Update latest metrics
        if (latest.has_data) {
            document.querySelectorAll('[data-key]').forEach(el => {
                const key = el.getAttribute('data-key');
                if (latest[key] !== undefined) {
                    el.textContent = formatVal(latest[key], key);
                }
            });

            document.getElementById('latest-card').style.display = 'block';
            document.getElementById('latest-loader').style.display = 'none';
        }

        // Update History Table
        if (history && history.length > 0) {
            document.getElementById('history-body').innerHTML = history.map(row => `
                <tr>
                    <td>${row.model || 'SVM'}</td>
                    <td>${formatVal(row.parallelism_type, 'parallelism_type')}</td>
                    <td>${formatVal(row.accuracy)} %</td>
                    <td>${formatVal(row.training_time)} s</td>
                    <td>${formatVal(row.test_time)} s</td>
                    <td>${formatVal(row.throughput)}</td>
                    <td>${formatVal(row.latency_per_batch)} ms</td>
                    <td>${row.world_size}</td>
                    <td>${row.epochs}</td>
                    <td>${row.timestamp}</td>
                </tr>`).join('');

            document.getElementById('history-table').style.display = 'table';
            document.getElementById('history-loader').style.display = 'none';
        }
    } catch (err) {
        console.error("Dashboard update failed:", err);
    }
}

refreshLogs();
setInterval(refreshLogs, 5000);