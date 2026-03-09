// Fetch history and render an accuracy-over-runs chart using Chart.js
(function () {
    let chart = null;

    function formatTimestamp(ts) {
        try {
            const d = new Date(ts);
            if (!isNaN(d.getTime())) return d.toLocaleString();
        } catch (e) {}
        return ts;
    }

    function buildChart(ctx, labels, data, displayLabel, isPercent) {
        return new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: displayLabel,
                    data: data,
                    borderColor: 'rgb(32, 215, 215)',
                    backgroundColor: 'rgba(25, 158, 158, 0.91)',
                    borderWidth: 1,
                    maxBarThickness: 48,
                    //tension: 0.2,
                    //fill: true,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: 'index', intersect: false },
                layout: { padding: 8 },
                scales: {
                    y: { beginAtZero: true },
                    x: {
                        ticks: { autoSkip: false },
                        title: { display: false }
                    }
                },
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                let v = context.formattedValue;
                                return isPercent ? `${v} %` : v;
                            }
                        }
                    }
                }
            }
        });
    }

    async function refreshChart() {
        try {
            const res = await fetch('/api/history');
            const history = await res.json();

            const canvas = document.getElementById('history-chart');
            const loader = document.getElementById('graph-loader');

            if (!history || history.length === 0) {
                if (canvas) canvas.style.display = 'none';
                if (loader) loader.style.display = 'block';
                return;
            }

            const metricSelect = document.getElementById('metric-select');
            const selectedMetric = metricSelect ? metricSelect.value : 'accuracy';

            // Prefer showing the parallelism type on the x-axis; fall back to timestamp/run number
            const labels = history.map((h, i) => {
                let name = h.parallelism_type ? h.parallelism_type.replace(/_/g, ' ') : `Run ${i+1}`;
                let time = h.timestamp ? h.timestamp.split(' ')[1] : '';
                return `${name}\n${time}`;
            });
            const data = history.map(h => {
                let val = null;

                if (h[selectedMetric] !== undefined) {
                    val = h[selectedMetric];
                }
                else if (h.tests && h.tests.length > 0) {
                    if (h.tests[0][selectedMetric] !== undefined) {
                        val = h.tests[0][selectedMetric];
                    }
                }

                const v = parseFloat(val);
                return Number.isFinite(v) ? v : null;
            });

            if (loader) loader.style.display = 'none';
            if (canvas) {
                canvas.style.display = 'block';
                // give the canvas a fixed visual height for better resolution
                canvas.style.width = '100%';
                canvas.style.height = '360px';
            }

            const ctx = canvas.getContext('2d');
            // ensure high-DPI rendering matches display size
            const scale = window.devicePixelRatio || 1;
            ctx.canvas.width = ctx.canvas.clientWidth * scale;
            ctx.canvas.height = ctx.canvas.clientHeight * scale;
            ctx.scale(scale, scale);

            const labelMap = {
                'accuracy': 'Accuracy (%)',
                'throughput': 'Throughput',
                'training_time': 'Training time (s)',
                'test_time': 'Test time (s)',
                'latency_per_batch_ms': 'Latency per batch (ms)',
                'epochs': 'Epochs'
            };
            const isPercent = selectedMetric === 'accuracy';
            const displayLabel = labelMap[selectedMetric] || selectedMetric;

            if (chart) {
                chart.data.labels = labels;
                chart.data.datasets[0].data = data;
                chart.data.datasets[0].label = displayLabel;
                chart.options.plugins.tooltip.callbacks.label = function(context) {
                    let v = context.formattedValue;
                    return isPercent ? `${v} %` : v;
                };
                chart.update();
            } else {
                chart = buildChart(ctx, labels, data, displayLabel, isPercent);
            }

            // attach change handler once
            if (metricSelect && !metricSelect._listenerAttached) {
                metricSelect.addEventListener('change', () => refreshChart());
                metricSelect._listenerAttached = true;
            }
        } catch (err) {
            console.error('Failed to refresh chart:', err);
        }
    }

    // Initial render and periodic refresh
    document.addEventListener('DOMContentLoaded', () => {
        refreshChart();
        setInterval(refreshChart, 2000);
    });
})();