// dynamic tiling manager for history charts
(function () {
    function log(msg) {
        console.log(msg);
        let d = document.getElementById('debug');
        if (!d) {
            d = document.createElement('pre');
            d.id = 'debug';
            d.style.color = 'white';
            d.style.background = 'rgba(0,0,0,0.75)';
            d.style.padding = '4px';
            d.style.position = 'fixed';
            d.style.top = '0';
            d.style.left = '0';
            d.style.zIndex = '9999';
            d.style.maxHeight = '200px';
            d.style.overflow = 'auto';
            document.body.appendChild(d);
        }
        d.textContent += msg + "\n";
    }
   // log('Graphs.js loaded');
    const labelMap = {
        'accuracy': 'Accuracy (%)',
        'throughput': 'Throughput',
        'training_time': 'Training Time (s)',
        'test_time': 'Test Time (s)',
        'latency_per_batch': 'Latency per batch (ms)',
        'epochs': 'Epochs'
    };

    // handlers map metric key to extractor from history entry
    const metricHandlers = {
        accuracy: h => (h.tests && h.tests[0] ? h.tests[0].accuracy : null),
        test_time: h => (h.tests && h.tests[0] ? h.tests[0].test_time : null),
        latency_per_batch: h => h.latency_per_batch_ms || (h.tests && h.tests[0] ? h.tests[0].inference_latency_ms : null),
        throughput: h => h.throughput,
        training_time: h => h.training_time,
        epochs: h => h.epochs
    };

    function formatTimestamp(ts) {
        try {
            const d = new Date(ts);
            if (!isNaN(d.getTime())) return d.toLocaleString();
        } catch (e) {}
        return ts;
    }

    function buildChart(ctx, labels, data, displayLabel, isPercent, chartType = 'bar') {
        return new Chart(ctx, {
            type: chartType,
            data: {
                labels,
                datasets: [{
                    label: displayLabel,
                    data,
                    borderColor: 'rgb(32, 215, 215)',
                    backgroundColor: 'rgba(25, 158, 158, 0.91)',
                    borderWidth: 1,
                    maxBarThickness: 48,
                    tension: 0.2,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: 'index', intersect: false },
                layout: { padding: 8 },
                scales: {
                    y: { beginAtZero: true },
                    x: { ticks: { autoSkip: false }, title: { display: false } }
                },
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: function(ctx) {
                                let v = ctx.formattedValue;
                                return isPercent ? `${v} %` : v;
                            }
                        }
                    }
                }
            }
        });
    }

    let historyData = [];
    const tiles = [];

    function updateAllCharts() {
        tiles.forEach(tile => {
            if (!tile.canvas || !tile.canvas.parentElement) return;
            
            const metric = tile.metricSelect.value;
            const chartType = tile.typeSelect ? tile.typeSelect.value : 'bar';
            const labels = historyData.map((h, i) => {
                if (h.parallelism_type) return h.parallelism_type;
                if (h.timestamp) return formatTimestamp(h.timestamp);
                return `Run ${i+1}`;
            });
            const handler = metricHandlers[metric] || (h => parseFloat(h[metric]));
            const data = historyData.map(h => {
                const raw = handler(h);
                const v = parseFloat(raw);
                return Number.isFinite(v) ? v : null;
            });
            const isPercent = metric === 'accuracy';
            const displayLabel = labelMap[metric] || metric;

            const rect = tile.canvas.parentElement.getBoundingClientRect();
            const width = rect.width || 300;
            const height = rect.height || 300;
            
            if (width === 0 || height === 0) {
                return;
            }

            if (tile.chart) {
                tile.chart.config.type = chartType;
                tile.chart.data.labels = labels;
                tile.chart.data.datasets[0].data = data;
                tile.chart.data.datasets[0].label = displayLabel;
                tile.chart.options.plugins.tooltip.callbacks.label = function(context) {
                    let v = context.formattedValue;
                    return isPercent ? `${v} %` : v;
                };
                tile.chart.update();
            } else {
                try {
                    const ctx = tile.canvas.getContext('2d');
                    const scale = window.devicePixelRatio || 1;
                    ctx.canvas.width = width * scale;
                    ctx.canvas.height = height * scale;
                    ctx.scale(scale, scale);
                    tile.chart = buildChart(ctx, labels, data, displayLabel, isPercent, chartType);
                } catch (err) {
                    console.error('Failed to create chart:', err);
                }
            }
        });
    }

    async function fetchHistory() {
        try {
            const res = await fetch('/api/history');
            const json = await res.json();
            historyData = json || [];
            updateAllCharts();
            document.querySelectorAll('.tile-loader').forEach(l => l.style.display = historyData.length ? 'none' : 'block');
        } catch (err) {
            console.error('Failed to fetch history:', err);
        }
    }

    function makeTileElement() {
        const tile = document.createElement('div');
        tile.className = 'tile';
        tile.setAttribute('draggable', 'true');

        const header = document.createElement('div');
        header.className = 'tile-header';

        const select = document.createElement('select');
        Object.keys(labelMap).forEach(key => {
            const opt = document.createElement('option');
            opt.value = key;
            opt.textContent = labelMap[key];
            select.appendChild(opt);
        });
        select.value = 'accuracy';
        header.appendChild(select);

        const typeSelect = document.createElement('select');
        ['bar','line'].forEach(t => {
            const opt = document.createElement('option');
            opt.value = t;
            opt.textContent = t.charAt(0).toUpperCase() + t.slice(1);
            typeSelect.appendChild(opt);
        });
        typeSelect.value = 'bar';
        typeSelect.style.marginLeft = '6px';
        header.appendChild(typeSelect);

        const removeBtn = document.createElement('button');
        removeBtn.type = 'button';
        removeBtn.className = 'remove-btn';
        removeBtn.textContent = '×';
        header.appendChild(removeBtn);

        const loader = document.createElement('div');
        loader.className = 'waiting-card tile-loader';
        loader.innerHTML = `<div class="spinner"></div><p>Waiting for history data...</p>`;

        const canvasWrap = document.createElement('div');
        canvasWrap.className = 'tile-canvas';
        const c = document.createElement('canvas');
        canvasWrap.appendChild(c);

        tile.appendChild(header);
        tile.appendChild(loader);
        tile.appendChild(canvasWrap);

        return { tile, select, removeBtn, canvas: c, loader, typeSelect };
    }

    function addTile() {
        //log('addTile called');
        const container = document.getElementById('graph-tiles');
        if (!container) {
            console.error('graph-tiles container not found');
            return;
        }
        const { tile, select, removeBtn, canvas, loader, typeSelect } = makeTileElement();
        container.appendChild(tile);
        const tileObj = { tile, metricSelect: select, typeSelect, canvas, loader, chart: null };
        tiles.push(tileObj);

        select.addEventListener('change', updateAllCharts);
        typeSelect.addEventListener('change', updateAllCharts);
        removeBtn.addEventListener('click', () => {
            container.removeChild(tile);
            const idx = tiles.indexOf(tileObj);
            if (idx !== -1) tiles.splice(idx, 1);
        });

        enableDrag(tile);
        
        requestAnimationFrame(() => {
            updateAllCharts();
        });
    }

    let dragSrc = null;
    function enableDrag(el) {
        el.addEventListener('dragstart', e => {
            dragSrc = el;
            e.dataTransfer.effectAllowed = 'move';
        });
        el.addEventListener('dragover', e => {
            e.preventDefault();
            e.dataTransfer.dropEffect = 'move';
        });
        el.addEventListener('drop', e => {
            e.stopPropagation();
            if (dragSrc && dragSrc !== el) {
                const parent = el.parentNode;
                parent.insertBefore(dragSrc, el.nextSibling);
            }
        });
    }

    document.addEventListener('DOMContentLoaded', () => {
        //log('DOMContentLoaded fired');
        //alert('Graphs.js DOMContentLoaded');
        const addBtn = document.getElementById('add-tile');
        if (addBtn) {
           // log('Add button found');
            addBtn.addEventListener('click', () => {
                //log('Add button clicked');
                //alert('Add button clicked');
                addTile();
            });
        } else {
            //log('No add button present');
            //alert('Add button missing');
        }
        addTile();
        fetchHistory();
        setInterval(fetchHistory, 2000);
    });
})();
