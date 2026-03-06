const API_BASE = '/api/training';

const btnStart = document.getElementById('btn-start');
const btnStop = document.getElementById('btn-stop');
const testModel = document.getElementById('test-model');
const testType = document.getElementById('test-type');
const statusText = document.getElementById('test-status');
const trainingForm = document.getElementById('config-form');

const configInputs = {
    model: document.getElementById('model-input'),
    parallelism: document.getElementById('parallelism-input'),
    saved: document.getElementById('saved-input')
};

let wasActive = null;

function updateUI(isActive, details = null) {
    if (btnStart) btnStart.disabled = isActive;
    if (btnStop) btnStop.disabled = !isActive;

    if (isActive && details) {
        if (statusText) {
            statusText.innerText = `Running (PID: ${details.pid})`;
            statusText.style.color = 'green';
        }
        if (testModel) testModel.innerText = details.model || 'N/A';
        if (testType) testType.innerText = details.parallelism || 'N/A';

    } else {
        if (statusText) {
            statusText.innerText = "Idle";
            statusText.style.color = 'gray';
        }

        if (testModel) testModel.innerText = 'N/A';
        if (testType) testType.innerText = 'N/A';
    }
}

async function startTraining() {
    if (!trainingForm.reportValidity()) {
        return;
    }

    const payload = {
        model: configInputs.model.value,
        parallelism: configInputs.parallelism.value,
        saved: configInputs.saved.value
    };

    try {
        const response = await fetch(`${API_BASE}/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await response.json();

        if (response.ok) {
            document.getElementById('latest-card').style.display = 'none';
            document.getElementById('latest-loader').style.display = 'block';
            checkStatus();
        } else {
            alert(`Error: ${data.message}`);
        }
    } catch (err) {
        console.error("Failed to start:", err);
    }
}

async function stopTraining() {
    try {
        const response = await fetch(`${API_BASE}/stop`, { method: 'POST' });
        if (response.ok) {
            checkStatus();
        }
    } catch (err) {
        console.error("Failed to stop:", err);
    }
}

async function checkStatus() {
    try {
        const response = await fetch(`${API_BASE}/status`);
        const data = await response.json();
        updateUI(data.active, data.details);

        if (wasActive === true && data.active === false) {
            refreshLogs();
            refreshChart();
        }
        wasActive = data.active;
    } catch (err) {
        console.error("Status check failed:", err);
    }
}

btnStart.addEventListener('click', startTraining);
btnStop.addEventListener('click', stopTraining);

checkStatus();
setInterval(checkStatus, 5000);