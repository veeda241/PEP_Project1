// ML Analytics Dashboard - JavaScript

const API_BASE = 'http://localhost:5000/api';

// Fetch data from API
async function fetchData(endpoint) {
    try {
        const response = await fetch(`${API_BASE}/${endpoint}`);
        if (!response.ok) throw new Error('Not found');
        return await response.json();
    } catch (error) {
        console.error(`Error fetching ${endpoint}:`, error);
        return null;
    }
}

// Format numbers
function formatNumber(num, decimals = 4) {
    return Number(num).toFixed(decimals);
}

function formatCurrency(num) {
    return '$' + Number(num).toLocaleString('en-US', { maximumFractionDigits: 0 });
}

// Load Regression Results
async function loadRegressionResults() {
    const data = await fetchData('regression');
    if (!data || data.error) {
        document.getElementById('regression-results').innerHTML =
            '<div class="loading-card"><p>No results found. Please run the ML pipeline first.</p></div>';
        return;
    }

    // Update best model - handle both possible structures
    const bestModel = data.best_model;
    const bestR2 = bestModel.test_r2 || bestModel.metrics?.r2 || 0;
    document.getElementById('regression-best').innerHTML = `
        <span class="model-name">${bestModel.name}</span>
        <span class="model-score">R² = ${formatNumber(bestR2)}</span>
    `;

    // Create model cards
    const resultsGrid = document.getElementById('regression-results');
    let html = '';

    for (const [name, metrics] of Object.entries(data.models)) {
        // Handle different key names: test_r2 or r2
        const r2Value = metrics.test_r2 !== undefined ? metrics.test_r2 : metrics.r2;
        html += `
            <div class="result-card">
                <div class="model-name">${name}</div>
                <div class="metrics">
                    <div class="metric">
                        <span class="metric-label">R²</span>
                        <span class="metric-value">${formatNumber(r2Value)}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">RMSE</span>
                        <span class="metric-value">${formatCurrency(metrics.rmse)}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">MAE</span>
                        <span class="metric-value">${formatCurrency(metrics.mae)}</span>
                    </div>
                </div>
            </div>
        `;
    }
    resultsGrid.innerHTML = html;

    // Update hero stats
    const modelCount = Object.keys(data.models).length;
    const modelsElement = document.getElementById('regression-models');
    if (modelsElement) modelsElement.textContent = `${modelCount} Models`;

    // Load images
    loadImage('reg-comparison', 'regression', 'model_comparison.png');
    loadImage('reg-actual-pred', 'regression', 'actual_vs_predicted.png');
    loadImage('reg-importance', 'regression', 'feature_importance.png');
    loadImage('reg-residuals', 'regression', 'residuals_distribution.png');
}

// Load Classification Results
async function loadClassificationResults() {
    const data = await fetchData('classification');
    if (!data || data.error) {
        document.getElementById('classification-results').innerHTML =
            '<div class="loading-card"><p>No results found. Please run the ML pipeline first.</p></div>';
        return;
    }

    // Update best model - handle both string name and object structures
    const bestModelName = typeof data.best_model === 'string' ? data.best_model : data.best_model.name;
    const bestMetrics = data.models[bestModelName] || data.best_model;
    document.getElementById('classification-best').innerHTML = `
        <span class="model-name">${bestModelName}</span>
        <span class="model-score">F1 = ${formatNumber(bestMetrics.f1_score)}</span>
    `;

    // Create model cards
    const resultsGrid = document.getElementById('classification-results');
    let html = '';

    for (const [name, metrics] of Object.entries(data.models)) {
        html += `
            <div class="result-card">
                <div class="model-name">${name}</div>
                <div class="metrics">
                    <div class="metric">
                        <span class="metric-label">F1</span>
                        <span class="metric-value">${formatNumber(metrics.f1_score)}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Accuracy</span>
                        <span class="metric-value">${formatNumber(metrics.accuracy)}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">ROC-AUC</span>
                        <span class="metric-value">${formatNumber(metrics.roc_auc)}</span>
                    </div>
                </div>
            </div>
        `;
    }
    resultsGrid.innerHTML = html;

    // Update hero stats
    const modelCount = Object.keys(data.models).length;
    const modelsElement = document.getElementById('classification-models');
    if (modelsElement) modelsElement.textContent = `${modelCount} Models`;

    // Load images
    loadImage('cls-comparison', 'classification', 'model_comparison.png');
    loadImage('cls-roc', 'classification', 'roc_curves.png');
    loadImage('cls-confusion', 'classification', 'confusion_matrix.png');
}

// Load Time Series Results
async function loadTimeseriesResults() {
    const data = await fetchData('timeseries');
    if (!data || data.error) {
        document.getElementById('timeseries-results').innerHTML =
            '<div class="loading-card"><p>No results found. Please run the ML pipeline first.</p></div>';
        return;
    }

    // Update best model - handle both string name and object structures
    const bestModelName = typeof data.best_model === 'string' ? data.best_model : data.best_model.name;
    const bestMetrics = data.models[bestModelName] || data.best_model;
    document.getElementById('timeseries-best').innerHTML = `
        <span class="model-name">${bestModelName}</span>
        <span class="model-score">RMSE = ${formatNumber(bestMetrics.rmse, 2)}</span>
    `;

    // Create model cards
    const resultsGrid = document.getElementById('timeseries-results');
    let html = '';

    for (const [name, metrics] of Object.entries(data.models)) {
        html += `
            <div class="result-card">
                <div class="model-name">${name}</div>
                <div class="metrics">
                    <div class="metric">
                        <span class="metric-label">RMSE</span>
                        <span class="metric-value">${formatNumber(metrics.rmse, 2)}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">MAE</span>
                        <span class="metric-value">${formatNumber(metrics.mae, 2)}</span>
                    </div>
                </div>
            </div>
        `;
    }
    resultsGrid.innerHTML = html;

    // Update hero stats
    const modelCount = Object.keys(data.models).length;
    const modelsElement = document.getElementById('timeseries-models');
    if (modelsElement) modelsElement.textContent = `${modelCount} Models`;

    // Load images
    loadImage('ts-forecast', 'timeseries', 'forecast_comparison.png');
    loadImage('ts-decomposition', 'timeseries', 'decomposition.png');
    loadImage('ts-comparison', 'timeseries', 'model_comparison.png');
}

// Load image with error handling
function loadImage(elementId, category, filename) {
    const img = document.getElementById(elementId);
    if (img) {
        img.src = `${API_BASE}/images/${category}/${filename}`;
        img.onerror = () => {
            img.style.display = 'none';
        };
    }
}

// Smooth scroll for navigation
document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        const target = document.querySelector(link.getAttribute('href'));
        if (target) {
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }

        // Update active state
        document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
        link.classList.add('active');
    });
});

// Update active nav on scroll
window.addEventListener('scroll', () => {
    const sections = document.querySelectorAll('.section');
    let current = '';

    sections.forEach(section => {
        const sectionTop = section.offsetTop - 200;
        if (window.scrollY >= sectionTop) {
            current = section.getAttribute('id');
        }
    });

    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === `#${current}`) {
            link.classList.add('active');
        }
    });
});

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadRegressionResults();
    loadClassificationResults();
    loadTimeseriesResults();
});
