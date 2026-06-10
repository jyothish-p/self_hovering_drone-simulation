/**
 * models/weights_export.js
 * ─────────────────────────────────────────────────────────────────────────────
 * Utility to export DQN agent weights from browser IndexedDB to JSON files
 * that can be saved alongside your project.
 *
 * USAGE — open browser console while rl-training.html is running:
 *
 *   // Export weights (downloads as JSON):
 *   await AGENT.saveWeights();
 *
 *   // Load previously saved weights:
 *   await AGENT.loadWeights();
 *
 * TensorFlow.js stores weights in browser IndexedDB under the key:
 *   'tensorflowjs_models/dqn-hover-agent/...'
 *
 * For persistent storage outside the browser, use the export script below.
 * ─────────────────────────────────────────────────────────────────────────────
 */

/**
 * Export weights to a downloadable JSON file.
 * Paste this in the browser console while training is running.
 */
async function exportWeightsToFile() {
    const model = window.AGENT._online;
    const saveResult = await model.save('downloads://rlt-dqn-weights');
    console.log('✅ Weights exported:', saveResult);
}

/**
 * Import weights from a JSON file.
 * Paste this in the browser console and select your weights file.
 */
async function importWeightsFromFile() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json,.bin';
    input.multiple = true;
    input.onchange = async (e) => {
        const files = Array.from(e.target.files);
        const jsonFile = files.find(f => f.name.endsWith('.json'));
        const binFiles = files.filter(f => f.name.endsWith('.bin'));
        if (!jsonFile) { alert('Please select the .json weights file'); return; }
        const model = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, ...binFiles]));
        // Copy weights to agent's online network
        window.AGENT._online.setWeights(model.getWeights());
        window.AGENT._target.setWeights(model.getWeights());
        console.log('✅ Weights imported successfully');
    };
    input.click();
}

// ── Training checkpoint info ──────────────────────────────────────────────────
// Run this in console to log current training state:
function checkpointInfo() {
    const a = window.AGENT;
    const e = window.RL_ENV;
    if (!a || !e) { console.error('Training not active'); return; }
    console.table({
        'Episode': e.episode,
        'Total Steps': a.stats.totalSteps,
        'Training Steps': a.stats.trainingSteps,
        'Epsilon': a.stats.epsilon.toFixed(4),
        'Buffer Size': a.stats.bufferSize,
        'Last Loss': a.stats.lastLoss?.toFixed(6) ?? 'N/A',
        'Avg Reward (50)': e.avgReward(50).toFixed(2),
        'Best Episode Reward': Math.max(...e.episodeHistory).toFixed(2),
    });
}

console.log('%c📦 weights_export.js loaded — use exportWeightsToFile(), importWeightsFromFile(), checkpointInfo()', 'color:#38bdf8');
