/**
 * rl-main.js — RL Training Bootstrap
 * ─────────────────────────────────────────────────────────────────────────────
 * Orchestrates: Cannon.js physics world + DroneModel + DroneScene + DQNAgent
 *               + RLEnvironment → continuous real-time learning loop.
 *
 * Loop per animation frame:
 *   1. Agent selects action from current state (ε-greedy)
 *   2. Action → drone motor thrust adjustment
 *   3. Physics world.step (60 Hz)
 *   4. Extract next state + reward + done
 *   5. Store (s,a,r,s',done) in replay buffer
 *   6. Async train() — no render-frame blocking
 *   7. If done → auto-reset episode
 *   8. Update all HUD elements
 *
 * Globals: window.AGENT, window.DRONE, window.RL_ENV
 * ─────────────────────────────────────────────────────────────────────────────
 */

(function () {
    // ── Physics World ────────────────────────────────────────────────────────
    const world = new CANNON.World();
    world.gravity.set(0, -9.81, 0);
    world.broadphase = new CANNON.SAPBroadphase(world);
    world.solver.iterations = 20;

    const groundBody = new CANNON.Body({ mass: 0 });
    groundBody.addShape(new CANNON.Plane());
    groundBody.quaternion.setFromAxisAngle(new CANNON.Vec3(1, 0, 0), -Math.PI / 2);
    world.addBody(groundBody);

    // ── Core systems ─────────────────────────────────────────────────────────
    const drone = new DroneModel(world, { x: 0, y: 1.5, z: 0 });
    const scene = new DroneScene(drone);
    const agent = new DQNAgent();
    const rlEnv = new RLEnvironment(drone, world);

    window.AGENT = agent;
    window.DRONE = drone;
    window.RL_ENV = rlEnv;
    window.SCENE = scene;

    // ── Training state ───────────────────────────────────────────────────────
    let currentState = null;   // declared here so manualReset() + initTakeoffStage() can assign to it
    let isTrainingOn = true;
    let frameCount = 0;
    let fpsFrames = 0;
    let fpsLast = performance.now();
    let currentFPS = 60;
    let stepCount = 0;
    let lastLoss = null;
    const lossHistory = [];     // for sparkline
    const rewardHistory = [];     // per-step rewards, cleared each episode

    // Keyboard
    const keys = {};
    window.addEventListener('keydown', e => {
        keys[e.code] = true;
        if (e.code === 'KeyT') { isTrainingOn = !isTrainingOn; updatePauseBtn(); }
        if (e.code === 'KeyC') {
            const modes = ['follow', 'orbit', 'top'];
            const i = modes.indexOf(scene.cameraMode);
            scene.setCameraMode(modes[(i + 1) % 3]);
            document.getElementById('cam-mode').textContent = modes[(i + 1) % 3].toUpperCase();
        }
        if (e.code === 'KeyR') { manualReset(); }
        if (e.code === 'KeyS') { agent.saveWeights(); showToast('Weights saved!'); }
        if (e.code === 'KeyL') { agent.loadWeights().then(ok => showToast(ok ? 'Weights loaded!' : 'No saved weights')); }
    });
    window.addEventListener('keyup', e => { keys[e.code] = false; });

    function manualReset() {
        currentState = rlEnv.reset();
        rewardHistory.length = 0;
        drawRewardChart();
    }

    function updatePauseBtn() {
        const btn = document.getElementById('btn-pause');
        if (btn) btn.textContent = isTrainingOn ? '⏸ Pause' : '▶ Resume';
    }

    // ── HUD refs ────────────────────────────────────────────────────────────
    const $ = id => document.getElementById(id);
    const hud = {
        episode: $('rl-episode'),
        steps: $('rl-steps'),
        epsilon: $('rl-epsilon'),
        epsBar: $('rl-eps-bar'),
        loss: $('rl-loss'),
        reward: $('rl-reward'),
        avgReward: $('rl-avg-reward'),
        bufSize: $('rl-buf-size'),
        bufBar: $('rl-buf-bar'),
        trainSteps: $('rl-train-steps'),
        warmup: $('rl-warmup'),
        action: $('rl-action'),
        targetSync: $('rl-target-sync'),
        fps: $('rl-fps'),
        totalT: $('rl-total-t'),
        alt: $('rl-alt'),
        targetAlt: $('rl-target-alt'),
        speed: $('rl-speed'),
        posY: $('rl-pos-y'),
        velY: $('rl-vel-y'),
        camMode: $('cam-mode'),
        // Q-value bars (17 actions)
        qBars: Array.from({ length: 17 }, (_, i) => $('q-bar-' + i)),
        qVals: Array.from({ length: 17 }, (_, i) => $('q-val-' + i)),
        // Motor bars (4 motors)
        mBars: Array.from({ length: 4 }, (_, i) => $('rl-m' + (i + 1) + '-bar')),
        mVals: Array.from({ length: 4 }, (_, i) => $('rl-m' + (i + 1) + '-val')),
        // Motor symmetry balance bars
        frBar: $('rl-fr-bar'),
        frVal: $('rl-fr-val'),
        lrBar: $('rl-lr-bar'),
        lrVal: $('rl-lr-val'),
    };

    const ACTION_LABELS = [
        'ALL ↑', 'ALL ↓',
        'Pitch+', 'Pitch-',
        'Roll+', 'Roll-',
        'Yaw CW', 'Yaw CCW',
        'M1 ↑', 'M1 ↓',
        'M2 ↑', 'M2 ↓',
        'M3 ↑', 'M3 ↓',
        'M4 ↑', 'M4 ↓',
        'Hold',
    ];

    // ── Reward sparkline canvas ──────────────────────────────────────────────
    function drawRewardChart() {
        const canvas = $('reward-chart');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const w = canvas.width, h = canvas.height;
        ctx.clearRect(0, 0, w, h);

        const hist = rlEnv.episodeHistory;
        if (hist.length < 2) return;

        const data = hist.slice(-80);
        const minR = Math.min(...data);
        const maxR = Math.max(...data);
        const range = Math.max(maxR - minR, 1);

        ctx.beginPath();
        ctx.strokeStyle = '#38bdf8';
        ctx.lineWidth = 1.5;
        data.forEach((r, i) => {
            const x = (i / (data.length - 1)) * w;
            const y = h - ((r - minR) / range) * h;
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        });
        ctx.stroke();

        // Zero line
        if (minR < 0 && maxR > 0) {
            const zeroY = h - ((0 - minR) / range) * h;
            ctx.beginPath();
            ctx.strokeStyle = 'rgba(255,255,255,0.15)';
            ctx.lineWidth = 1;
            ctx.setLineDash([3, 4]);
            ctx.moveTo(0, zeroY); ctx.lineTo(w, zeroY);
            ctx.stroke();
            ctx.setLineDash([]);
        }
    }

    // Loss sparkline
    function drawLossChart() {
        const canvas = $('loss-chart');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const w = canvas.width, h = canvas.height;
        ctx.clearRect(0, 0, w, h);
        if (lossHistory.length < 2) return;
        const data = lossHistory.slice(-80);
        const maxL = Math.max(...data, 0.01);
        ctx.beginPath();
        ctx.strokeStyle = '#f59e0b';
        ctx.lineWidth = 1.5;
        data.forEach((l, i) => {
            const x = (i / (data.length - 1)) * w;
            const y = h - (l / maxL) * h;
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        });
        ctx.stroke();
    }

    // ── HUD update ─────────────────────────────────────────────────────────
    function updateHUD(action, qVals) {
        const stats = agent.stats;
        const drSt = drone.state;
        const f2 = v => typeof v === 'number' ? v.toFixed(2) : '—';
        const f4 = v => typeof v === 'number' ? v.toFixed(4) : '—';

        if (hud.episode) hud.episode.textContent = rlEnv.episode;
        if (hud.steps) hud.steps.textContent = stats.totalSteps.toLocaleString();
        if (hud.trainSteps) hud.trainSteps.textContent = stats.trainingSteps.toLocaleString();

        const epsPct = ((stats.epsilon - 0.05) / (1.0 - 0.05)) * 100;
        if (hud.epsilon) hud.epsilon.textContent = stats.epsilon.toFixed(4);
        if (hud.epsBar) hud.epsBar.style.width = epsPct.toFixed(0) + '%';

        const bufPct = (stats.bufferSize / 100000) * 100;
        if (hud.bufSize) hud.bufSize.textContent = stats.bufferSize.toLocaleString();
        if (hud.bufBar) hud.bufBar.style.width = Math.min(100, bufPct).toFixed(1) + '%';

        if (hud.loss) hud.loss.textContent = stats.lastLoss != null ? f4(stats.lastLoss) : '–';
        if (hud.reward) hud.reward.textContent = f2(rlEnv.episodeReward);
        if (hud.avgReward) hud.avgReward.textContent = f2(rlEnv.avgReward(50));

        if (hud.warmup) {
            hud.warmup.textContent = stats.warmupDone ? '✔ Training' : `Warmup… ${stats.bufferSize}/${DQNAgent.HP.warmupSteps}`;
            hud.warmup.style.color = stats.warmupDone ? '#34d399' : '#fbbf24';
        }

        if (hud.action && action != null) hud.action.textContent = ACTION_LABELS[action];

        // Q-value bars
        if (qVals && hud.qBars[0]) {
            const minQ = Math.min(...qVals), maxQ = Math.max(...qVals);
            const qRange = maxQ - minQ || 1;
            qVals.forEach((q, i) => {
                const pct = ((q - minQ) / qRange) * 100;
                if (hud.qBars[i]) {
                    hud.qBars[i].style.width = pct.toFixed(0) + '%';
                    hud.qBars[i].style.background = i === action ? '#38bdf8' : '#818cf8';
                }
                if (hud.qVals[i]) hud.qVals[i].textContent = q.toFixed(3);
            });
        }

        // Drone telemetry
        if (hud.alt) hud.alt.textContent = f2(drSt.altitude) + ' m';
        if (hud.posY) hud.posY.textContent = f2(drSt.position.y) + ' m';
        if (hud.velY) hud.velY.textContent = f2(drSt.velocity.y) + ' m/s';
        if (hud.speed) hud.speed.textContent = (drSt.speed * 3.6).toFixed(1) + ' km/h';
        if (hud.totalT) hud.totalT.textContent = drSt.totalThrust.toFixed(1) + ' N';

        // Motor bars
        drSt.thrusts.forEach((T, i) => {
            const pct = (T / 15) * 100;
            if (hud.mBars[i]) {
                hud.mBars[i].style.width = pct.toFixed(0) + '%';
                const hue = 120 - pct * 1.2;
                hud.mBars[i].style.background = `hsl(${hue},80%,50%)`;
            }
            if (hud.mVals[i]) hud.mVals[i].textContent = T.toFixed(1);
        });

        // Motor symmetry balance bars (F/R = front vs rear, L/R = left vs right)
        // Motor layout: 0=FR, 1=FL, 2=RL, 3=RR
        const th = drSt.thrusts;
        const frontPair = (th[0] || 0) + (th[1] || 0);
        const rearPair = (th[2] || 0) + (th[3] || 0);
        const leftPair = (th[1] || 0) + (th[2] || 0);
        const rightPair = (th[0] || 0) + (th[3] || 0);
        const totalPair = frontPair + rearPair;
        const frRatio = totalPair > 0.01 ? (frontPair / totalPair * 100) : 50;
        const lrTot = leftPair + rightPair;
        const lrRatio = lrTot > 0.01 ? (leftPair / lrTot * 100) : 50;
        const frImbal = Math.abs(frRatio - 50);
        const lrImbal = Math.abs(lrRatio - 50);
        if (hud.frBar) hud.frBar.style.width = frRatio.toFixed(0) + '%';
        if (hud.lrBar) hud.lrBar.style.width = lrRatio.toFixed(0) + '%';
        const symColor = (imbal) => imbal < 5 ? 'var(--gn)' : imbal < 12 ? 'var(--yw)' : 'var(--rd)';
        if (hud.frVal) { hud.frVal.textContent = frImbal < 5 ? 'OK' : frImbal.toFixed(0) + '%'; hud.frVal.style.color = symColor(frImbal); }
        if (hud.lrVal) { hud.lrVal.textContent = lrImbal < 5 ? 'OK' : lrImbal.toFixed(0) + '%'; hud.lrVal.style.color = symColor(lrImbal); }

        // FPS
        fpsFrames++;
        const now = performance.now();
        if (now - fpsLast >= 500) {
            currentFPS = Math.round(fpsFrames / ((now - fpsLast) / 1000));
            fpsFrames = 0;
            fpsLast = now;
            if (hud.fps) hud.fps.textContent = currentFPS;
        }
    }

    // ── Toast notification ──────────────────────────────────────────────────
    function showToast(msg) {
        const el = $('toast');
        if (!el) return;
        el.textContent = msg;
        el.style.opacity = '1';
        setTimeout(() => { el.style.opacity = '0'; }, 2000);
    }

    // ── Main RL loop — 10-step continuous training cycle ─────────────────────
    //
    //  [1]  Observe current drone state (position, velocity, orientation, ω)
    //  [2]  Feed state into neural network (online Q-network forward pass)
    //  [3]  Select action using ε-greedy policy (explore or exploit)
    //  [4]  Apply motor thrust adjustments to simulator
    //  [5]  Simulate physics timestep (Cannon.js + DroneModel)
    //  [6]  Observe new state after physics
    //  [7]  Calculate reward using reward function
    //  [8]  Store (s, a, r, s', done) in replay buffer
    //  [9]  Train neural network using random mini-batch from buffer (async)
    //  [10] Loop continues — crash → auto-reset to (0,10,0), no episode limit
    //
    // ─────────────────────────────────────────────────────────────────────────
    const FIXED_STEP = 1 / 60;
    let accumulator = 0;
    let lastTime = performance.now();
    let lastAction = null;
    let lastQVals = null;

    window.RL_LOOP_STEP = 0;

    // ── Stage management ─────────────────────────────────────────────────────
    // FLIGHT_STAGE 1 = Autonomous PD takeoff (from ground → 10 m)
    // FLIGHT_STAGE 2 = RL hover training (continuous)
    let FLIGHT_STAGE = 1;
    let takeoffStableTimer = 0;   // frames spent at target altitude with low error

    /** Initialize drone at ground level for Stage 1. */
    function initTakeoffStage() {
        const cfg = RLEnvironment.CONFIG;
        // Reset to ground — y=0, zero velocity, level orientation
        drone.reset({ x: 0, y: 0.3, z: 0 });
        drone.arm();
        // Seed thrusts at hover so the PD controller starts from a sane point
        const h = drone.hoverThrust;
        rlEnv._currentThrusts = [h, h, h, h];
        drone.setThrusts(h, h, h, h);
        // Patch CONFIG.hoverThrust with the actual physics value
        RLEnvironment.CONFIG.hoverThrust = h;
        console.log(`%c🚀 Stage 1: Takeoff initiated from ground. hoverThrust=${h.toFixed(3)} N`, 'color:#38bdf8;font-weight:bold');
    }

    /**
     * Stage 1 — PD takeoff controller (runs every physics tick).
     * Algorithm:
     *   baseThr  = hoverThrust + climbTerm  (altitude-error P controller)
     *   rollCorr / pitchCorr: PD on roll/pitch → differential per-motor correction
     *   Once stable at targetY for 1 s → switch to Stage 2
     */
    function runTakeoffStep() {
        const cfg = RLEnvironment.CONFIG;
        const s = rlEnv.getState();
        const [px, py, pz, vx, vy, vz, roll, pitch] = s;
        const DEG = Math.PI / 180;
        const h = cfg.hoverThrust;

        // Altitude P+D control: target climb speed decreases as altitude nears 10 m
        const altErr = cfg.targetY - py;
        const targetVy = Math.max(0.1, Math.min(3.5, altErr * 0.6));
        const vyErr = targetVy - vy;
        const baseThr = Math.max(0.5, Math.min(14, h + vyErr * 0.8));

        // Attitude PD control: roll/pitch correction applied differentially to motors
        // Motor layout: 0=FR, 1=FL, 2=RL, 3=RR
        // Roll: roll> 0 (right side low) → increase left motors, reduce right
        // Pitch: pitch > 0 (nose down / front low) → increase front, reduce rear
        const rollCorr = roll * 2.8 + s[9] * 0.45;  // P+D on roll
        const pitchCorr = pitch * 2.8 + s[10] * 0.45;  // P+D on pitch

        const t0 = Math.max(0, Math.min(15, baseThr - rollCorr * 0.5 + pitchCorr * 0.5));  // FR
        const t1 = Math.max(0, Math.min(15, baseThr + rollCorr * 0.5 + pitchCorr * 0.5));  // FL
        const t2 = Math.max(0, Math.min(15, baseThr + rollCorr * 0.5 - pitchCorr * 0.5));  // RL
        const t3 = Math.max(0, Math.min(15, baseThr - rollCorr * 0.5 - pitchCorr * 0.5));  // RR

        drone.setThrusts(t0, t1, t2, t3);
        rlEnv._currentThrusts = [t0, t1, t2, t3];

        // Transition check: altitude reached, velocity calm, attitude level
        const altOK = Math.abs(altErr) < 0.25;
        const vyOK = Math.abs(vy) < 0.25;
        const tiltOK = Math.max(Math.abs(roll), Math.abs(pitch)) < 4 * DEG;

        if (altOK && vyOK && tiltOK) {
            takeoffStableTimer++;
        } else {
            takeoffStableTimer = 0;
        }

        // Transition after 1 s (60 frames) of stable hover at target altitude
        if (takeoffStableTimer >= 60) {
            FLIGHT_STAGE = 2;
            // Reset RL env at current position to start RL from a clean slate
            currentState = rlEnv.reset();
            console.log('%c✅ Stage 2: RL hover training begins!', 'color:#10b981;font-size:14px;font-weight:bold');
        }
    }

    // Boot Stage 1 immediately
    initTakeoffStage();
    currentState = rlEnv.getState();   // initial state after takeoff init

    async function loop() {
        const now = performance.now();
        const dt = Math.min((now - lastTime) / 1000, 0.05);
        lastTime = now;
        frameCount++;

        if (isTrainingOn) {

            if (FLIGHT_STAGE === 1) {
                // ── STAGE 1: Autonomous PD Takeoff Controller ─────────────────────
                // Physics tick every fixed step — controller sets motor thrusts
                accumulator += dt;
                while (accumulator >= FIXED_STEP) {
                    runTakeoffStep();
                    drone.tick(FIXED_STEP);    // motor lag update + apply thrust forces
                    world.step(FIXED_STEP);    // Cannon.js physics integration
                    accumulator -= FIXED_STEP;
                }
                // Show takeoff progress in HUD
                const ts = rlEnv.getState();
                if (hud.episode) hud.episode.textContent = '🚀 Stage 1 — Takeoff';
                if (hud.altitude) hud.altitude.textContent = ts[1].toFixed(2) + ' m';
                if (hud.epReward) hud.epReward.textContent = (10 - ts[1]).toFixed(2) + ' m left';

            } else {
                // ── STAGE 2: RL 10-step training cycle ───────────────────────────
                window.RL_LOOP_STEP = 1;
                const s = rlEnv.getState();
                // s = [px, py, pz,  vx, vy, vz,  roll, pitch, yaw,  ωx, ωy, ωz]
                // (position, velocity, orientation, angular velocity — 12 floats)

                // ── [2] Neural network forward pass (Q-values) ────────────────────
                window.RL_LOOP_STEP = 2;
                if (frameCount % 6 === 0) lastQVals = agent.getQValues(s);

                // ── [3] ε-greedy action selection ─────────────────────────────────
                window.RL_LOOP_STEP = 3;
                const action = agent.selectAction(s);
                lastAction = action;

                // ── [4] Apply motor thrust adjustments ────────────────────────────
                window.RL_LOOP_STEP = 4;
                rlEnv.applyAction(action);   // updates _currentThrusts, calls drone.setThrusts()

                // ── [5] Simulate physics timestep ─────────────────────────────────
                window.RL_LOOP_STEP = 5;
                accumulator += dt;
                while (accumulator >= FIXED_STEP) {
                    drone.tick(FIXED_STEP);    // motor lag, thrust forces, aero drag
                    world.step(FIXED_STEP);   // Cannon.js rigid body integration
                    accumulator -= FIXED_STEP;
                }

                // ── [6] Observe new state ─────────────────────────────────────────
                window.RL_LOOP_STEP = 6;
                const sNext = rlEnv.getState();
                const done = rlEnv.isDone(sNext);

                // ── [7] Calculate reward ──────────────────────────────────────────
                window.RL_LOOP_STEP = 7;
                const reward = rlEnv.reward(sNext, done);
                rlEnv.recordStep(reward, done);   // accumulates episodeReward, steps, history

                // ── [8] Store experience in replay buffer ─────────────────────────
                window.RL_LOOP_STEP = 8;
                agent.remember(s, action, reward, sNext, done);

                // ── [9] Train neural network (non-blocking async) ─────────────────
                window.RL_LOOP_STEP = 9;
                agent.train().then(loss => {
                    if (loss != null) {
                        lossHistory.push(loss);
                        if (lossHistory.length > 1000) lossHistory.shift();
                        if (frameCount % 30 === 0) drawLossChart();
                    }
                });

                // ── [10] Continue — auto-reset on crash or timeout ────────────────
                window.RL_LOOP_STEP = 10;
                if (done || rlEnv.stepCount >= RLEnvironment.CONFIG.maxEpisodeSteps) {
                    // Crash/timeout: auto-reset to (0, 10, 0) with zero velocity — ALWAYS
                    rlEnv.reset();     // calls DroneModel.reset({x:0, y:10, z:0})
                    drawRewardChart();
                    // No episode limit — the loop continues immediately
                }
            } // end Stage 2 RL block

        } else {
            // Paused: still advance physics so the 3D scene remains rendered
            accumulator += dt;
            while (accumulator >= FIXED_STEP) accumulator -= FIXED_STEP;
        }

        // ── Render + HUD (every frame, regardless of training state) ─────────
        scene.update(dt);
        scene.render();
        updateHUD(lastAction, lastQVals);

        // ── Telemetry export → Real-Time Dashboard ────────────────────────────
        // Push via BroadcastChannel EVERY frame (instant, no I/O cost).
        // Write to localStorage every 2 frames (persists for reconnect).
        {
            const s = rlEnv.getState();
            const telPayload = {
                t: performance.now(),  // use high-res timer for sub-ms precision
                pos: [s[0], s[1], s[2]],
                vel: [s[3], s[4], s[5]],
                ori: [s[6], s[7], s[8]],
                angVel: [s[9], s[10], s[11]],
                thrusts: Array.from(rlEnv.currentThrusts),
                epReward: rlEnv.episodeReward,
                episode: rlEnv.episode,
                epsilon: agent._epsilon,
                trainSteps: agent._totalTrainSteps,
                lastLoss: agent._lastLoss || 0,
                targetY: RLEnvironment.CONFIG.targetY,
                bufferSize: agent._bufferSize,
                goodSize: agent._goodSize,
                training: isTrainingOn,
            };
            // Instant push to dashboard via BroadcastChannel (every frame)
            if (window._telChannel) {
                try { window._telChannel.postMessage(telPayload); } catch (_) { }
            }
            // Persist to localStorage for tab reconnection (every 2 frames)
            if (frameCount % 2 === 0) {
                try { localStorage.setItem('rl_telemetry', JSON.stringify(telPayload)); } catch (_) { }
            }
        }
    }

    // ── Expose control buttons ────────────────────────────────────────────────
    window.toggleTraining = function () {
        isTrainingOn = !isTrainingOn;
        updatePauseBtn();
    };
    window.resetEpisode = manualReset;
    window.saveWeights = () => agent.saveWeights().then(() => showToast('Saved ✓'));
    window.loadWeights = () => agent.loadWeights().then(ok => showToast(ok ? 'Loaded ✓' : 'No save found'));
    window.setTargetAlt = (v) => {
        const alt = parseFloat(v);
        RLEnvironment.CONFIG.targetY = alt;
        RLEnvironment.CONFIG.innerBandLo = alt - 0.3;
        RLEnvironment.CONFIG.innerBandHi = alt + 0.3;
        RLEnvironment.CONFIG.outerBandLo = alt - 1.0;
        RLEnvironment.CONFIG.outerBandHi = alt + 1.0;
        $('target-alt-val').textContent = alt.toFixed(1) + ' m';
    };

    // ── Visibility-aware loop: RAF when visible, setInterval when hidden ──────
    //   Browsers throttle/pause requestAnimationFrame for background tabs.
    //   We switch to setInterval at 30 Hz when the tab is hidden so physics
    //   keeps running and telemetry keeps flowing to the dashboard.
    let _rafId = null;
    let _intervalId = null;

    function startRAF() {
        if (_intervalId) { clearInterval(_intervalId); _intervalId = null; }
        if (!_rafId) {
            function rafLoop() { _rafId = requestAnimationFrame(rafLoop); loop(); }
            _rafId = requestAnimationFrame(rafLoop);
        }
    }

    function startInterval() {
        if (_rafId) { cancelAnimationFrame(_rafId); _rafId = null; }
        if (!_intervalId) {
            _intervalId = setInterval(loop, 1000 / 30); // 30 Hz background tick
        }
    }

    document.addEventListener('visibilitychange', () => {
        if (document.hidden) {
            startInterval();
        } else {
            startRAF();
        }
    });

    // Boot
    try { window._telChannel = new BroadcastChannel('rl_telemetry_channel'); } catch (_) { window._telChannel = null; }
    startRAF(); // start with smooth RAF loop

    console.log('%c🤖 DQN Training Active — 10-step continuous loop', 'color:#38bdf8;font-size:14px;font-weight:bold');
    console.log('%cSteps: [1]Observe [2]ForwardPass [3]ε-greedy [4]Apply [5]Physics [6]NextState [7]Reward [8]Buffer [9]Train [10]AutoReset', 'color:#94a3b8;font-size:11px');
    console.log('%cAPI: AGENT.stats | RL_ENV.episode | RL_LOOP_STEP | DRONE.state', 'color:#94a3b8;font-size:11px');
})();
