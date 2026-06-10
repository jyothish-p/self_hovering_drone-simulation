/**
 * dqn-agent.js — Deep Q-Network Agent
 * ─────────────────────────────────────────────────────────────────────────────
 * Architecture:   Input(12) → Dense(128,relu) → Dense(128,relu) → Dense(64,relu) → Dense(9,linear)
 * Algorithm:      DQN with experience replay + target network
 * Optimizer:      Adam (lr = 0.001)
 * Loss:           Huber loss (robust to outliers)
 * Exploration:    ε-greedy, exponential decay 1.0 → 0.05
 * Buffer:         Circular replay buffer, capacity 100,000
 * Target sync:    Every 1,000 training steps
 *
 * State (12 floats, all normalised to ≈ [-1, +1]):
 *   [px, py, pz,  vx, vy, vz,  roll, pitch, yaw,  ωx, ωy, ωz]
 *
 * Actions (9 discrete):
 *   0 M1↑  1 M1↓  2 M2↑  3 M2↓  4 M3↑  5 M3↓  6 M4↑  7 M4↓  8 hold
 *
 * API:
 *   agent.selectAction(state)           → action index
 *   agent.remember(s, a, r, s', done)   → stores in buffer
 *   agent.train()                       → async, returns loss (or null)
 *   agent.stats                         → { epsilon, bufferSize, totalSteps, trainingSteps }
 * ─────────────────────────────────────────────────────────────────────────────
 */
class DQNAgent {
    // ── Hyper-parameters ──────────────────────────────────────────────────────
    static HP = {
        // ── Architecture (4 hidden layers — stronger stabilization capacity) ──
        stateSize: 16,           // [px,py,pz, vx,vy,vz, roll,pitch,yaw, wx,wy,wz, t1,t2,t3,t4]
        actionSize: 17,           // 17 symmetry-aware multi-motor actions
        hiddenLayers: [256, 256, 128, 64],

        // ── Learning schedule ────────────────────────────────────────────────
        lr: 0.0005,       // lower — stable with large network + high update freq
        gamma: 0.995,        // higher — values long-term stability more

        // ── Exploration ──────────────────────────────────────────────────────
        epsilonStart: 1.0,
        epsilonEnd: 0.05,
        epsilonDecay: 0.9992,       // slightly slower decay for larger action space

        // ── Replay buffer ────────────────────────────────────────────────────
        bufferCapacity: 200_000,  // 2× larger for richer experience diversity
        batchSize: 128,      // 2× larger — more stable gradient estimates
        warmupSteps: 500,      // fill enough before training starts

        // ── Update frequency ─────────────────────────────────────────────────
        trainEveryNSteps: 1,
        nGradientSteps: 2,

        // ── Target network ───────────────────────────────────────────────────
        targetSyncFreq: 750,      // sync every 750 training steps

        // ── Prioritized dual-buffer (orientation-good experiences) ───────────
        goodRewardThreshold: 60.0,  // reward ≥ +60 → good pool (level + near altitude)
        goodSampleRatio: 0.70,
        goodPoolCapacity: 40_000,

        gradientClip: 1.0,
    };

    // State normalisation bounds (16 values)
    static STATE_BOUNDS = [
        100, 100, 100,                     // position        (m)       → /100
        20, 20, 20,                      // velocity        (m/s)     → /20
        Math.PI, Math.PI / 2, Math.PI,     // RPY             (rad)     → /π
        15, 15, 15,                      // angular vel     (rad/s)   → /15
        15, 15, 15, 15,                 // motor thrusts   (N)       → /15
    ];

    constructor() {
        const HP = DQNAgent.HP;
        this._epsilon = HP.epsilonStart;
        this._totalEnvSteps = 0;
        this._totalTrainSteps = 0;
        this._targetSyncTimer = 0;

        // ── General circular replay buffer ────────────────────────────────────
        this._buffer = new Array(HP.bufferCapacity);
        this._bufferHead = 0;
        this._bufferSize = 0;

        // ── Good-experience pool (prioritised hover stabilization) ────────────
        //    Filled with experiences whose reward ≥ goodRewardThreshold.
        //    Sampled at goodSampleRatio×batchSize to accelerate stabilization.
        this._goodBuf = new Array(HP.goodPoolCapacity);
        this._goodHead = 0;
        this._goodSize = 0;

        // Running stats
        this._lastLoss = null;
        this._lossSum = 0;   // accumulated over nGradientSteps for display
        this._isTraining = false;

        this._buildNetworks();
    }

    // ── Network construction ──────────────────────────────────────────────────
    _buildNetworks() {
        this._online = this._makeNet('online');
        this._target = this._makeNet('target');
        this._syncTarget();
        this._optimizer = tf.train.adam(DQNAgent.HP.lr);
    }

    _makeNet(name) {
        const HP = DQNAgent.HP;
        const net = tf.sequential({ name });
        // Hidden layers
        HP.hiddenLayers.forEach((units, i) => {
            net.add(tf.layers.dense({
                inputShape: i === 0 ? [HP.stateSize] : undefined,
                units,
                activation: 'relu',
                kernelInitializer: 'glorotUniform',
                name: `${name}_h${i + 1}`,
            }));
        });
        // Output: one Q-value per action (linear)
        net.add(tf.layers.dense({
            units: HP.actionSize,
            activation: 'linear',
            name: `${name}_out`,
        }));
        return net;
    }

    _syncTarget() {
        // getWeights() returns cloned tensors — safe to pass to setWeights().
        // Do NOT call dispose() on them; setWeights() consumes them internally.
        this._target.setWeights(this._online.getWeights());
    }

    // ── Policy ────────────────────────────────────────────────────────────────

    /** ε-greedy action selection. Returns action index 0–8. */
    selectAction(state) {
        if (Math.random() < this._epsilon) {
            return Math.floor(Math.random() * DQNAgent.HP.actionSize);
        }
        return tf.tidy(() => {
            const s = tf.tensor2d([this._normalize(state)], [1, DQNAgent.HP.stateSize]);
            return this._online.predict(s).argMax(1).dataSync()[0];
        });
    }

    /** Returns the Q-values array for the current state (for HUD). */
    getQValues(state) {
        return tf.tidy(() => {
            const s = tf.tensor2d([this._normalize(state)], [1, DQNAgent.HP.stateSize]);
            return Array.from(this._online.predict(s).dataSync());
        });
    }

    // ── Replay buffer ─────────────────────────────────────────────────────────

    remember(state, action, reward, nextState, done) {
        const HP = DQNAgent.HP;
        const entry = {
            state: this._normalize(state),
            action,
            reward,
            nextState: this._normalize(nextState),
            done: done ? 1.0 : 0.0,
        };

        // General buffer (circular)
        this._buffer[this._bufferHead] = entry;
        this._bufferHead = (this._bufferHead + 1) % HP.bufferCapacity;
        this._bufferSize = Math.min(this._bufferSize + 1, HP.bufferCapacity);
        this._totalEnvSteps++;

        // Good-experience pool — prioritise stabilization successes
        if (reward >= HP.goodRewardThreshold) {
            this._goodBuf[this._goodHead] = entry;
            this._goodHead = (this._goodHead + 1) % HP.goodPoolCapacity;
            this._goodSize = Math.min(this._goodSize + 1, HP.goodPoolCapacity);
        }
    }

    // ── Training ──────────────────────────────────────────────────────────────

    /**
     * Perform nGradientSteps gradient updates from a prioritised mini-batch.
     * Returns mean loss across all steps, or null if skipped.
     *
     * Optimizations active:
     *   · Runs every env step          (trainEveryNSteps = 1)
     *   · 2 gradient passes per call   (nGradientSteps   = 2)
     *   · 70 % of batch from good pool (stabilization focus)
     *   · Faster ε-decay               (0.9990 per step)
     *   · Target net synced at 500 steps
     */
    async train() {
        const HP = DQNAgent.HP;
        if (this._isTraining) return null;
        if (this._bufferSize < HP.warmupSteps) return null;
        if (this._totalEnvSteps % HP.trainEveryNSteps !== 0) return null;

        this._isTraining = true;
        let totalLoss = 0;

        try {
            // ── nGradientSteps successive gradient updates ─────────────────────
            for (let g = 0; g < HP.nGradientSteps; g++) {

                // Fresh batch each gradient step (reduces correlation)
                const batch = this._sampleBatch();

                const lossScalar = this._optimizer.minimize(() => {
                    return tf.tidy(() => {
                        const states = tf.tensor2d(batch.map(e => e.state), [HP.batchSize, HP.stateSize]);
                        const nextStates = tf.tensor2d(batch.map(e => e.nextState), [HP.batchSize, HP.stateSize]);
                        const rewards = tf.tensor1d(batch.map(e => e.reward));
                        const dones = tf.tensor1d(batch.map(e => e.done));
                        const actions = tf.tensor1d(batch.map(e => e.action), 'int32');

                        // ── Double DQN Bellman target ─────────────────────────
                        //    Online net selects action, target net evaluates value.
                        const nextQOnline = this._online.predict(nextStates);
                        const bestActions = nextQOnline.argMax(1);
                        const nextQTarget = this._target.predict(nextStates);

                        const notDone = tf.scalar(1.0).sub(dones);
                        const bellman = rewards.add(
                            notDone.mul(HP.gamma).mul(
                                nextQTarget.mul(tf.oneHot(bestActions, HP.actionSize)).sum(1)
                            )
                        );

                        // ── Current Q for taken actions ───────────────────────
                        const currentQ = this._online.predict(states);
                        const actionMask = tf.oneHot(actions, HP.actionSize).toFloat();
                        const takenQ = currentQ.mul(actionMask).sum(1);

                        // Huber loss — robust to large TD errors during exploration
                        return tf.losses.huberLoss(bellman, takenQ);
                    });
                }, true);  // no varList — optimizer auto-detects online net vars

                totalLoss += (await lossScalar.data())[0];
                lossScalar.dispose();

                this._totalTrainSteps++;
                this._targetSyncTimer++;

                // Target network sync
                if (this._targetSyncTimer >= HP.targetSyncFreq) {
                    this._syncTarget();
                    this._targetSyncTimer = 0;
                }
            }

            // ── Decay ε once per train() call (after all gradient steps) ──────
            this._epsilon = Math.max(HP.epsilonEnd, this._epsilon * HP.epsilonDecay);

            this._lastLoss = totalLoss / HP.nGradientSteps;

        } finally {
            this._isTraining = false;
        }

        return this._lastLoss;
    }

    // ── Prioritised dual-buffer sampling ─────────────────────────────────────
    //
    //  Splits the batch into:
    //    · nGood  = floor(batchSize × goodSampleRatio) from good-experience pool
    //    · nRest  = batchSize - nGood from the general buffer (uniform)
    //
    //  This ensures the agent continuously revisits successful hover transitions
    //  rather than drowning in crash/tumble experiences during warmup.
    _sampleBatch() {
        const HP = DQNAgent.HP;
        const batch = [];

        const nGood = this._goodSize > 0
            ? Math.floor(HP.batchSize * HP.goodSampleRatio)
            : 0;
        const nRest = HP.batchSize - nGood;

        // Good-experience pool samples (prioritised hover stabilisation)
        for (let i = 0; i < nGood; i++) {
            const idx = Math.floor(Math.random() * this._goodSize);
            batch.push(this._goodBuf[idx]);
        }

        // General buffer samples (covers exploration diversity)
        for (let i = 0; i < nRest; i++) {
            const idx = Math.floor(Math.random() * this._bufferSize);
            batch.push(this._buffer[idx]);
        }

        return batch;
    }

    // ── Normalisation ─────────────────────────────────────────────────────────
    _normalize(state) {
        return state.map((v, i) => {
            const bound = DQNAgent.STATE_BOUNDS[i];
            return Math.max(-1, Math.min(1, v / bound));
        });
    }

    // ── Accessors ─────────────────────────────────────────────────────────────
    get epsilon() { return this._epsilon; }
    get bufferSize() { return this._bufferSize; }
    get totalSteps() { return this._totalEnvSteps; }
    get trainSteps() { return this._totalTrainSteps; }
    get lastLoss() { return this._lastLoss; }
    get warmupDone() { return this._bufferSize >= DQNAgent.HP.warmupSteps; }

    get stats() {
        return {
            epsilon: this._epsilon,
            bufferSize: this._bufferSize,
            totalSteps: this._totalEnvSteps,
            trainingSteps: this._totalTrainSteps,
            lastLoss: this._lastLoss,
            warmupDone: this.warmupDone,
        };
    }

    /** Save weights to localStorage. */
    async saveWeights(key = 'dqn_drone') {
        await this._online.save(`localstorage://${key}`);
    }

    /** Load weights from localStorage. */
    async loadWeights(key = 'dqn_drone') {
        try {
            const loaded = await tf.loadLayersModel(`localstorage://${key}`);
            this._online.setWeights(loaded.getWeights());
            this._syncTarget();
            return true;
        } catch { return false; }
    }
}


// ─────────────────────────────────────────────────────────────────────────────
//  RLEnvironment — wraps DroneModel with step/reset/reward/done logic
// ─────────────────────────────────────────────────────────────────────────────
class RLEnvironment {
    static CONFIG = {
        // ── Target hover position (x=0, y=10, z=0) ───────────────
        targetX: 0,
        targetY: 10.0,
        targetZ: 0,

        // ── Physics hover thrust (N per motor at equilibrium) ─────
        // Approximate value — refined at runtime from drone.hoverThrust
        hoverThrust: 2.45,

        // ── Episode params ────────────────────────────────────────
        maxEpisodeSteps: 1800,   // 30 s — longer episodes reward sustained hover
        thrustDelta: 0.6,    // N per discrete action step (smaller = finer)

        // ── Termination thresholds ────────────────────────────────
        crashHeight: 0.10,                // ground collision
        tiltResetDeg: 35,                  // immediate flip reset (°)
        maxAngle: Math.PI * 0.65,      // ~117° hard limit
        maxAltitude: 95,

        // ── Spawn: always start at target for RL (Stage 2) ───────
        spawnAltLow: 10.0,
        spawnAltHigh: 10.0,

        // ── Reward thresholds (orientation-first scale) ───────────
        tiltPrimaryDeg: 2,    // °  +50 reward
        tiltGoodDeg: 5,    // °  +20 reward
        tiltSevereDeg: 15,   // °  −100 penalty
        tiltExtremeDeg: 25,   // °  −200 penalty (reset >35°)
        altPreciseLo: 9.8,  // m  +30 reward
        altPreciseHi: 10.2,
        altModLo: 9.5,  // m  +10 reward
        altModHi: 10.5,
        velStableThresh: 0.3,  // m/s  +15 reward
        angStableThresh: 0.3,  // rad/s +10 reward
        thrustSymmTol: 1.0,  // N  max pair imbalance for +20 symmetry reward
        thrustAsymPen: 4.0,  // N  min pair imbalance for −25 penalty
    };

    /**
     * 17 symmetry-aware actions.
     * Each entry is an array of [motorIndex, direction] pairs applied together.
     * Motors: 0=FR, 1=FL, 2=RL, 3=RR
     *
     * Symmetric (all-motor):         0=ALL↑   1=ALL↓
     * Pitch pair (front/rear):        2=nose↑  3=nose↓
     * Roll pair  (left/right):        4=roll+  5=roll−
     * Yaw diagonal:                   6=yaw CW 7=yaw CCW
     * Fine single-motor:              8–15 (M1↑ M1↓ M2↑ M2↓ M3↑ M3↓ M4↑ M4↓)
     * Hold:                           16
     */
    static ACTION_MAP = [
        /* 0  ALL↑ */[[0, +1], [1, +1], [2, +1], [3, +1]],
        /* 1  ALL↓ */[[0, -1], [1, -1], [2, -1], [3, -1]],
        /* 2  PITCH+ */[[0, +1], [1, +1], [2, -1], [3, -1]],  // FR+FL up, RL+RR down
        /* 3  PITCH− */[[0, -1], [1, -1], [2, +1], [3, +1]],  // FR+FL down, RL+RR up
        /* 4  ROLL+  */[[1, +1], [2, +1], [0, -1], [3, -1]],  // FL+RL up, FR+RR down
        /* 5  ROLL−  */[[0, +1], [3, +1], [1, -1], [2, -1]],  // FR+RR up, FL+RL down
        /* 6  YAW CW  */[[0, +1], [2, +1], [1, -1], [3, -1]],  // diagonal A up
        /* 7  YAW CCW */[[1, +1], [3, +1], [0, -1], [2, -1]],  // diagonal B up
        /* 8  M1↑ */[[0, +1]],
        /* 9  M1↓ */[[0, -1]],
        /* 10 M2↑ */[[1, +1]],
        /* 11 M2↓ */[[1, -1]],
        /* 12 M3↑ */[[2, +1]],
        /* 13 M3↓ */[[2, -1]],
        /* 14 M4↑ */[[3, +1]],
        /* 15 M4↓ */[[3, -1]],
        /* 16 HOLD */[],
    ];

    constructor(drone, world) {
        this._drone = drone;
        this._world = world;
        this._steps = 0;
        this._episode = 0;
        this._episodeReward = 0;
        this._episodeHistory = [];
        this._historyMax = 200;
        this._currentThrusts = [0, 0, 0, 0];
        this._airborneSteps = 0;   // total steps the drone has survived (across all episodes)
        this._lastRewardInfo = {};  // per-component breakdown for HUD
    }

    /**
     * Reset episode.
     * Always spawns at the target hover position (0, targetY, 0) with
     * all velocities zeroed.  DroneModel.reset() handles the physics body.
     * No upper limit on the number of episodes allowed.
     */
    reset() {
        const cfg = RLEnvironment.CONFIG;

        // ── Step 1 of the training loop: full state reset ──────────────────
        //    Position  → (0, targetY, 0)   [user-specified crash-reset coords]
        //    Velocity  → (0, 0, 0)          [DroneModel.reset() calls setZero()]
        //    Ang. vel. → (0, 0, 0)          [DroneModel.reset() calls setZero()]
        //    Rotation  → identity quaternion [DroneModel.reset() sets to (0,0,0,1)]
        this._drone.reset({ x: cfg.targetX, y: cfg.targetY, z: cfg.targetZ });

        // Start with symmetric hover thrust so the drone doesn't immediately fall
        const h = this._drone.hoverThrust;
        this._currentThrusts = [h, h, h, h];
        this._drone.arm();
        this._drone.setThrusts(...this._currentThrusts);

        this._steps = 0;
        this._episodeReward = 0;
        this._lastRewardInfo = {};
        this._episode++;
        return this.getState();
    }

    step(actionIdx) {
        const cfg = RLEnvironment.CONFIG;
        const action = RLEnvironment.ACTION_MAP[actionIdx];

        // Apply action
        if (action !== null) {
            const [mIdx, dir] = action;
            const maxT = 15.0;
            this._currentThrusts[mIdx] = Math.max(0, Math.min(maxT,
                this._currentThrusts[mIdx] + dir * cfg.thrustDelta
            ));
        }
        this._drone.setThrusts(...this._currentThrusts);

        // Physics step happens in the main loop externally —
        // we just read the new state here after it's stepped.
        const nextState = this.getState();
        const done = this._isDone(nextState);
        const reward = this._computeReward(nextState, done);

        this._episodeReward += reward;
        this._steps++;

        if (done || this._steps >= cfg.maxEpisodeSteps) {
            this._episodeHistory.push(this._episodeReward);
            if (this._episodeHistory.length > this._historyMax) {
                this._episodeHistory.shift();
            }
        }

        return { nextState, reward, done: done || this._steps >= cfg.maxEpisodeSteps };
    }

    getState() {
        const s = this._drone.state;
        const [t1, t2, t3, t4] = this._currentThrusts;
        // 16-element state: position, velocity, orientation, angular velocity, motor thrusts
        return [
            s.position.x, s.position.y, s.position.z,
            s.velocity.x, s.velocity.y, s.velocity.z,
            s.orientation.roll, s.orientation.pitch, s.orientation.yaw,
            s.angularVelocity.x, s.angularVelocity.y, s.angularVelocity.z,
            t1, t2, t3, t4,      // motor thrusts (N) — normalised in DQNAgent._normalize()
        ];
    }

    _isDone(state) {
        const [px, py, pz, , , , roll, pitch] = state;
        const cfg = RLEnvironment.CONFIG;
        const DEG = Math.PI / 180;
        if (py < cfg.crashHeight) return true;  // ground collision
        if (Math.abs(roll) > cfg.tiltResetDeg * DEG) return true;  // extreme flip
        if (Math.abs(pitch) > cfg.tiltResetDeg * DEG) return true;
        if (Math.abs(roll) > cfg.maxAngle) return true;  // hard limit
        if (Math.abs(pitch) > cfg.maxAngle) return true;
        if (py > cfg.maxAltitude) return true;  // out of world
        return false;
    }

    /**
     * Orientation-first reward function for staged hover stabilization.
     * Priority: orientation stability → altitude → velocity → motor symmetry.
     *
     * COMPONENT                       VALUE    CONDITION
     * ──────────────────────────────────────────────────────────────────────
     * Orientation primary              +50     tilt ≤ 2°
     * Orientation good                 +20     tilt ≤ 5°
     * Tilt Gaussian (smooth gradient)  +8      always (exp over roll²+pitch²)
     * Severe tilt penalty             -100     tilt > 15°
     * Extreme tilt penalty            -200     tilt > 25°
     * Altitude precise                +30     9.8–10.2 m
     * Altitude moderate               +10     9.5–10.5 m
     * Altitude Gaussian               +10     always (exp over err²)
     * Vertical velocity               +15     |vy| < 0.3 m/s
     * Angular velocity                +10     |ω| < 0.3 rad/s
     * Motor symmetry (balanced)       +20     opp. pairs within 1 N
     * Hover thrust magnitude          +10     total ≈ 4×hoverThrust
     * Motor asymmetry penalty         -25     opp. pairs differ > 4 N
     * Survival                         +2     per stable airborne step
     * Crash                           -300     episode-terminal
     *
     * Peak per step (perfect stable hover): ≈95
     */
    _computeReward(state, done) {
        if (done) return -300;

        const [px, py, pz, vx, vy, vz, roll, pitch, yaw, wx, wy, wz, t1, t2, t3, t4] = state;
        const cfg = RLEnvironment.CONFIG;
        const DEG = Math.PI / 180;
        let reward = 0;

        // Orientation metrics
        const absRoll  = Math.abs(roll);
        const absPitch = Math.abs(pitch);
        const tiltDeg  = Math.max(absRoll, absPitch) / DEG;

        // ── 1–3. ORIENTATION STABILITY (primary objective) ──────────────────
        if (tiltDeg <= cfg.tiltPrimaryDeg) {
            reward += 50;   // near-perfect level
        } else if (tiltDeg <= cfg.tiltGoodDeg) {
            reward += 20;   // good level (±5°)
        }
        // Smooth Gaussian — gradient at every angle
        reward += 8.0 * Math.exp(-6.0 * (roll * roll + pitch * pitch));

        // ── 4–5. SEVERE TILT PENALTIES ──────────────────────────────────────
        if (tiltDeg > cfg.tiltExtremeDeg) {
            reward -= 200;  // extreme — immediate reset coming
        } else if (tiltDeg > cfg.tiltSevereDeg) {
            reward -= 100;  // severe — strong discouragement
        }

        // ── 6–8. ALTITUDE REWARD (secondary objective) ──────────────────────
        const altErr  = py - cfg.targetY;
        const altErr2 = altErr * altErr;
        if (py >= cfg.altPreciseLo && py <= cfg.altPreciseHi) {
            reward += 30;   // ±0.2 m precision band
        } else if (py >= cfg.altModLo && py <= cfg.altModHi) {
            reward += 10;   // ±0.5 m moderate band
        }
        reward += 10.0 * Math.exp(-2.0 * altErr2);  // smooth gradient

        // ── 9. VERTICAL VELOCITY ─────────────────────────────────────────────
        if (Math.abs(vy) < cfg.velStableThresh) {
            reward += 15;
        }

        // ── 10. ANGULAR VELOCITY ─────────────────────────────────────────────
        const angSpeed = Math.sqrt(wx*wx + wy*wy + wz*wz);
        if (angSpeed < cfg.angStableThresh) {
            reward += 10;
        }

        // ── 11–13. MOTOR SYMMETRY (critical flight dynamics) ────────────────
        // Motor layout: 0=FR, 1=FL, 2=RL, 3=RR
        const _t1 = t1 || 0, _t2 = t2 || 0, _t3 = t3 || 0, _t4 = t4 || 0;
        const frontPair     = _t1 + _t2;
        const rearPair      = _t3 + _t4;
        const rightPair     = _t1 + _t4;
        const leftPair      = _t2 + _t3;
        const frontRearDiff = Math.abs(frontPair - rearPair);
        const leftRightDiff = Math.abs(leftPair  - rightPair);
        const totalThrust   = _t1 + _t2 + _t3 + _t4;

        if (frontRearDiff <= cfg.thrustSymmTol && leftRightDiff <= cfg.thrustSymmTol) {
            reward += 20;   // balanced opposite pairs
        } else if (frontRearDiff > cfg.thrustAsymPen || leftRightDiff > cfg.thrustAsymPen) {
            reward -= 25;   // dangerous asymmetry
        }
        if (Math.abs(totalThrust - 4 * cfg.hoverThrust) < 2.0) {
            reward += 10;   // thrust near hover equilibrium
        }

        // ── 14. SURVIVAL ─────────────────────────────────────────────────────
        if (tiltDeg <= cfg.tiltSevereDeg && py > cfg.crashHeight) {
            reward += 2;    // +2 per stable airborne step
        }

        return reward;
    }


    // ── Public methods used by the decomposed 10-step training loop ──────────

    /**
     * Step 4: Apply a discrete action — update motor thrusts.
     * (In the new loop, physics is stepped separately in Step 5.)
     */
    applyAction(actionIdx) {
        const cfg = RLEnvironment.CONFIG;
        // ACTION_MAP entries are arrays of [motorIdx, dir] pairs — apply all atomically
        const moves = RLEnvironment.ACTION_MAP[actionIdx] || [];
        for (const [mIdx, dir] of moves) {
            this._currentThrusts[mIdx] = Math.max(0, Math.min(15,
                this._currentThrusts[mIdx] + dir * cfg.thrustDelta
            ));
        }
        this._drone.setThrusts(...this._currentThrusts);
    }

    /** Step 6: Check if the episode is done (crash / flip / out-of-bounds). */
    isDone(state) { return this._isDone(state); }

    /** Step 7: Compute and return the step reward. */
    reward(state, done) { return this._computeReward(state, done); }

    /**
     * Step 7b: Record a step's reward and manage episode history.
     * Separates accounting from computation so the loop stays clean.
     */
    recordStep(reward, done) {
        this._episodeReward += reward;
        this._steps++;
        if (done || this._steps >= RLEnvironment.CONFIG.maxEpisodeSteps) {
            this._episodeHistory.push(this._episodeReward);
            if (this._episodeHistory.length > this._historyMax) {
                this._episodeHistory.shift();
            }
        }
    }

    // ── Accessors ─────────────────────────────────────────────────────────────
    // Running average reward (last N episodes)
    avgReward(n = 50) {
        const h = this._episodeHistory;
        if (h.length === 0) return 0;
        const slice = h.slice(-n);
        return slice.reduce((a, b) => a + b, 0) / slice.length;
    }

    get episode() { return this._episode; }
    get stepCount() { return this._steps; }           // step 10 uses this
    get stepInEpisode() { return this._steps; }           // legacy alias
    get episodeReward() { return this._episodeReward; }
    get episodeHistory() { return this._episodeHistory; }
    get currentThrusts() { return this._currentThrusts; }
    get targetAltitude() { return RLEnvironment.CONFIG.targetY; }   // updated key
}
