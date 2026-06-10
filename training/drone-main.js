/**
 * drone-main.js — Main bootstrap, game loop, keyboard controls, HUD, PID hover
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * Controls (keyboard):
 *   W / S         — throttle up / down           (all motors ±)
 *   ↑ / ↓        — pitch forward / backward       (M1,M4 vs M2,M3)
 *   ← / →        — roll left / right             (M2,M3 vs M1,M4)
 *   Q / E         — yaw left / right              (M2,M4 vs M1,M3)
 *   SPACE         — ARM / DISARM toggle
 *   H             — Hold altitude (PID hover mode)
 *   C             — cycle camera mode (follow → orbit → top)
 *   R             — reset drone to origin
 *
 * Exposed globals:
 *   window.DRONE  — DroneModel
 *   window.WORLD  — CANNON.World
 *   window.SCENE_MGR — DroneScene
 * ─────────────────────────────────────────────────────────────────────────────
 */

(function () {
    // ── Physics world ────────────────────────────────────────────────────────
    const world = new CANNON.World();
    world.gravity.set(0, -9.81, 0);
    world.broadphase = new CANNON.SAPBroadphase(world);
    world.solver.iterations = 20;

    // Ground body
    const groundMat = new CANNON.Material('ground');
    const groundBody = new CANNON.Body({ mass: 0, material: groundMat });
    groundBody.addShape(new CANNON.Plane());
    groundBody.quaternion.setFromAxisAngle(new CANNON.Vec3(1, 0, 0), -Math.PI / 2);
    world.addBody(groundBody);

    // Default contact
    world.defaultContactMaterial.friction = 0.4;
    world.defaultContactMaterial.restitution = 0.1;

    // ── Drone ────────────────────────────────────────────────────────────────
    const drone = new DroneModel(world, { x: 0, y: 0.15, z: 0 });
    window.DRONE = drone;
    window.WORLD = world;

    // ── 3D Scene ─────────────────────────────────────────────────────────────
    const sceneManager = new DroneScene(drone);
    window.SCENE_MGR = sceneManager;

    // ── Keyboard state ───────────────────────────────────────────────────────
    const keys = {};
    window.addEventListener('keydown', e => { keys[e.code] = true; handleKeyDown(e); });
    window.addEventListener('keyup', e => { keys[e.code] = false; });

    // ── Control parameters ───────────────────────────────────────────────────
    const BASE_HOVER = drone.hoverThrust;    // ~3.43 N per motor
    let throttle = 0;                    // 0–1 (0 = no thrust, 1 = max)
    const MAX_DELTA = 3.0;                  // N max pitch/roll/yaw differential
    const THROTTLE_RATE = 2.0;               // N/s change per frame
    const TILT_RATE = 4.0;

    let pitchInput = 0;   // −1…+1
    let rollInput = 0;
    let yawInput = 0;
    let hoverMode = false;

    // ── PID hover controller ─────────────────────────────────────────────────
    const pid = {
        kp: 3.0, ki: 0.5, kd: 1.5,
        intErr: 0, prevErr: 0,
        target: 1.5,   // target altitude (m)
        compute(altErr, dt) {
            this.intErr += altErr * dt;
            this.intErr = Math.max(-5, Math.min(5, this.intErr));  // anti-windup
            const deriv = (altErr - this.prevErr) / dt;
            this.prevErr = altErr;
            return this.kp * altErr + this.ki * this.intErr + this.kd * deriv;
        }
    };

    function handleKeyDown(e) {
        if (e.code === 'Space') {
            e.preventDefault();
            if (drone.isArmed) { drone.disarm(); throttle = 0; hoverMode = false; }
            else { drone.arm(); }
            updateArmStatus();
        }
        if (e.code === 'KeyH') {
            hoverMode = !hoverMode;
            if (hoverMode) {
                pid.target = Math.max(0.5, drone.state.altitude);
                pid.intErr = 0;
                pid.prevErr = 0;
                if (!drone.isArmed) drone.arm();
                throttle = BASE_HOVER / DroneModel.SPECS.maxThrust;
            }
            document.getElementById('hover-badge').textContent =
                hoverMode ? '🔵 HOLD ALT ON' : '⚪ HOLD ALT OFF';
        }
        if (e.code === 'KeyC') {
            const modes = ['follow', 'orbit', 'top'];
            const cur = modes.indexOf(sceneManager.cameraMode);
            const next = modes[(cur + 1) % 3];
            sceneManager.setCameraMode(next);
            document.getElementById('cam-mode').textContent = next.toUpperCase();
        }
        if (e.code === 'KeyR') { drone.reset({ x: 0, y: 0.15, z: 0 }); throttle = 0; hoverMode = false; updateArmStatus(); }
    }

    // ── Per-frame control update ──────────────────────────────────────────────
    function applyControls(dt) {
        if (!drone.isArmed) return;

        // Throttle
        if (keys['KeyW']) throttle = Math.min(1.0, throttle + THROTTLE_RATE * dt / DroneModel.SPECS.maxThrust);
        if (keys['KeyS']) throttle = Math.max(0.0, throttle - THROTTLE_RATE * dt / DroneModel.SPECS.maxThrust);

        // Attitude stick inputs (smooth)
        pitchInput = 0; rollInput = 0; yawInput = 0;
        if (keys['ArrowUp']) pitchInput = +1;
        if (keys['ArrowDown']) pitchInput = -1;
        if (keys['ArrowLeft']) rollInput = +1;
        if (keys['ArrowRight']) rollInput = -1;
        if (keys['KeyQ']) yawInput = +1;
        if (keys['KeyE']) yawInput = -1;

        // Base thrust (N)
        let baseThrust = throttle * DroneModel.SPECS.maxThrust;

        // PID hover override on baseThrust
        if (hoverMode) {
            const altErr = pid.target - drone.state.altitude;
            const pidCorr = pid.compute(altErr, dt);
            baseThrust = BASE_HOVER + pidCorr;
            baseThrust = Math.max(0, Math.min(DroneModel.SPECS.maxThrust, baseThrust));
        }

        // Mix motor thrusts
        //   M1(+x,+z) M2(-x,+z) M3(-x,-z) M4(+x,-z)
        //   pitch+ : M1,M4 up, M2,M3 down  (tilt forward: +z motors  up)
        //   roll+  : M1,M3 up, M2,M4 down  (tilt right: +x motors up)
        //   yaw+   : M2,M4 up, M1,M3 down
        const pd = pitchInput * MAX_DELTA;
        const rd = rollInput * MAX_DELTA;
        const yd = yawInput * MAX_DELTA;

        const t0 = baseThrust + pd * 0 - pd + rd - yd;  // M1: fwd-,  right+, CW-yaw
        const t1 = baseThrust + pd - 0 - rd + yd;  // M2: fwd+,  left+,  CCW+yaw wait rewrite

        // Standard quadcopter mixing:
        // M1(FR,CW):  T + pitch_corr_front + roll_right - yaw_cw
        // M2(FL,CCW): T + pitch_corr_front - roll_right + yaw_ccw
        // M3(RL,CW):  T - pitch_corr_rear  - roll_right - yaw_cw
        // M4(RR,CCW): T - pitch_corr_rear  + roll_right + yaw_ccw
        //
        // pitch+ → nose up → reduce front motors, increase rear
        // roll+  → right bank → reduce right motors, increase left

        const pitchD = pitchInput * MAX_DELTA;
        const rollD = rollInput * MAX_DELTA;
        const yawD = yawInput * MAX_DELTA;

        drone.setThrusts(
            clampT(baseThrust - pitchD + rollD - yawD),   // M1 FR CW
            clampT(baseThrust - pitchD - rollD + yawD),   // M2 FL CCW
            clampT(baseThrust + pitchD - rollD - yawD),   // M3 RL CW
            clampT(baseThrust + pitchD + rollD + yawD)    // M4 RR CCW
        );
    }

    function clampT(v) { return Math.max(0, Math.min(DroneModel.SPECS.maxThrust, v)); }

    // ── HUD references ────────────────────────────────────────────────────────
    const $ = id => document.getElementById(id);
    const hud = {
        posX: $('hud-px'), posY: $('hud-py'), posZ: $('hud-pz'),
        velX: $('hud-vx'), velY: $('hud-vy'), velZ: $('hud-vz'),
        accX: $('hud-ax'), accY: $('hud-ay'), accZ: $('hud-az'),
        roll: $('hud-roll'), pitch: $('hud-pitch'), yaw: $('hud-yaw'),
        omgX: $('hud-wx'), omgY: $('hud-wy'), omgZ: $('hud-wz'),
        mT: [$('hud-m1'), $('hud-m2'), $('hud-m3'), $('hud-m4')],
        mBar: [$('bar-m1'), $('bar-m2'), $('bar-m3'), $('bar-m4')],
        alt: $('hud-alt'), speed: $('hud-speed'),
        fps: $('hud-fps'), steps: $('hud-steps'),
        armed: $('armed-status'),
        totalT: $('hud-total-t'),
    };

    let fpsFrames = 0, fpsLast = performance.now(), stepCount = 0;

    function updateHUD() {
        const s = drone.state;
        const r2d = v => (v * 180 / Math.PI).toFixed(1);
        const f2 = v => (typeof v === 'number' ? v.toFixed(2) : '—');
        const f3 = v => (typeof v === 'number' ? v.toFixed(3) : '—');

        hud.posX.textContent = f2(s.position.x);
        hud.posY.textContent = f2(s.position.y);
        hud.posZ.textContent = f2(s.position.z);

        hud.velX.textContent = f2(s.velocity.x);
        hud.velY.textContent = f2(s.velocity.y);
        hud.velZ.textContent = f2(s.velocity.z);

        hud.accX.textContent = f2(s.acceleration.x);
        hud.accY.textContent = f2(s.acceleration.y);
        hud.accZ.textContent = f2(s.acceleration.z);

        hud.roll.textContent = r2d(s.orientation.roll) + '°';
        hud.pitch.textContent = r2d(s.orientation.pitch) + '°';
        hud.yaw.textContent = r2d(s.orientation.yaw) + '°';

        hud.omgX.textContent = f3(s.angularVelocity.x);
        hud.omgY.textContent = f3(s.angularVelocity.y);
        hud.omgZ.textContent = f3(s.angularVelocity.z);

        hud.alt.textContent = s.altitude.toFixed(2) + ' m';
        hud.speed.textContent = (s.speed * 3.6).toFixed(1) + ' km/h';

        hud.totalT.textContent = s.totalThrust.toFixed(1) + ' N';

        s.thrusts.forEach((T, i) => {
            const pct = T / DroneModel.SPECS.maxThrust * 100;
            hud.mT[i].textContent = T.toFixed(1) + ' N';
            hud.mBar[i].style.width = pct.toFixed(0) + '%';
            const hue = 120 - pct * 1.2;
            hud.mBar[i].style.background = `hsl(${hue},80%,50%)`;
        });

        // Attitude indicator (artificial horizon)
        updateAH(s.orientation.roll, s.orientation.pitch);

        // FPS
        fpsFrames++;
        const now = performance.now();
        if (now - fpsLast >= 500) {
            hud.fps.textContent = Math.round(fpsFrames / ((now - fpsLast) / 1000));
            hud.steps.textContent = stepCount;
            fpsFrames = 0; fpsLast = now;
        }
    }

    function updateArmStatus() {
        hud.armed.textContent = drone.isArmed ? '🟢 ARMED' : '🔴 DISARMED';
        hud.armed.style.color = drone.isArmed ? '#34d399' : '#f87171';
    }
    updateArmStatus();

    // ── Artificial Horizon ───────────────────────────────────────────────────
    let ahCanvas, ahCtx;
    function initAH() {
        ahCanvas = $('ah-canvas');
        if (!ahCanvas) return;
        ahCtx = ahCanvas.getContext('2d');
        ahCanvas.width = 120;
        ahCanvas.height = 120;
    }

    function updateAH(roll, pitch) {
        if (!ahCtx) return;
        const w = ahCanvas.width, h = ahCanvas.height, cx = w / 2, cy = h / 2, r = 54;

        ahCtx.clearRect(0, 0, w, h);

        // Clip to circle
        ahCtx.save();
        ahCtx.beginPath();
        ahCtx.arc(cx, cy, r, 0, Math.PI * 2);
        ahCtx.clip();

        // Rotate canvas by roll
        ahCtx.translate(cx, cy);
        ahCtx.rotate(roll);
        ahCtx.translate(-cx, -cy);

        // Pitch offset (pixels per radian — 20px/rad)
        const po = pitch * 40;

        // Sky
        ahCtx.fillStyle = '#1a4a8a';
        ahCtx.fillRect(0, 0, w, cy + po);

        // Ground
        ahCtx.fillStyle = '#5c3a1e';
        ahCtx.fillRect(0, cy + po, w, h - (cy + po));

        // Horizon line
        ahCtx.strokeStyle = '#ffffff';
        ahCtx.lineWidth = 2;
        ahCtx.beginPath();
        ahCtx.moveTo(0, cy + po); ahCtx.lineTo(w, cy + po);
        ahCtx.stroke();

        // Pitch lines
        ahCtx.strokeStyle = 'rgba(255,255,255,0.6)';
        ahCtx.lineWidth = 1;
        ahCtx.font = '8px monospace';
        ahCtx.fillStyle = '#ffffff';
        for (let deg = -30; deg <= 30; deg += 10) {
            if (deg === 0) continue;
            const py = cy + po - deg * (40 / 30);
            const hw = deg % 20 === 0 ? 22 : 14;
            ahCtx.beginPath();
            ahCtx.moveTo(cx - hw, py); ahCtx.lineTo(cx + hw, py);
            ahCtx.stroke();
            ahCtx.fillText(deg, cx + hw + 2, py + 3);
        }

        ahCtx.restore();

        // Fixed aircraft symbol
        ahCtx.strokeStyle = '#f59e0b';
        ahCtx.lineWidth = 2.5;
        ahCtx.beginPath();
        ahCtx.moveTo(cx - 20, cy); ahCtx.lineTo(cx - 5, cy);
        ahCtx.moveTo(cx + 5, cy); ahCtx.lineTo(cx + 20, cy);
        ahCtx.stroke();
        ahCtx.fillStyle = '#f59e0b';
        ahCtx.beginPath();
        ahCtx.arc(cx, cy, 3, 0, Math.PI * 2);
        ahCtx.fill();

        // Roll indicator ring
        ahCtx.strokeStyle = 'rgba(255,255,255,0.3)';
        ahCtx.lineWidth = 1;
        ahCtx.beginPath();
        ahCtx.arc(cx, cy, r, 0, Math.PI * 2);
        ahCtx.stroke();

        // Roll triangle
        ahCtx.save();
        ahCtx.translate(cx, cy);
        ahCtx.rotate(roll);
        ahCtx.fillStyle = '#f59e0b';
        ahCtx.beginPath();
        ahCtx.moveTo(0, -r + 2); ahCtx.lineTo(-4, -r + 10); ahCtx.lineTo(4, -r + 10);
        ahCtx.closePath(); ahCtx.fill();
        ahCtx.restore();
    }

    // ── Main loop ─────────────────────────────────────────────────────────────
    const FIXED_STEP = 1 / 60;
    let lastTime = performance.now();
    let accumulator = 0;

    function loop() {
        requestAnimationFrame(loop);

        const now = performance.now();
        const dt = Math.min((now - lastTime) / 1000, 0.05);
        lastTime = now;

        applyControls(dt);

        // Fixed-step physics
        accumulator += dt;
        while (accumulator >= FIXED_STEP) {
            drone.tick(FIXED_STEP);
            world.step(FIXED_STEP);
            stepCount++;
            accumulator -= FIXED_STEP;
        }

        sceneManager.update(dt);
        sceneManager.render();
        updateHUD();
    }

    initAH();
    loop();

    console.log('%c🚁 Drone Simulation Ready', 'color:#38bdf8;font-size:14px;font-weight:bold');
    console.log('%cSPACE=ARM  W/S=Throttle  ↑↓=Pitch  ←→=Roll  Q/E=Yaw  H=HoldAlt  C=Camera  R=Reset', 'color:#94a3b8;font-size:11px');
    console.log('%cAPI: window.DRONE.state | setThrusts(t0,t1,t2,t3) | setMotorThrust(idx,N) | arm() | disarm() | reset()', 'color:#818cf8;font-size:11px');
})();
