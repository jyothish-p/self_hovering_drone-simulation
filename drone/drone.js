/**
 * drone.js — DroneModel
 * ─────────────────────────────────────────────────────────────────────────────
 * Commercial X-frame quadcopter physics model.
 *
 * Specs:
 *   Frame:           X-frame, 450 mm motor-to-motor diagonal
 *   Arm length:      225 mm (from center to motor)
 *   Body:            180 × 180 × 70 mm
 *   Total mass:      1.4 kg
 *   Central body:    0.80 kg
 *   Each motor:      0.12 kg  (×4 = 0.48 kg)
 *   Battery:         0.25 kg  (mounted below body)
 *   Prop diameter:   10 inch (0.254 m)
 *   Max thrust/motor:15 N
 *
 * Motor positions (body frame, meters):
 *   M1: (+0.225,  0, +0.225)  — front-right, CW  prop
 *   M2: (-0.225,  0, +0.225)  — front-left,  CCW prop
 *   M3: (-0.225,  0, -0.225)  — rear-left,   CW  prop
 *   M4: (+0.225,  0, -0.225)  — rear-right,  CCW prop
 *
 * Inertia tensor (X-frame, computed from mass distribution):
 *   Ixx = Izz ≈ 0.0277 kg·m²  (roll / pitch axes)
 *   Iyy       ≈ 0.0522 kg·m²  (yaw axis)
 *
 * Physics:
 *   - Cannon.js rigid body, custom inertia tensor
 *   - Per-motor thrust applied at motor world position
 *   - Reaction torque (yaw) per motor: τ = ±κ·F, κ = 0.016 N·m/N
 *   - Aerodynamic drag: F = −½·ρ·Cd·A·v²·v̂
 *   - Gravity handled by Cannon.js world
 *   - Ground collision with soft recovery
 *
 * State exposed (window.DRONE):
 *   position, velocity, acceleration, orientation (RPY), angularVelocity,
 *   thrusts[], rpm[], totalThrust, totalTorque, isArmed, isGrounded
 * ─────────────────────────────────────────────────────────────────────────────
 */

class DroneModel {
    // ── Specs ──────────────────────────────────────────────────────────────────
    static SPECS = {
        totalMass: 1.4,
        bodyMass: 0.80,
        motorMass: 0.12,
        batteryMass: 0.25,
        bodyDims: { x: 0.180, y: 0.070, z: 0.180 },
        armLength: 0.225,        // metres — center to motor
        motorDiag: 0.450,        // diagonal motor-to-motor
        propDiameter: 0.254,        // 10 inch
        maxThrust: 15.0,         // N per motor
        kappa: 0.016,        // yaw torque coefficient N·m/N
        // Inertia tensor components (kg·m²)
        Ixx: 0.0277,
        Iyy: 0.0522,
        Izz: 0.0277,
        // Aerodynamics
        airDensity: 1.225,
        dragCoeff: 0.3,
        crossSection: 0.03,         // m² (frontal area of body)
        // Motor physics
        motorTimeConst: 0.05,       // seconds (motor lag τ)
    };

    // Motor positions in body frame [forward = +z, right = +x, up = +y]
    static MOTOR_POS = [
        new CANNON.Vec3(+0.225, 0, +0.225),  // M1 front-right  CW
        new CANNON.Vec3(-0.225, 0, +0.225),  // M2 front-left   CCW
        new CANNON.Vec3(-0.225, 0, -0.225),  // M3 rear-left    CW
        new CANNON.Vec3(+0.225, 0, -0.225),  // M4 rear-right   CCW
    ];

    // Reaction torque sign: +1 = CCW (positive yaw), -1 = CW
    static YAW_SIGN = [-1, +1, -1, +1];

    // ────────────────────────────────────────────────────────────────────────────
    constructor(world, startPos = { x: 0, y: 1.5, z: 0 }) {
        this._world = world;
        this._prevVel = new CANNON.Vec3();
        this._prevTime = null;

        // Target thrusts (from control input), actual thrusts (motor lag)
        this.targetThrusts = [0, 0, 0, 0];
        this.thrusts = [0, 0, 0, 0];

        this.isArmed = false;
        this.isGrounded = true;

        // Derived state (updated each tick)
        this.state = {
            position: { x: 0, y: 0, z: 0 },
            velocity: { x: 0, y: 0, z: 0 },
            acceleration: { x: 0, y: 0, z: 0 },
            orientation: { roll: 0, pitch: 0, yaw: 0 },   // radians
            angularVelocity: { x: 0, y: 0, z: 0 },
            thrusts: [0, 0, 0, 0],
            totalThrust: 0,
            altitude: 0,
            speed: 0,
        };

        this._buildBody(startPos);
    }

    // ── Physics body ───────────────────────────────────────────────────────────
    _buildBody(pos) {
        const s = DroneModel.SPECS;
        const body = new CANNON.Body({ mass: s.totalMass });

        // Represent the frame as a flat box (visual detail handled in Three.js)
        const hw = s.bodyDims.x / 2;
        const hh = s.bodyDims.y / 2;
        const hd = s.bodyDims.z / 2;
        body.addShape(new CANNON.Box(new CANNON.Vec3(hw, hh, hd)));

        body.position.set(pos.x, pos.y, pos.z);
        body.linearDamping = 0.02;
        body.angularDamping = 0.05;
        body.allowSleep = false;    // drones never sleep

        // Apply mass properties first (needed before overriding inertia)
        body.updateMassProperties();

        // Override with physically-correct inertia tensor
        body.inertia.set(s.Ixx, s.Iyy, s.Izz);
        body.invInertia.set(1 / s.Ixx, 1 / s.Iyy, 1 / s.Izz);

        this._world.addBody(body);
        this._body = body;
    }

    // ── Control API ────────────────────────────────────────────────────────────

    /** Set thrust for a single motor [0–15 N]. */
    setMotorThrust(motorIdx, newtons) {
        this.targetThrusts[motorIdx] =
            Math.max(0, Math.min(DroneModel.SPECS.maxThrust, newtons));
    }

    /** Set all four motor thrusts at once. */
    setThrusts(t0, t1, t2, t3) {
        this.targetThrusts[0] = Math.max(0, Math.min(DroneModel.SPECS.maxThrust, t0));
        this.targetThrusts[1] = Math.max(0, Math.min(DroneModel.SPECS.maxThrust, t1));
        this.targetThrusts[2] = Math.max(0, Math.min(DroneModel.SPECS.maxThrust, t2));
        this.targetThrusts[3] = Math.max(0, Math.min(DroneModel.SPECS.maxThrust, t3));
    }

    arm() { this.isArmed = true; }
    disarm() {
        this.isArmed = false;
        this.targetThrusts = [0, 0, 0, 0];
        this.thrusts = [0, 0, 0, 0];
    }

    // ── Physics tick ───────────────────────────────────────────────────────────
    tick(dt) {
        this._updateMotorLag(dt);
        if (this.isArmed) {
            this._applyThrustsAndTorques();
        }
        this._applyAeroDrag();
        this._groundCheck();
        this._updateState(dt);
    }

    /** First-order lag: thrusts[i] → targetThrusts[i], τ = 0.05 s. */
    _updateMotorLag(dt) {
        const tau = DroneModel.SPECS.motorTimeConst;
        const alpha = Math.exp(-dt / tau);
        for (let i = 0; i < 4; i++) {
            this.thrusts[i] = alpha * this.thrusts[i] + (1 - alpha) * this.targetThrusts[i];
        }
    }

    _applyThrustsAndTorques() {
        const body = this._body;
        const kappa = DroneModel.SPECS.kappa;
        const specs = DroneModel.SPECS;

        // Body-up unit vector in world frame
        const localUp = new CANNON.Vec3(0, 1, 0);
        const worldUp = new CANNON.Vec3();
        body.vectorToWorldFrame(localUp, worldUp);

        let totalThrust = 0;

        for (let i = 0; i < 4; i++) {
            const T = this.thrusts[i];
            if (T < 0.0001) continue;
            totalThrust += T;

            // World-space position of this motor
            const worldMotorPos = new CANNON.Vec3();
            body.pointToWorldFrame(DroneModel.MOTOR_POS[i], worldMotorPos);

            // Thrust force upward in world frame
            const force = new CANNON.Vec3(
                worldUp.x * T, worldUp.y * T, worldUp.z * T
            );
            body.applyForce(force, worldMotorPos);

            // Reaction (yaw) torque about body-up axis
            const tauMag = DroneModel.YAW_SIGN[i] * kappa * T;
            const torque = new CANNON.Vec3(
                worldUp.x * tauMag, worldUp.y * tauMag, worldUp.z * tauMag
            );
            body.torque.vadd(torque, body.torque);
        }

        this.state.totalThrust = totalThrust;
    }

    _applyAeroDrag() {
        const s = DroneModel.SPECS;
        const v = this._body.velocity;
        const speed = Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
        if (speed < 0.001) return;
        const dragMag = 0.5 * s.airDensity * s.dragCoeff * s.crossSection * speed * speed;
        const inv = -dragMag / speed;
        this._body.applyForce(
            new CANNON.Vec3(v.x * inv, v.y * inv, v.z * inv),
            this._body.position
        );
    }

    _groundCheck() {
        this.isGrounded = this._body.position.y <=
            (DroneModel.SPECS.bodyDims.y / 2 + 0.15 + 0.05);   // half-height + landing gear height
    }

    // ── State extraction ───────────────────────────────────────────────────────
    _updateState(dt) {
        const b = this._body;
        const p = b.position;
        const v = b.velocity;
        const w = b.angularVelocity;
        const q = b.quaternion;
        const s = this.state;

        // acceleration = Δv / Δt
        const ax = (v.x - this._prevVel.x) / dt;
        const ay = (v.y - this._prevVel.y) / dt;
        const az = (v.z - this._prevVel.z) / dt;
        this._prevVel.set(v.x, v.y, v.z);

        // Euler angles from quaternion (ZYX convention → roll, pitch, yaw)
        const { roll, pitch, yaw } = this._quatToEuler(q);

        s.position = { x: p.x, y: p.y, z: p.z };
        s.velocity = { x: v.x, y: v.y, z: v.z };
        s.acceleration = { x: ax, y: ay, z: az };
        s.orientation = { roll, pitch, yaw };
        s.angularVelocity = { x: w.x, y: w.y, z: w.z };
        s.thrusts = [...this.thrusts];
        s.altitude = Math.max(0, p.y - DroneModel.SPECS.bodyDims.y / 2);
        s.speed = Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    }

    /** Convert quaternion to Euler angles (ZYX / aerospace convention). */
    _quatToEuler(q) {
        const { x, y, z, w } = q;
        // Roll (x-axis rotation)
        const sinr_cosp = 2 * (w * x + y * z);
        const cosr_cosp = 1 - 2 * (x * x + y * y);
        const roll = Math.atan2(sinr_cosp, cosr_cosp);
        // Pitch (y-axis rotation)  ← using z as up-substitute; body z = body forward
        const sinp = 2 * (w * y - z * x);
        const pitch = Math.abs(sinp) >= 1 ? Math.sign(sinp) * Math.PI / 2 : Math.asin(sinp);
        // Yaw (z-axis rotation)
        const siny_cosp = 2 * (w * z + x * y);
        const cosy_cosp = 1 - 2 * (y * y + z * z);
        const yaw = Math.atan2(siny_cosp, cosy_cosp);
        return { roll, pitch, yaw };
    }

    // ── Accessors ──────────────────────────────────────────────────────────────
    get body() { return this._body; }
    get position() { return this._body.position; }
    get quaternion() { return this._body.quaternion; }

    /** Hover thrust — the thrust per motor needed to counteract gravity. */
    get hoverThrust() {
        return (DroneModel.SPECS.totalMass * 9.81) / 4;
    }

    /** Motor RPM estimate from thrust (k_T · n² → T, use empirical). */
    getMotorRPM(motorIdx) {
        const T = this.thrusts[motorIdx];
        // RPM ∝ sqrt(T) — empirical for 10" props: ~4000 RPM at 3.43 N hover
        return Math.sqrt(Math.max(0, T) / 3.43) * 4000;
    }

    /** Reset drone to spawn position. */
    reset(pos = { x: 0, y: 1.5, z: 0 }) {
        this._body.position.set(pos.x, pos.y, pos.z);
        this._body.velocity.setZero();
        this._body.angularVelocity.setZero();
        this._body.quaternion.set(0, 0, 0, 1);
        this._prevVel.setZero();
        this.disarm();
    }
}
