/**
 * drone-scene.js — DroneScene
 * ─────────────────────────────────────────────────────────────────────────────
 * Three.js renderer for the quadcopter simulation.
 *
 * Drone mesh assembly (all units in metres):
 *   Central body:   180×70×180 mm dark-carbon box
 *   4 Arms:         225 mm diagonal carbon tubes to each motor
 *   4 Motor bells:  Cylinder + cap at each arm end
 *   4 Prop discs:   Semi-transparent animated spinning discs (10")
 *   Landing gear:   4 legs below body corners
 *   LED indicators: Red = front-right, Green = front-left
 *   Camera mount:   Small pod at front of body
 *   Battery:        Visible grey slab under body
 *
 * Camera modes: follow (chase), free orbit, top-down
 * ─────────────────────────────────────────────────────────────────────────────
 */

class DroneScene {
    constructor(droneModel) {
        this._drone = droneModel;
        this._propAngle = [0, 0, 0, 0];   // current prop rotation angles

        this._initRenderer();
        this._initScene();
        this._initCamera();
        this._buildDroneMesh();
        this._buildEnvironment();
        this._initOrbitControls();

        this.cameraMode = 'follow';  // 'follow' | 'orbit' | 'top'
        window.addEventListener('resize', () => this._onResize());
    }

    // ── Renderer ────────────────────────────────────────────────────────────────
    _initRenderer() {
        this._renderer = new THREE.WebGLRenderer({ antialias: true });
        this._renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this._renderer.setSize(window.innerWidth, window.innerHeight);
        this._renderer.shadowMap.enabled = true;
        this._renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this._renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this._renderer.toneMappingExposure = 1.15;
        document.getElementById('canvas-wrap').appendChild(this._renderer.domElement);
    }

    // ── Scene ───────────────────────────────────────────────────────────────────
    _initScene() {
        this._scene = new THREE.Scene();
        this._scene.fog = new THREE.FogExp2(0x0d1f35, 0.004);

        // Sky gradient
        const skyGeo = new THREE.SphereGeometry(900, 16, 16);
        const skyMat = new THREE.ShaderMaterial({
            side: THREE.BackSide,
            uniforms: {
                top: { value: new THREE.Color(0x05111f) },
                bot: { value: new THREE.Color(0x0d3555) },
            },
            vertexShader: `varying vec3 vP; void main(){vP=position;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}`,
            fragmentShader: `uniform vec3 top,bot;varying vec3 vP;void main(){float h=normalize(vP).y*.5+.5;gl_FragColor=vec4(mix(bot,top,h),1.);}`,
        });
        this._scene.add(new THREE.Mesh(skyGeo, skyMat));

        // Lighting
        this._scene.add(new THREE.AmbientLight(0x8ba8d0, 0.6));

        const sun = new THREE.DirectionalLight(0xfff5e0, 1.8);
        sun.position.set(-30, 80, 40);
        sun.castShadow = true;
        sun.shadow.mapSize.set(2048, 2048);
        sun.shadow.camera.left = sun.shadow.camera.bottom = -60;
        sun.shadow.camera.right = sun.shadow.camera.top = 60;
        sun.shadow.camera.near = 1;
        sun.shadow.camera.far = 300;
        this._scene.add(sun);
        this._sun = sun;

        this._scene.add(new THREE.HemisphereLight(0x6ab4e8, 0x2a3a2a, 0.35));
    }

    // ── Camera ──────────────────────────────────────────────────────────────────
    _initCamera() {
        this._camera = new THREE.PerspectiveCamera(
            60, window.innerWidth / window.innerHeight, 0.01, 2000
        );
        this._camera.position.set(0, 3, 5);
        this._camTarget = new THREE.Vector3();
    }

    _onResize() {
        this._camera.aspect = window.innerWidth / window.innerHeight;
        this._camera.updateProjectionMatrix();
        this._renderer.setSize(window.innerWidth, window.innerHeight);
    }

    // ── Drone Mesh ──────────────────────────────────────────────────────────────
    _buildDroneMesh() {
        this._droneRoot = new THREE.Group();
        this._scene.add(this._droneRoot);

        this._buildBody();
        this._buildArms();
        this._buildMotors();
        this._buildProps();
        this._buildLandingGear();
        this._buildLEDs();
        this._buildCamera();
        this._buildBattery();
        this._buildPropThrustLine();
    }

    // Materials
    _carbonMat() {
        return new THREE.MeshStandardMaterial({
            color: 0x1a1a1a, roughness: 0.7, metalness: 0.2,
        });
    }
    _motorMat() {
        return new THREE.MeshStandardMaterial({
            color: 0x111828, roughness: 0.3, metalness: 0.85,
        });
    }
    _propMat(tint = 0x222222) {
        return new THREE.MeshStandardMaterial({
            color: tint, transparent: true, opacity: 0.85,
            roughness: 0.3, metalness: 0.1, side: THREE.DoubleSide,
        });
    }

    _buildBody() {
        // Main body frame (carbon plate)
        const bGeo = new THREE.BoxGeometry(0.180, 0.030, 0.180);
        const body = new THREE.Mesh(bGeo, this._carbonMat());
        body.position.y = 0;
        body.castShadow = true;
        this._droneRoot.add(body);

        // Top plate (slightly smaller, with cutouts – approximated as thinner box)
        const tGeo = new THREE.BoxGeometry(0.140, 0.010, 0.140);
        const top = new THREE.Mesh(tGeo, new THREE.MeshStandardMaterial({
            color: 0x222222, roughness: 0.6, metalness: 0.3,
        }));
        top.position.y = 0.022;
        this._droneRoot.add(top);

        // Body midframe (standoffs simulated as 4 small cylinders)
        const standoffGeo = new THREE.CylinderGeometry(0.004, 0.004, 0.04, 8);
        const standoffMat = new THREE.MeshStandardMaterial({
            color: 0x888888, metalness: 0.9, roughness: 0.2,
        });
        const sfPos = [[0.06, 0.06], [-0.06, 0.06], [0.06, -0.06], [-0.06, -0.06]];
        sfPos.forEach(([x, z]) => {
            const sf = new THREE.Mesh(standoffGeo, standoffMat);
            sf.position.set(x, 0.01, z);
            this._droneRoot.add(sf);
        });
    }

    _buildArms() {
        // 4 diagonal arms at 45°, 225mm long
        const armMat = this._carbonMat();

        const motorPositions = [
            [+0.225, 0, +0.225],
            [-0.225, 0, +0.225],
            [-0.225, 0, -0.225],
            [+0.225, 0, -0.225],
        ];

        // Arm tube geo - oriented along the axis from center to motor
        const length = Math.sqrt(0.225 * 0.225 + 0.225 * 0.225);  // ≈0.318 m
        const armGeo = new THREE.CylinderGeometry(0.010, 0.010, length, 8);

        motorPositions.forEach(([mx, , mz]) => {
            const arm = new THREE.Mesh(armGeo, armMat.clone());
            arm.castShadow = true;
            this._droneRoot.add(arm);

            // Point from center to motor position: normalize and rotate
            arm.position.set(mx / 2, 0, mz / 2);     // midpoint of arm
            // angle in xz plane (arms are at 45°, 135°, 225°, 315°)
            const angle = Math.atan2(mx, mz);
            arm.rotation.set(Math.PI / 2, 0, 0);   // lay horizontal
            arm.rotateOnWorldAxis(new THREE.Vector3(0, 1, 0), -angle);
        });
    }

    _buildMotors() {
        this._motorMeshes = [];

        const motorPositions = [
            [+0.225, 0, +0.225],
            [-0.225, 0, +0.225],
            [-0.225, 0, -0.225],
            [+0.225, 0, -0.225],
        ];

        motorPositions.forEach((pos, i) => {
            const group = new THREE.Group();
            group.position.set(...pos);
            this._droneRoot.add(group);

            // Motor stator (dark cylinder)
            const stator = new THREE.Mesh(
                new THREE.CylinderGeometry(0.023, 0.020, 0.020, 24),
                this._motorMat()
            );
            stator.castShadow = true;
            group.add(stator);

            // Motor bell (rotating part - same group, spins with prop visually)
            const bell = new THREE.Mesh(
                new THREE.CylinderGeometry(0.018, 0.022, 0.018, 24),
                new THREE.MeshStandardMaterial({
                    color: 0x2a2a3a, roughness: 0.2, metalness: 0.9,
                })
            );
            bell.position.y = 0.019;
            group.add(bell);

            // Motor shaft
            const shaft = new THREE.Mesh(
                new THREE.CylinderGeometry(0.003, 0.003, 0.025, 8),
                new THREE.MeshStandardMaterial({ color: 0xcccccc, metalness: 1, roughness: 0.1 })
            );
            shaft.position.y = 0.030;
            group.add(shaft);

            this._motorMeshes.push(group);
        });
    }

    _buildProps() {
        this._propGroups = [];

        const motorPositions = [
            [+0.225, 0, +0.225],
            [-0.225, 0, +0.225],
            [-0.225, 0, -0.225],
            [+0.225, 0, -0.225],
        ];

        const propColors = [0x1a1a1a, 0x303030, 0x1a1a1a, 0x303030];

        motorPositions.forEach((pos, i) => {
            const group = new THREE.Group();
            group.position.set(pos[0], 0.045, pos[2]);
            this._droneRoot.add(group);
            this._propGroups.push(group);

            // 2 propeller blades (opposing pair)
            const bladeLen = 0.127;    // 10 inch → 254mm diameter → 127mm radius
            const bladeW = 0.022;    // chord width
            const bladeT = 0.004;    // thickness

            for (let b = 0; b < 2; b++) {
                const bladeGeo = new THREE.BoxGeometry(bladeLen, bladeT, bladeW);
                // Taper the blade (simple approach: scale one end)
                const blade = new THREE.Mesh(bladeGeo, this._propMat(propColors[i]));
                blade.rotation.y = b * Math.PI;   // 180° apart
                blade.position.x = b === 0 ? bladeLen / 2 : -bladeLen / 2;
                group.add(blade);

                // Blade twist: small pitch angle
                blade.rotation.z = (b === 0 ? 1 : -1) * 0.15;
            }

            // Hub
            const hub = new THREE.Mesh(
                new THREE.CylinderGeometry(0.010, 0.010, 0.008, 16),
                new THREE.MeshStandardMaterial({ color: propColors[i], metalness: 0.3, roughness: 0.5 })
            );
            group.add(hub);

            // Spinning blur disc (visible when rotating fast)
            const discGeo = new THREE.CircleGeometry(0.127, 32);
            const discMat = new THREE.MeshBasicMaterial({
                color: propColors[i] === 0x1a1a1a ? 0x333333 : 0x444444,
                transparent: true, opacity: 0, side: THREE.DoubleSide,
                depthWrite: false,
            });
            const disc = new THREE.Mesh(discGeo, discMat);
            disc.rotation.x = -Math.PI / 2;
            group.add(disc);
            group.userData.blurDisc = disc;
        });
    }

    _buildLandingGear() {
        const legMat = new THREE.MeshStandardMaterial({
            color: 0x111111, roughness: 0.8,
        });

        // 4 legs (thin cylinders slanting outward below body corners)
        const legPos = [[0.07, -0.07], [- 0.07, -0.07], [0.07, 0.07], [-0.07, 0.07]];
        legPos.forEach(([lx, lz]) => {
            const legGeo = new THREE.CylinderGeometry(0.005, 0.005, 0.10, 8);
            const leg = new THREE.Mesh(legGeo, legMat);
            leg.position.set(lx, -0.065, lz);
            leg.rotation.z = (lx > 0 ? 1 : -1) * 0.22;
            leg.rotation.x = (lz > 0 ? 1 : -1) * 0.22;
            this._droneRoot.add(leg);

            // Foot skid
            const skidGeo = new THREE.CylinderGeometry(0.005, 0.005, 0.12, 8);
            const skid = new THREE.Mesh(skidGeo, legMat);
            skid.position.set(lx * 1.4, -0.115, 0);
            skid.rotation.z = Math.PI / 2;
            this._droneRoot.add(skid);
        });
    }

    _buildLEDs() {
        // Front-right: red | Front-left: green
        const ledData = [
            { pos: [+0.200, 0.018, +0.200], color: 0xff2222 },
            { pos: [-0.200, 0.018, +0.200], color: 0x00ff44 },
        ];
        ledData.forEach(({ pos, color }) => {
            const geo = new THREE.SphereGeometry(0.005, 8, 8);
            const mat = new THREE.MeshBasicMaterial({ color });
            const led = new THREE.Mesh(geo, mat);
            led.position.set(...pos);
            this._droneRoot.add(led);

            const light = new THREE.PointLight(color, 0.6, 0.3);
            light.position.set(...pos);
            this._droneRoot.add(light);
        });

        // Rear indicator: blue (solid)
        const rear = new THREE.Mesh(
            new THREE.SphereGeometry(0.005, 8, 8),
            new THREE.MeshBasicMaterial({ color: 0x4444ff })
        );
        rear.position.set(0, 0.018, -0.10);
        this._droneRoot.add(rear);
    }

    _buildCamera() {
        // FPV/gimbal camera at front
        const camGeo = new THREE.BoxGeometry(0.025, 0.020, 0.020);
        const camMat = new THREE.MeshStandardMaterial({
            color: 0x1a1a2a, roughness: 0.2, metalness: 0.7,
        });
        const cam = new THREE.Mesh(camGeo, camMat);
        cam.position.set(0, 0, 0.10);
        this._droneRoot.add(cam);

        // Lens
        const lens = new THREE.Mesh(
            new THREE.CylinderGeometry(0.006, 0.006, 0.006, 16),
            new THREE.MeshStandardMaterial({ color: 0x111111, metalness: 0.9, roughness: 0.05 })
        );
        lens.rotation.x = Math.PI / 2;
        lens.position.set(0, 0, 0.113);
        this._droneRoot.add(lens);
    }

    _buildBattery() {
        const batGeo = new THREE.BoxGeometry(0.100, 0.025, 0.055);
        const batMat = new THREE.MeshStandardMaterial({
            color: 0x1a2a44, roughness: 0.5, metalness: 0.15,
        });
        const bat = new THREE.Mesh(batGeo, batMat);
        bat.position.set(0, -0.028, 0);
        bat.castShadow = true;
        this._droneRoot.add(bat);
    }

    /** Invisible thrust lines for force visualization */
    _buildPropThrustLine() {
        this._thrustIndicators = [];
        const motorPositions = [
            [+0.225, 0.05, +0.225],
            [-0.225, 0.05, +0.225],
            [-0.225, 0.05, -0.225],
            [+0.225, 0.05, -0.225],
        ];
        motorPositions.forEach(pos => {
            const geo = new THREE.CylinderGeometry(0.003, 0.008, 0.1, 8);
            const mat = new THREE.MeshBasicMaterial({
                color: 0x00d4ff, transparent: true, opacity: 0.5,
            });
            const arrow = new THREE.Mesh(geo, mat);
            arrow.position.set(...pos);
            this._droneRoot.add(arrow);
            this._thrustIndicators.push(arrow);
        });
    }

    // ── Environment ─────────────────────────────────────────────────────────────
    _buildEnvironment() {
        // Ground
        const gGeo = new THREE.PlaneGeometry(500, 500, 80, 80);
        const gMat = new THREE.MeshStandardMaterial({
            color: 0x1a2a14, roughness: 0.97, metalness: 0.0,
        });
        const ground = new THREE.Mesh(gGeo, gMat);
        ground.rotation.x = -Math.PI / 2;
        ground.receiveShadow = true;
        this._scene.add(ground);

        // Tarmac pad (landing zone)
        const padGeo = new THREE.CircleGeometry(3, 32);
        const padMat = new THREE.MeshStandardMaterial({ color: 0x1a1a1a, roughness: 0.9 });
        const pad = new THREE.Mesh(padGeo, padMat);
        pad.rotation.x = -Math.PI / 2;
        pad.position.y = 0.002;
        this._scene.add(pad);

        // Landing H marker
        const hMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
        [[0.5, 0], [-0.5, 0], [0, 0]].forEach(([x, z]) => {
            const barGeo = new THREE.PlaneGeometry(x === 0 ? 0.1 : 0.1, x === 0 ? 0.8 : 0.2);
            const bar = new THREE.Mesh(barGeo, hMat);
            bar.rotation.x = -Math.PI / 2;
            bar.position.set(x, 0.004, z);
            this._scene.add(bar);
        });

        // Grid
        const grid = new THREE.GridHelper(200, 40, 0x1a3c5a, 0x0e2535);
        grid.position.y = 0.01;
        this._scene.add(grid);

        // Reference markers (every 10m)
        for (let r = 10; r <= 50; r += 10) {
            const rGeo = new THREE.RingGeometry(r - 0.05, r + 0.05, 64);
            const rMat = new THREE.MeshBasicMaterial({
                color: 0x38bdf8, transparent: true, opacity: 0.15, side: THREE.DoubleSide,
            });
            const ring = new THREE.Mesh(rGeo, rMat);
            ring.rotation.x = -Math.PI / 2;
            ring.position.y = 0.01;
            this._scene.add(ring);
        }
    }

    // ── Orbit Controls (inline) ─────────────────────────────────────────────────
    _initOrbitControls() {
        const cam = this._camera;
        const dom = this._renderer.domElement;
        let isDragging = false, isPanning = false;
        let lastX = 0, lastY = 0;
        this._orbitTheta = 0.5;
        this._orbitPhi = 0.5;
        this._orbitRadius = 5;
        this._orbitOffset = new THREE.Vector3(0, 0, 0);

        dom.addEventListener('mousedown', e => {
            if (this.cameraMode !== 'orbit') return;
            if (e.button === 0) isDragging = true;
            if (e.button === 2) isPanning = true;
            lastX = e.clientX; lastY = e.clientY;
        });
        window.addEventListener('mouseup', () => { isDragging = false; isPanning = false; });
        window.addEventListener('mousemove', e => {
            const dx = e.clientX - lastX, dy = e.clientY - lastY;
            lastX = e.clientX; lastY = e.clientY;
            if (isDragging && this.cameraMode === 'orbit') {
                this._orbitTheta -= dx * 0.006;
                this._orbitPhi = Math.max(0.05, Math.min(Math.PI / 2 - 0.01, this._orbitPhi + dy * 0.006));
            }
            if (isPanning && this.cameraMode === 'orbit') {
                this._orbitOffset.y -= dy * 0.02;
                this._orbitOffset.y = Math.max(0, this._orbitOffset.y);
            }
        });
        dom.addEventListener('wheel', e => {
            if (this.cameraMode === 'orbit') {
                this._orbitRadius = Math.max(0.5, Math.min(80, this._orbitRadius + e.deltaY * 0.02));
            }
        }, { passive: true });
        dom.addEventListener('contextmenu', e => e.preventDefault());
    }

    // ── Per-frame Update ────────────────────────────────────────────────────────
    update(dt) {
        this._syncDroneTransform();
        this._animateProps(dt);
        this._updateThrustIndicators();
        this._updateCamera();
    }

    _syncDroneTransform() {
        const p = this._drone.position;
        const q = this._drone.quaternion;
        this._droneRoot.position.set(p.x, p.y, p.z);
        this._droneRoot.quaternion.set(q.x, q.y, q.z, q.w);
    }

    _animateProps(dt) {
        const thrusts = this._drone.thrusts;
        const yawSign = DroneModel.YAW_SIGN;

        this._propGroups.forEach((group, i) => {
            const T = thrusts[i];
            // RPM proportional to sqrt(thrust) — spins faster with more thrust
            const rpm = this._drone.getMotorRPM(i);
            const rps = rpm / 60;
            const dir = yawSign[i];  // +1 = CCW, -1 = CW (looking from above)

            this._propAngle[i] += dir * rps * 2 * Math.PI * dt;
            group.rotation.y = this._propAngle[i];

            // Blur disc opacity increases with RPM
            const blurOpacity = Math.min(0.55, (rpm / 10000) * 0.55);
            if (group.userData.blurDisc) {
                group.userData.blurDisc.material.opacity = blurOpacity;
            }
        });
    }

    _updateThrustIndicators() {
        const thrusts = this._drone.thrusts;
        const maxT = DroneModel.SPECS.maxThrust;
        this._thrustIndicators.forEach((arrow, i) => {
            const frac = thrusts[i] / maxT;
            arrow.scale.y = 0.3 + frac * 1.5;
            arrow.material.opacity = 0.2 + frac * 0.7;
            // Colour: blue (low) → cyan → white (max)
            arrow.material.color.setHSL(0.6 - frac * 0.2, 1.0, 0.4 + frac * 0.3);
        });
    }

    _updateCamera() {
        const dronePos = new THREE.Vector3(
            this._drone.position.x,
            this._drone.position.y,
            this._drone.position.z
        );

        if (this.cameraMode === 'follow') {
            // Chase camera: behind and above, smoothly following orientation
            const q = this._drone.quaternion;
            const threeQ = new THREE.Quaternion(q.x, q.y, q.z, q.w);
            const behind = new THREE.Vector3(0, 1.5, -3.5).applyQuaternion(threeQ);
            const target = dronePos.clone().add(behind);

            this._camera.position.lerp(target, 0.08);
            this._camTarget.lerp(dronePos, 0.1);
            this._camera.lookAt(this._camTarget);

        } else if (this.cameraMode === 'orbit') {
            const ct = dronePos.clone().add(this._orbitOffset);
            this._camera.position.set(
                ct.x + this._orbitRadius * Math.sin(this._orbitPhi) * Math.sin(this._orbitTheta),
                ct.y + this._orbitRadius * Math.cos(this._orbitPhi),
                ct.z + this._orbitRadius * Math.sin(this._orbitPhi) * Math.cos(this._orbitTheta)
            );
            this._camera.lookAt(ct);

        } else if (this.cameraMode === 'top') {
            this._camera.position.set(dronePos.x, dronePos.y + 12, dronePos.z);
            this._camera.lookAt(dronePos);
        }
    }

    render() {
        this._renderer.render(this._scene, this._camera);
    }

    setCameraMode(mode) { this.cameraMode = mode; }
}
