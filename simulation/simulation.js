"use strict";

const MIN_ORBIT_DISTANCE = 0.05;
const TRAIL_SAMPLE_INTERVAL = 0.04;
const MAX_TRAIL_POINTS = 1400;
const STEPS_PER_FRAME = 4;
const FULL_CIRCLE = 2 * Math.PI;

function createSeededRandom(seed) {
    let state = seed >>> 0;
    return function nextRandom() {
        state += 0x6D2B79F5;
        let value = state;
        value = Math.imul(value ^ (value >>> 15), value | 1);
        value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
        return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
    };
}

function createDiskBodies(settings) {
    const random = createSeededRandom(settings.seed);
    return Array.from({ length: settings.count }, (_, bodyIndex) => {
        const distanceRandom = random();
        const angleRandom = random();
        const heightRandom = random();
        const massRandom = random();
        const distanceFromCenter =
            settings.minimumRadius + distanceRandom * (settings.maximumRadius - settings.minimumRadius);
        const angleAroundCenter = angleRandom * FULL_CIRCLE;

        return {
            bodyIndex,
            distanceRandom,
            angleRandom,
            heightRandom,
            distanceFromCenter,
            angleAroundCenter,
            x: distanceFromCenter * Math.cos(angleAroundCenter),
            y: distanceFromCenter * Math.sin(angleAroundCenter),
            z: (heightRandom - 0.5) * settings.thickness,
            mass: 1 + massRandom * 5
        };
    });
}

function calculateOrbitAcceleration(position, parameters) {
    const distance = Math.max(Math.hypot(position.x, position.y, position.z), MIN_ORBIT_DISTANCE);
    const accelerationScale = -(parameters.gravity * parameters.centralMass) / (distance * distance * distance);
    return {
        x: position.x * accelerationScale,
        y: position.y * accelerationScale,
        z: position.z * accelerationScale
    };
}

function advanceOrbit(state, parameters) {
    const timeStep = parameters.timeStep;
    const oldAcceleration = calculateOrbitAcceleration(state.position, parameters);
    const position = {};
    const velocity = {};

    // Velocity-Verlet uses the acceleration at both ends of the step.
    for (const axis of ["x", "y", "z"]) {
        position[axis] = state.position[axis] + (state.velocity[axis] * timeStep
            + 0.5 * oldAcceleration[axis] * timeStep * timeStep);
    }
    const newAcceleration = calculateOrbitAcceleration(position, parameters);
    for (const axis of ["x", "y", "z"]) {
        velocity[axis] = state.velocity[axis]
            + 0.5 * (oldAcceleration[axis] + newAcceleration[axis]) * timeStep;
    }
    if (![...Object.values(position), ...Object.values(velocity), state.elapsedTime + timeStep]
        .every(Number.isFinite)) {
        throw new RangeError("The orbit exceeded the numeric range. Reduce the values and reset the orbit.");
    }

    state.position = position;
    state.velocity = velocity;
    state.elapsedTime += timeStep;
    const previousPoint = state.trail.at(-1);
    if (!previousPoint || state.elapsedTime - previousPoint.time >= TRAIL_SAMPLE_INTERVAL) {
        state.trail.push({ ...position, time: state.elapsedTime });
    }
    if (state.trail.length > MAX_TRAIL_POINTS) {
        state.trail.shift();
    }
}

function createNBodyState(settings) {
    if (!Number.isSafeInteger(settings.seed) || !Number.isInteger(settings.count) || settings.count < 1) {
        throw new RangeError("Use a whole-number seed and a positive whole-number body count.");
    }
    const initialBodies = createDiskBodies({
        seed: settings.seed,
        count: settings.count,
        minimumRadius: 1,
        maximumRadius: 4,
        thickness: 0.5
    });
    const bodies = initialBodies.map((body) => {
        const speed = 0.5 * Math.sqrt(1 / body.distanceFromCenter);
        return {
            bodyIndex: body.bodyIndex,
            x: body.x,
            y: body.y,
            z: body.z,
            mass: body.mass,
            vx: -speed * Math.sin(body.angleAroundCenter),
            vy: speed * Math.cos(body.angleAroundCenter),
            vz: 0
        };
    });
    return {
        settings: { ...settings },
        bodies,
        nextBodies: bodies.map((body) => ({ ...body })),
        elapsedTime: 0,
        steps: 0,
        viewRadius: 5
    };
}

function advanceNBody(state, parameters) {
    const { gravity, softening, timeStep } = parameters;
    if (!Number.isFinite(gravity) || gravity < 0 || !Number.isFinite(softening) || softening <= 0
        || !Number.isFinite(timeStep) || timeStep <= 0) {
        throw new RangeError("Gravity must be nonnegative, and softening and time step must be positive finite numbers.");
    }

    // Every force calculation reads the old positions before either buffer is swapped.
    for (let i = 0; i < state.bodies.length; i++) {
        const body = state.bodies[i];
        let accelerationX = 0;
        let accelerationY = 0;
        let accelerationZ = 0;
        for (const source of state.bodies) {
            const deltaX = source.x - body.x;
            const deltaY = source.y - body.y;
            const deltaZ = source.z - body.z;
            const distanceSquared = deltaX * deltaX + deltaY * deltaY + deltaZ * deltaZ + softening;
            const inverseDistance = 1 / Math.sqrt(distanceSquared);
            const accelerationScale = gravity * source.mass * inverseDistance * inverseDistance * inverseDistance;
            accelerationX += deltaX * accelerationScale;
            accelerationY += deltaY * accelerationScale;
            accelerationZ += deltaZ * accelerationScale;
        }
        const next = state.nextBodies[i];
        next.vx = body.vx + accelerationX * timeStep;
        next.vy = body.vy + accelerationY * timeStep;
        next.vz = body.vz + accelerationZ * timeStep;
        next.x = body.x + next.vx * timeStep;
        next.y = body.y + next.vy * timeStep;
        next.z = body.z + next.vz * timeStep;
        if (![next.x, next.y, next.z, next.vx, next.vy, next.vz,
            Math.hypot(next.x, next.y, next.z) * 1.15, Math.hypot(next.vx, next.vy, next.vz)]
            .every(Number.isFinite)) {
            throw new RangeError("The system exceeded the numeric range. Reduce gravity or time step, then reset.");
        }
    }
    const elapsedTime = state.elapsedTime + timeStep;
    if (!Number.isFinite(elapsedTime)) {
        throw new RangeError("Simulated time exceeded the numeric range. Reset the system.");
    }
    [state.bodies, state.nextBodies] = [state.nextBodies, state.bodies];
    state.elapsedTime = elapsedTime;
    state.steps++;
}

function createAnimationLoop(advance, draw, onError) {
    let running = false;
    let frame = null;

    function stop() {
        running = false;
        cancelAnimationFrame(frame);
        frame = null;
    }

    function animate() {
        if (!running) {
            return;
        }
        try {
            for (let step = 0; step < STEPS_PER_FRAME; step++) {
                advance();
            }
            draw();
        } catch (error) {
            if (!(error instanceof RangeError)) {
                throw error;
            }
            stop();
            onError(error);
            return;
        }
        frame = requestAnimationFrame(animate);
    }

    return {
        get running() {
            return running;
        },
        start() {
            if (!running) {
                running = true;
                frame = requestAnimationFrame(animate);
            }
        },
        stop
    };
}

function projectPoint(point, camera, centerX, centerY, scale) {
    const yawX = point.x * Math.cos(camera.yaw) - point.y * Math.sin(camera.yaw);
    const yawY = point.x * Math.sin(camera.yaw) + point.y * Math.cos(camera.yaw);
    const tiltedY = yawY * Math.cos(camera.pitch) - point.z * Math.sin(camera.pitch);
    return {
        x: centerX + yawX * scale,
        y: centerY - tiltedY * scale,
        depth: yawY * Math.sin(camera.pitch) + point.z * Math.cos(camera.pitch)
    };
}

function prepareCanvas(canvas) {
    const pixelRatio = window.devicePixelRatio || 1;
    const bounds = canvas.getBoundingClientRect();
    const pixelWidth = Math.round(bounds.width * pixelRatio);
    const pixelHeight = Math.round(bounds.height * pixelRatio);
    if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
        canvas.width = pixelWidth;
        canvas.height = pixelHeight;
    }
    const context = canvas.getContext("2d");
    if (!context) {
        throw new Error("A 2D canvas context is needed to draw the simulation.");
    }
    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    context.clearRect(0, 0, bounds.width, bounds.height);
    return { context, width: bounds.width, height: bounds.height };
}

function drawLine(context, start, end, color, lineWidth = 1.5) {
    context.strokeStyle = color;
    context.lineWidth = lineWidth;
    context.beginPath();
    context.moveTo(start.x, start.y);
    context.lineTo(end.x, end.y);
    context.stroke();
}

function drawPoint(context, point, radius, color) {
    context.fillStyle = color;
    context.beginPath();
    context.arc(point.x, point.y, radius, 0, FULL_CIRCLE);
    context.fill();
}

function getBodyColor(bodyIndex) {
    return bodyColors[bodyIndex % bodyColors.length];
}

function drawGrid(context, width, height, verticalAxis) {
    const centerX = width / 2;
    const centerY = height / 2;
    for (let x = centerX % 40; x < width; x += 40) {
        drawLine(context, { x, y: 0 }, { x, y: height }, chartColors.grid, 1);
    }
    for (let y = centerY % 40; y < height; y += 40) {
        drawLine(context, { x: 0, y }, { x: width, y }, chartColors.grid, 1);
    }
    drawLine(context, { x: centerX, y: 0 }, { x: centerX, y: height },
        chartColors[`axis-${verticalAxis}`], 2);
    drawLine(context, { x: 0, y: centerY }, { x: width, y: centerY }, chartColors["axis-x"], 2);
}

function drawAxes(context, camera, width, height, scale, axisLength, zFactor = 1, lineWidth = 2) {
    const origin = projectPoint({ x: 0, y: 0, z: 0 }, camera, width / 2, height / 2, scale);
    const axes = [
        { name: "x", end: { x: axisLength, y: 0, z: 0 } },
        { name: "y", end: { x: 0, y: axisLength, z: 0 } },
        { name: "z", end: { x: 0, y: 0, z: axisLength * zFactor } }
    ];
    context.font = "600 12px Arial, sans-serif";
    for (const axis of axes) {
        const end = projectPoint(axis.end, camera, width / 2, height / 2, scale);
        const color = chartColors[`axis-${axis.name}`];
        drawLine(context, origin, end, color, lineWidth);
        context.fillStyle = color;
        context.fillText(`+${axis.name}`, end.x + 6, end.y);
    }
    return origin;
}

function drawArrow(context, start, end, color) {
    if (Math.hypot(end.x - start.x, end.y - start.y) < 1) {
        return;
    }
    drawLine(context, start, end, color, 2);
    const angle = Math.atan2(end.y - start.y, end.x - start.x);
    context.fillStyle = color;
    context.beginPath();
    context.moveTo(end.x, end.y);
    context.lineTo(end.x - 9 * Math.cos(angle - Math.PI / 6), end.y - 9 * Math.sin(angle - Math.PI / 6));
    context.lineTo(end.x - 9 * Math.cos(angle + Math.PI / 6), end.y - 9 * Math.sin(angle + Math.PI / 6));
    context.closePath();
    context.fill();
}

function bindViewControls(canvas, camera, redraw, selectBody) {
    let activePointer = null;
    let previousPoint;
    let startPoint;
    let suppressClick = false;

    function finishDrag(event) {
        if (event.pointerId !== activePointer) {
            return;
        }
        activePointer = null;
        canvas.classList.remove("is-dragging");
        if (canvas.hasPointerCapture(event.pointerId)) {
            canvas.releasePointerCapture(event.pointerId);
        }
    }

    canvas.addEventListener("wheel", (event) => {
        event.preventDefault();
        camera.zoom = Math.max(0.35, Math.min(8, camera.zoom * Math.exp(-event.deltaY * 0.0015)));
        redraw();
    }, { passive: false });

    if (camera.rotatable) {
        canvas.addEventListener("pointerdown", (event) => {
            if (event.button !== 0 || !event.isPrimary) {
                return;
            }
            activePointer = event.pointerId;
            startPoint = { x: event.clientX, y: event.clientY };
            previousPoint = startPoint;
            suppressClick = false;
            canvas.classList.add("is-dragging");
            canvas.setPointerCapture(event.pointerId);
        });
        canvas.addEventListener("pointermove", (event) => {
            if (event.pointerId !== activePointer) {
                return;
            }
            const deltaX = event.clientX - previousPoint.x;
            const deltaY = event.clientY - previousPoint.y;
            camera.yaw += deltaX * 0.01;
            camera.pitch = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, camera.pitch + deltaY * 0.01));
            suppressClick ||= Math.hypot(event.clientX - startPoint.x, event.clientY - startPoint.y) > 4;
            previousPoint = { x: event.clientX, y: event.clientY };
            redraw();
        });
        canvas.addEventListener("pointerup", finishDrag);
        canvas.addEventListener("pointercancel", (event) => {
            suppressClick = true;
            finishDrag(event);
        });
        canvas.addEventListener("lostpointercapture", finishDrag);
    }
    if (selectBody) {
        canvas.addEventListener("click", (event) => {
            if (!suppressClick) {
                selectBody(event);
            }
            suppressClick = false;
        });
    }
    canvas.addEventListener("keydown", (event) => {
        if (event.key === "+" || event.key === "=" || event.key === "-") {
            event.preventDefault();
            camera.zoom = Math.max(0.35, Math.min(8, camera.zoom * (event.key === "-" ? 0.9 : 1.1)));
            redraw();
        } else if (!selectBody && event.key.startsWith("Arrow")) {
            event.preventDefault();
            if (event.key === "ArrowLeft") {
                camera.yaw -= 0.1;
            }
            if (event.key === "ArrowRight") {
                camera.yaw += 0.1;
            }
            if (event.key === "ArrowUp") {
                camera.pitch += 0.1;
            }
            if (event.key === "ArrowDown") {
                camera.pitch -= 0.1;
            }
            camera.pitch = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, camera.pitch));
            redraw();
        }
    });
}

function readNumberInput(input, label, status) {
    const value = input.valueAsNumber;
    let error = "";
    if (!Number.isFinite(value)) {
        error = `Enter a number for ${label}.`;
    } else if (input.min !== "" && value < Number(input.min)) {
        error = `${label} must be at least ${input.min}.`;
    } else if (input.max !== "" && value > Number(input.max)) {
        error = `${label} must be at most ${input.max}.`;
    }
    input.setAttribute("aria-invalid", String(Boolean(error)));
    if (error) {
        status.textContent = error;
        return null;
    }
    return value;
}

function initializeDisk() {
    const controls = {
        seed: document.querySelector("#seed-value"),
        count: document.querySelector("#body-count"),
        minimumRadius: document.querySelector("#minimum-distance"),
        maximumRadius: document.querySelector("#maximum-distance"),
        thickness: document.querySelector("#thickness")
    };
    const picker = document.querySelector("#selected-body");
    const status = document.querySelector("#disk-status");
    const selectionStatus = document.querySelector("#selection-status");
    const generateButton = document.querySelector("#regenerate");
    const details = {
        title: document.querySelector("#selected-title"),
        x: document.querySelector("#x-equation"),
        y: document.querySelector("#y-equation"),
        z: document.querySelector("#z-equation"),
        summary: document.querySelector("#body-summary")
    };
    const views = [
        { kind: "top", canvas: document.querySelector("#top-view"), camera: { zoom: 1 } },
        { kind: "side", canvas: document.querySelector("#side-view"), camera: { zoom: 1 } },
        {
            kind: "3d",
            canvas: document.querySelector("#three-dimensional-view"),
            camera: { zoom: 1, yaw: -0.65, pitch: 0.55, rotatable: true }
        }
    ];
    let settings;
    let bodies = [];
    let selectedIndex = 0;

    function projectBody(body, view, width, height) {
        const centerX = width / 2;
        const centerY = height / 2;
        if (view.kind === "3d") {
            const scale = Math.min(width, height) * 0.4 / settings.maximumRadius * view.camera.zoom;
            return projectPoint(body, view.camera, centerX, centerY, scale);
        }
        if (view.kind === "top") {
            const scale = Math.min(width, height) * 0.42 / settings.maximumRadius * view.camera.zoom;
            return { x: centerX + body.x * scale, y: centerY - body.y * scale, depth: 0 };
        }
        const horizontalScale = width * 0.42 / settings.maximumRadius * view.camera.zoom;
        const verticalScale = height * 0.34 / (Math.max(settings.thickness, 0.5) / 2) * view.camera.zoom;
        return { x: centerX + body.x * horizontalScale, y: centerY - body.z * verticalScale, depth: 0 };
    }

    function drawView(view) {
        const { context, width, height } = prepareCanvas(view.canvas);
        const center = { x: width / 2, y: height / 2 };
        if (view.kind === "3d") {
            const scale = Math.min(width, height) * 0.4 / settings.maximumRadius * view.camera.zoom;
            drawAxes(context, view.camera, width, height, scale, settings.maximumRadius * 0.7, 0.45);
        } else {
            const verticalAxis = view.kind === "top" ? "y" : "z";
            drawGrid(context, width, height, verticalAxis);
            context.font = "600 12px Arial, sans-serif";
            context.fillStyle = chartColors["axis-x"];
            context.fillText("+x", width - 24, center.y - 8);
            context.fillStyle = chartColors[`axis-${verticalAxis}`];
            context.fillText(`+${verticalAxis}`, center.x + 8, 18);
        }
        const projectedBodies = bodies.map((body, index) => ({
            ...projectBody(body, view, width, height), index
        })).sort((first, second) => first.depth - second.depth);

        for (const point of projectedBodies) {
            if (point.index === selectedIndex) {
                continue;
            }
            if (view.kind === "3d") {
                context.globalAlpha = Math.max(0.8, Math.min(1,
                    0.8 + 0.2 * ((point.depth + settings.maximumRadius) / (2 * settings.maximumRadius))));
            }
            drawPoint(context, point, 2.8, getBodyColor(point.index));
            context.globalAlpha = 1;
        }
        const selected = projectBody(bodies[selectedIndex], view, width, height);
        if (view.kind === "side") {
            drawLine(context, { x: selected.x, y: center.y }, selected, chartColors.height, 2);
        } else {
            context.setLineDash(view.kind === "3d" ? [4, 4] : []);
            drawLine(context, center, selected, chartColors.selected, 1.5);
            context.setLineDash([]);
        }
        drawPoint(context, selected, 8, chartColors.selected);
        drawPoint(context, selected, 5, getBodyColor(selectedIndex));
    }

    function render() {
        for (const view of views) {
            drawView(view);
        }
        const body = bodies[selectedIndex];
        const degrees = body.angleAroundCenter * 180 / Math.PI;
        picker.value = selectedIndex;
        picker.max = bodies.length - 1;
        details.title.textContent = `Selected body #${body.bodyIndex}`;
        details.x.textContent =
            `${body.distanceFromCenter.toFixed(3)} * cos(${degrees.toFixed(1)} deg) = ${body.x.toFixed(3)}`;
        details.y.textContent =
            `${body.distanceFromCenter.toFixed(3)} * sin(${degrees.toFixed(1)} deg) = ${body.y.toFixed(3)}`;
        details.z.textContent =
            `(${body.heightRandom.toFixed(3)} - 0.5) * ${settings.thickness.toFixed(2)} = ${body.z.toFixed(3)}`;
        details.summary.textContent =
            `Radius fraction ${body.distanceRandom.toFixed(3)} gives radius ${body.distanceFromCenter.toFixed(3)}, `
            + `and angle fraction ${body.angleRandom.toFixed(3)} gives ${degrees.toFixed(1)} degrees. `
            + `Position: (${body.x.toFixed(3)}, ${body.y.toFixed(3)}, ${body.z.toFixed(3)}).`;
    }

    function generate() {
        const seed = readNumberInput(controls.seed, "random seed", status);
        if (seed === null) {
            return;
        }
        if (!Number.isSafeInteger(seed)) {
            controls.seed.setAttribute("aria-invalid", "true");
            status.textContent = "The random seed must be a whole number within JavaScript's safe integer range.";
            return;
        }
        status.textContent = "";
        settings = {
            seed,
            count: Number(controls.count.value),
            minimumRadius: Number(controls.minimumRadius.value),
            maximumRadius: Number(controls.maximumRadius.value),
            thickness: Number(controls.thickness.value)
        };
        bodies = createDiskBodies(settings);
        selectedIndex = Math.min(selectedIndex, bodies.length - 1);
        picker.setAttribute("aria-invalid", "false");
        selectionStatus.textContent = "";
        document.querySelector("#seed-value-display").value = seed;
        generateButton.textContent = `Generate with seed ${seed}`;
        render();
    }

    function selectFromPointer(event, view) {
        const bounds = view.canvas.getBoundingClientRect();
        let nearestIndex = selectedIndex;
        let nearestDistance = Infinity;
        for (const body of bodies) {
            const point = projectBody(body, view, bounds.width, bounds.height);
            const distance = Math.hypot(event.clientX - bounds.left - point.x, event.clientY - bounds.top - point.y);
            if (distance < nearestDistance) {
                nearestDistance = distance;
                nearestIndex = body.bodyIndex;
            }
        }
        selectedIndex = nearestIndex;
        picker.setAttribute("aria-invalid", "false");
        selectionStatus.textContent = "";
        render();
    }

    for (const control of Object.values(controls)) {
        if (control === controls.seed) {
            continue;
        }
        control.addEventListener("input", () => {
            document.querySelector(`#${control.id}-value`).value = control === controls.count
                ? control.value : Number(control.value).toFixed(control === controls.thickness ? 2 : 1);
            generate();
        });
    }
    controls.seed.addEventListener("input", () => {
        const seed = controls.seed.valueAsNumber;
        document.querySelector("#seed-value-display").value = Number.isSafeInteger(seed) ? seed : "-";
        generateButton.textContent = Number.isSafeInteger(seed) ? `Generate with seed ${seed}` : "Generate bodies";
    });
    controls.seed.addEventListener("change", generate);
    generateButton.addEventListener("click", generate);
    picker.addEventListener("change", () => {
        const index = readNumberInput(picker, "body index", selectionStatus);
        if (index === null) {
            return;
        }
        if (!Number.isInteger(index)) {
            picker.setAttribute("aria-invalid", "true");
            selectionStatus.textContent = "The body index must be a whole number.";
            return;
        }
        selectionStatus.textContent = "";
        selectedIndex = index;
        render();
    });
    for (const view of views) {
        bindViewControls(view.canvas, view.camera, render, (event) => selectFromPointer(event, view));
        view.canvas.addEventListener("keydown", (event) => {
            if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"].includes(event.key)) {
                return;
            }
            event.preventDefault();
            const direction = ["ArrowLeft", "ArrowDown"].includes(event.key) ? -1 : 1;
            selectedIndex = (selectedIndex + direction + bodies.length) % bodies.length;
            if (event.key === "Home") {
                selectedIndex = 0;
            }
            if (event.key === "End") {
                selectedIndex = bodies.length - 1;
            }
            picker.setAttribute("aria-invalid", "false");
            selectionStatus.textContent = "";
            render();
        });
    }
    window.addEventListener("resize", render);
    generate();
}

function initializeOrbit() {
    const canvas = document.querySelector("#orbit-view");
    const toggle = document.querySelector("#orbit-toggle");
    const status = document.querySelector("#orbit-status");
    const inputIds = {
        positionX: "orbit-position-x", positionY: "orbit-position-y", positionZ: "orbit-position-z",
        velocityX: "orbit-velocity-x", velocityY: "orbit-velocity-y", velocityZ: "orbit-velocity-z",
        centralMass: "orbit-central-mass", gravity: "orbit-gravity", timeStep: "orbit-time-step"
    };
    const inputs = Object.fromEntries(Object.entries(inputIds).map(([name, id]) => [
        name, document.getElementById(id)
    ]));
    const outputs = {
        position: document.querySelector("#orbit-position-output"),
        velocity: document.querySelector("#orbit-velocity-output"),
        speed: document.querySelector("#orbit-speed-output"),
        circularSpeed: document.querySelector("#orbit-circular-speed-output")
    };
    const camera = { yaw: -0.7, pitch: 0.65, zoom: 1, rotatable: true };
    let state;
    let parameters;
    const animation = createAnimationLoop(() => advanceOrbit(state, parameters), draw, (error) => {
        setRunning(false);
        status.textContent = error.message;
    });

    function readSettings() {
        const settings = {};
        for (const [name, input] of Object.entries(inputs)) {
            const label = input.getAttribute("aria-label") || input.closest("label").firstChild.textContent.trim();
            const value = readNumberInput(input, label, status);
            if (value === null) {
                return null;
            }
            settings[name] = value;
        }
        if (!Number.isFinite(settings.gravity * settings.centralMass)) {
            status.textContent = "Mass times gravity exceeds the numeric range. Reduce either value.";
            return null;
        }
        status.textContent = "";
        return settings;
    }

    function formatVector(vector) {
        return `(${vector.x.toFixed(3)}, ${vector.y.toFixed(3)}, ${vector.z.toFixed(3)})`;
    }

    function draw() {
        const { context, width, height } = prepareCanvas(canvas);
        const viewRadius = Math.max(3, state.initialDistance * 1.7);
        const scale = Math.min(width, height) * 0.39 / viewRadius * camera.zoom;
        const project = (point) => projectPoint(point, camera, width / 2, height / 2, scale);
        const origin = drawAxes(context, camera, width, height, scale, viewRadius * 0.65);

        if (state.trail.length > 1) {
            context.strokeStyle = chartColors.path;
            context.lineWidth = 1.5;
            context.beginPath();
            state.trail.forEach((point, index) => {
                const projected = project(point);
                if (index === 0) {
                    context.moveTo(projected.x, projected.y);
                } else {
                    context.lineTo(projected.x, projected.y);
                }
            });
            context.stroke();
        }
        const body = project(state.position);
        drawLine(context, origin, body, chartColors.selected);
        const speed = Math.hypot(state.velocity.x, state.velocity.y, state.velocity.z);
        if (speed > 0) {
            const arrowLength = viewRadius * 0.55;
            const velocityEnd = {
                x: state.position.x + state.velocity.x / speed * arrowLength,
                y: state.position.y + state.velocity.y / speed * arrowLength,
                z: state.position.z + state.velocity.z / speed * arrowLength
            };
            drawArrow(context, body, project(velocityEnd), chartColors.height);
        }
        drawPoint(context, origin, 8, chartColors.central);
        drawPoint(context, body, 6, chartColors.body);
        const distance = Math.max(Math.hypot(state.position.x, state.position.y, state.position.z), 0.0001);
        outputs.position.value = formatVector(state.position);
        outputs.velocity.value = formatVector(state.velocity);
        outputs.speed.value = speed.toFixed(3);
        outputs.circularSpeed.value = Math.sqrt(parameters.gravity * parameters.centralMass / distance).toFixed(3);
    }

    function setRunning(nextRunning) {
        if (nextRunning) {
            animation.start();
        } else {
            animation.stop();
        }
        toggle.textContent = nextRunning ? "Pause orbit" : state.elapsedTime > 0 ? "Resume orbit" : "Start orbit";
        toggle.setAttribute("aria-pressed", String(nextRunning));
    }

    function reset() {
        const settings = readSettings();
        if (!settings) {
            if (state) {
                setRunning(false);
            }
            return;
        }
        const position = { x: settings.positionX, y: settings.positionY, z: settings.positionZ };
        const velocity = { x: settings.velocityX, y: settings.velocityY, z: settings.velocityZ };
        const initialDistance = Math.max(Math.hypot(position.x, position.y, position.z), 0.25);
        if (!Number.isFinite(initialDistance * 1.7) || !Number.isFinite(Math.hypot(...Object.values(velocity)))) {
            status.textContent = "Starting coordinates exceed the numeric range. Reduce them before resetting.";
            if (state) {
                setRunning(false);
            }
            return;
        }
        parameters = settings;
        state = { position, velocity, initialDistance, trail: [], elapsedTime: 0 };
        setRunning(false);
        draw();
    }

    toggle.addEventListener("click", () => {
        const settings = readSettings();
        if (!settings) {
            setRunning(false);
            return;
        }
        parameters = settings;
        setRunning(!animation.running);
    });
    document.querySelector("#orbit-reset").addEventListener("click", reset);
    for (const input of Object.values(inputs)) {
        input.addEventListener("input", () => {
            const settings = readSettings();
            if (!settings) {
                setRunning(false);
                return;
            }
            parameters = settings;
            draw();
        });
    }
    bindViewControls(canvas, camera, draw);
    window.addEventListener("resize", draw);
    reset();
}

function initializeNBody() {
    const canvas = document.querySelector("#nbody-view");
    const toggle = document.querySelector("#nbody-toggle");
    const status = document.querySelector("#nbody-status");
    const inputs = {
        seed: document.querySelector("#nbody-seed"),
        count: document.querySelector("#nbody-count"),
        gravity: document.querySelector("#nbody-gravity"),
        softening: document.querySelector("#nbody-softening"),
        timeStep: document.querySelector("#nbody-time-step")
    };
    const outputs = {
        count: document.querySelector("#nbody-count-output"),
        time: document.querySelector("#nbody-time-output"),
        steps: document.querySelector("#nbody-steps-output"),
        speed: document.querySelector("#nbody-speed-output")
    };
    const camera = { yaw: -0.65, pitch: 0.55, zoom: 1, rotatable: true };
    let state;
    let needsReset = true;
    const animation = createAnimationLoop(() => advanceNBody(state, state.settings), draw, (error) => {
        setRunning(false);
        status.textContent = error.message;
    });

    function readSettings() {
        const settings = {};
        for (const [name, input] of Object.entries(inputs)) {
            const label = input.labels[0].textContent;
            const value = readNumberInput(input, label, status);
            if (value === null) {
                return null;
            }
            if (["seed", "count"].includes(name) && !Number.isSafeInteger(value)) {
                input.setAttribute("aria-invalid", "true");
                status.textContent = `${label} must be a whole number within JavaScript's safe integer range.`;
                return null;
            }
            settings[name] = value;
        }
        status.textContent = "";
        return settings;
    }

    function draw() {
        const { context, width, height } = prepareCanvas(canvas);
        let meanSpeed = 0;
        for (const body of state.bodies) {
            state.viewRadius = Math.max(state.viewRadius, Math.hypot(body.x, body.y, body.z) * 1.15);
            meanSpeed += Math.hypot(body.vx, body.vy, body.vz) / state.bodies.length;
        }
        const scale = Math.min(width, height) * 0.4 / state.viewRadius * camera.zoom;
        const points = state.bodies.map((body) => ({
            ...projectPoint(body, camera, width / 2, height / 2, scale),
            mass: body.mass,
            bodyIndex: body.bodyIndex
        })).sort((first, second) => first.depth - second.depth);
        for (const point of points) {
            drawPoint(context, point, 2 + point.mass * 0.3, getBodyColor(point.bodyIndex));
        }
        drawAxes(context, camera, width, height, scale, state.viewRadius * 0.65, 1, 4);
        outputs.count.value = state.bodies.length;
        outputs.time.value = state.elapsedTime.toFixed(3);
        outputs.steps.value = state.steps;
        outputs.speed.value = meanSpeed.toFixed(3);
    }

    function setRunning(nextRunning) {
        if (nextRunning) {
            animation.start();
        } else {
            animation.stop();
        }
        toggle.textContent = nextRunning ? "Pause"
            : needsReset || !state || state.steps === 0 ? "Start" : "Resume";
        toggle.setAttribute("aria-pressed", String(nextRunning));
    }

    function reset() {
        setRunning(false);
        const settings = readSettings();
        if (!settings) {
            return false;
        }
        state = createNBodyState(settings);
        needsReset = false;
        setRunning(false);
        draw();
        return true;
    }

    toggle.addEventListener("click", () => {
        if (animation.running) {
            setRunning(false);
            return;
        }
        if (needsReset && !reset()) {
            return;
        }
        setRunning(true);
    });
    document.querySelector("#nbody-reset").addEventListener("click", reset);
    for (const input of Object.values(inputs)) {
        input.addEventListener("input", () => {
            needsReset = true;
            setRunning(false);
            readSettings();
        });
    }
    bindViewControls(canvas, camera, draw);
    window.addEventListener("resize", draw);
    reset();
}

const computedStyles = getComputedStyle(document.documentElement);
const chartColors = Object.fromEntries(
    ["body", "selected", "height", "axis-x", "axis-y", "axis-z", "grid", "text", "central", "path"].map((name) => [
        name, computedStyles.getPropertyValue(`--plot-${name}`).trim()
    ])
);
const bodyColors = ["body", "body-teal", "body-purple", "body-gold", "body-rose", "body-slate"].map(
    (name) => computedStyles.getPropertyValue(`--plot-${name}`).trim()
);

initializeDisk();
initializeOrbit();
initializeNBody();
