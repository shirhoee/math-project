
let scene, camera, renderer, controls, mesh;

function init3D() {
    const container = document.getElementById('viewer-container');
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x222222);

    camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 100);
    // Position will be updated based on mesh data
    camera.position.set(0, 0, 0); 

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    container.appendChild(renderer.domElement);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    // We will constrain orbit angles later based on mesh bounds

    window.addEventListener('resize', onWindowResize, false);
    animate();
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

function animate() {
    requestAnimationFrame(animate);
    if(controls) controls.update();
    if(renderer && scene && camera) renderer.render(scene, camera);
}

function b64ToFloat32Array(b64) {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) {
        bytes[i] = bin.charCodeAt(i);
    }
    return new Float32Array(bytes.buffer);
}

function b64ToUint32Array(b64) {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) {
        bytes[i] = bin.charCodeAt(i);
    }
    return new Uint32Array(bytes.buffer);
}

function displayError(msg) {
    const box = document.getElementById('status-box');
    box.className = 'error';
    box.innerText = msg;
}

function displaySuccess(msg, timeMs) {
    const box = document.getElementById('status-box');
    box.className = 'success';
    box.innerText = msg +  (ms);
}

function setLoading(isLoading) {
    document.getElementById('loading-overlay').className = isLoading ? '' : 'hidden';
}

async function loadSamplesList() {
    try {
        const res = await fetch('/samples_list');
        const list = await res.json();
        const select = document.getElementById('sample-select');
        list.forEach(s => {
            const opt = document.createElement('option');
            opt.value = s.name;
            opt.innerText = s.name;
            select.appendChild(opt);
        });
    } catch (e) {
        console.error(e);
    }
}

async function processRequest(formData) {
    setLoading(true);
    const t0 = performance.now();
    try {
        const res = await fetch('/process', {
            method: 'POST',
            body: formData
        });
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.error || "Server error");
        }
        
        const data = await res.json();
        buildThreeMesh(data);
        
        const t1 = performance.now();
        displaySuccess("Loaded successfully", Math.round(t1 - t0));
    } catch (e) {
        displayError(e.message);
    } finally {
        setLoading(false);
    }
}

function buildThreeMesh(data) {
    if(mesh) {
        scene.remove(mesh);
        mesh.geometry.dispose();
        mesh.material.map.dispose();
        mesh.material.dispose();
    }

    const vertices = b64ToFloat32Array(data.vertices);
    const indices = b64ToUint32Array(data.indices);
    const uvs = b64ToFloat32Array(data.uvs);

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
    geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
    geometry.setIndex(new THREE.BufferAttribute(indices, 1));
    geometry.computeVertexNormals();

    const image = new Image();
    image.src = data.texture;
    const texture = new THREE.Texture(image);
    image.onload = () => texture.needsUpdate = true;
    texture.colorSpace = THREE.SRGBColorSpace; // Optional in older versions, using default

    const material = new THREE.MeshBasicMaterial({ 
        map: texture,
        side: THREE.DoubleSide
    });

    mesh = new THREE.Mesh(geometry, material);
    
    // The points are (X, Y, Z). Let's adjust scale and position to fit the view nicely.
    // In our math engine Z is positive depth. In Three.js camera looks down -Z.
    // So we flip Z, or we put camera at 0,0,0 and rotate mesh.
    // Actually, X = X, Y = Y, Z = -Z (so it's in front of camera)
    
    mesh.scale.set(1, -1, -1); // flip Y and Z so it looks correct
    scene.add(mesh);

    // Position camera
    // We can compute the median depth or just set it to 0,0,0
    camera.position.set(0, 0, 0);
    controls.target.set(0, 0, -2); // look into the screen
    controls.update();
    
    // Limits
    controls.minDistance = 0.5;
    controls.maxDistance = 5.0;
    // Limit angle to avoid extreme smearing
    controls.minAzimuthAngle = -Math.PI/6; // -30 deg
    controls.maxAzimuthAngle = Math.PI/6;
    controls.minPolarAngle = Math.PI/2 - Math.PI/6;
    controls.maxPolarAngle = Math.PI/2 + Math.PI/6;
}

document.getElementById('load-sample-btn').addEventListener('click', () => {
    const val = document.getElementById('sample-select').value;
    if(!val) return;
    const fd = new FormData();
    fd.append('sample_name', val);
    processRequest(fd);
});

document.getElementById('upload-btn').addEventListener('click', () => {
    const file = document.getElementById('upload-input').files[0];
    if(!file) return;
    const fd = new FormData();
    fd.append('file', file);
    processRequest(fd);
});

init3D();
loadSamplesList();
