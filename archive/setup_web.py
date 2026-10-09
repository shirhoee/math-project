import os
from pathlib import Path

os.makedirs('v2/web', exist_ok=True)

html_code = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>3D Photo Viewer (V2)</title>
    <link rel="stylesheet" href="/web/style.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
    <div id="ui-container">
        <h1>3D Photo Viewer</h1>
        
        <div class="controls">
            <label for="sample-select">Samples:</label>
            <select id="sample-select">
                <option value="">-- Select a sample --</option>
            </select>
            <button id="load-sample-btn">Load Sample</button>
        </div>
        
        <div class="controls">
            <label for="upload-input">Or Upload:</label>
            <input type="file" id="upload-input" accept="image/*">
            <button id="upload-btn">Upload & Process</button>
        </div>

        <div id="status-box" class="hidden"></div>
    </div>

    <div id="loading-overlay" class="hidden">
        <div class="spinner"></div>
        <p>Processing...</p>
    </div>

    <div id="viewer-container"></div>

    <script src="/web/app.js"></script>
</body>
</html>
'''
Path('v2/web/index.html').write_text(html_code)

css_code = '''body {
    margin: 0;
    overflow: hidden;
    font-family: sans-serif;
    background-color: #111;
    color: #eee;
}

#ui-container {
    position: absolute;
    top: 10px;
    left: 10px;
    background: rgba(0, 0, 0, 0.7);
    padding: 15px;
    border-radius: 8px;
    z-index: 10;
}

h1 {
    margin: 0 0 15px 0;
    font-size: 1.2rem;
}

.controls {
    margin-bottom: 10px;
}

button {
    background: #4caf50;
    color: white;
    border: none;
    padding: 5px 10px;
    cursor: pointer;
    border-radius: 3px;
}
button:hover { background: #45a049; }
button:disabled { background: #888; cursor: not-allowed; }

#status-box {
    margin-top: 10px;
    padding: 10px;
    background: #333;
    border-radius: 4px;
    font-size: 0.9rem;
}
.hidden { display: none !important; }
.error { color: #ff5555; }
.success { color: #55ff55; }

#loading-overlay {
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0,0,0,0.8);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    z-index: 20;
}

.spinner {
    border: 4px solid #f3f3f3;
    border-top: 4px solid #3498db;
    border-radius: 50%;
    width: 40px;
    height: 40px;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

#viewer-container {
    width: 100vw;
    height: 100vh;
}
'''
Path('v2/web/style.css').write_text(css_code)

js_code = '''
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
'''
Path('v2/web/app.js').write_text(js_code)
