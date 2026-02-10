/**
 * PetAI - Dog Breed & Audio Classifier Frontend
 * Handles image upload, audio upload, API calls, and results display
 */

document.addEventListener('DOMContentLoaded', () => {
    // ===== DOM Elements - Image Mode =====
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const uploadContent = document.getElementById('uploadContent');
    const previewContainer = document.getElementById('previewContainer');
    const previewImage = document.getElementById('previewImage');
    const changeImageBtn = document.getElementById('changeImageBtn');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const imageUploadSection = document.getElementById('imageUploadSection');
    const loadingSection = document.getElementById('loadingSection');
    const resultsSection = document.getElementById('resultsSection');
    const tryAgainBtn = document.getElementById('tryAgainBtn');

    // ===== DOM Elements - Audio Mode =====
    const audioUploadArea = document.getElementById('audioUploadArea');
    const audioFileInput = document.getElementById('audioFileInput');
    const audioUploadContent = document.getElementById('audioUploadContent');
    const audioPreviewContainer = document.getElementById('audioPreviewContainer');
    const audioPlayer = document.getElementById('audioPlayer');
    const audioFileName = document.getElementById('audioFileName');
    const changeAudioBtn = document.getElementById('changeAudioBtn');
    const analyzeAudioBtn = document.getElementById('analyzeAudioBtn');
    const audioUploadSection = document.getElementById('audioUploadSection');
    const audioResultsSection = document.getElementById('audioResultsSection');
    const tryAgainAudioBtn = document.getElementById('tryAgainAudioBtn');

    // ===== DOM Elements - Tabs =====
    const imageTab = document.getElementById('imageTab');
    const audioTab = document.getElementById('audioTab');

    let selectedFile = null;
    let selectedAudioFile = null;
    let currentMode = 'image';

    // ===== Tab Switching =====

    imageTab.addEventListener('click', () => switchMode('image'));
    audioTab.addEventListener('click', () => switchMode('audio'));

    function switchMode(mode) {
        currentMode = mode;

        // Update tab styles
        imageTab.classList.toggle('active', mode === 'image');
        audioTab.classList.toggle('active', mode === 'audio');

        // Toggle sections
        imageUploadSection.style.display = mode === 'image' ? 'block' : 'none';
        audioUploadSection.style.display = mode === 'audio' ? 'block' : 'none';
        
        // Hide results & loading
        loadingSection.style.display = 'none';
        resultsSection.style.display = 'none';
        audioResultsSection.style.display = 'none';
    }

    // ===== IMAGE: File Upload Handling =====
    
    uploadArea.addEventListener('click', (e) => {
        if (e.target !== changeImageBtn && !changeImageBtn.contains(e.target)) {
            fileInput.click();
        }
    });

    changeImageBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) handleFileSelect(file);
    });

    // Image Drag and Drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });

    uploadArea.addEventListener('dragleave', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            handleFileSelect(file);
        }
    });

    function handleFileSelect(file) {
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            uploadContent.style.display = 'none';
            previewContainer.style.display = 'block';
            analyzeBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    // ===== AUDIO: File Upload Handling =====

    audioUploadArea.addEventListener('click', (e) => {
        if (e.target !== changeAudioBtn && !changeAudioBtn.contains(e.target)) {
            audioFileInput.click();
        }
    });

    changeAudioBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        audioFileInput.click();
    });

    audioFileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) handleAudioSelect(file);
    });

    // Audio Drag and Drop
    audioUploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        audioUploadArea.classList.add('drag-over');
    });

    audioUploadArea.addEventListener('dragleave', (e) => {
        e.preventDefault();
        audioUploadArea.classList.remove('drag-over');
    });

    audioUploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        audioUploadArea.classList.remove('drag-over');
        const file = e.dataTransfer.files[0];
        if (file && (file.type.startsWith('audio/') || file.name.toLowerCase().endsWith('.wav'))) {
            handleAudioSelect(file);
        }
    });

    function handleAudioSelect(file) {
        selectedAudioFile = file;
        audioFileName.textContent = file.name;

        // Create audio preview URL
        const url = URL.createObjectURL(file);
        audioPlayer.src = url;

        // Draw waveform preview
        drawWaveformPreview(file);

        audioUploadContent.style.display = 'none';
        audioPreviewContainer.style.display = 'flex';
        analyzeAudioBtn.disabled = false;
    }

    async function drawWaveformPreview(file) {
        try {
            const arrayBuffer = await file.arrayBuffer();
            const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
            const data = audioBuffer.getChannelData(0);

            const canvas = document.getElementById('waveformCanvas');
            fitCanvasToDisplay(canvas);
            const ctx = canvas.getContext('2d');
            drawWaveform(ctx, canvas, data);
            audioCtx.close();
        } catch (err) {
            console.log('Could not decode audio for preview:', err);
        }
    }

    function fitCanvasToDisplay(canvas) {
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        const ctx = canvas.getContext('2d');
        ctx.scale(dpr, dpr);
    }

    function drawWaveform(ctx, canvas, data) {
        const dpr = window.devicePixelRatio || 1;
        const w = canvas.width / dpr;
        const h = canvas.height / dpr;
        ctx.clearRect(0, 0, w, h);

        // Background
        ctx.fillStyle = 'rgba(99, 102, 241, 0.05)';
        ctx.fillRect(0, 0, w, h);

        // Center line
        ctx.strokeStyle = 'rgba(99, 102, 241, 0.15)';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(0, h / 2);
        ctx.lineTo(w, h / 2);
        ctx.stroke();
        ctx.setLineDash([]);

        if (!data || data.length === 0) return;

        // Normalize data to [-1, 1]
        let absMax = 0;
        for (let i = 0; i < data.length; i++) {
            const a = Math.abs(data[i]);
            if (a > absMax) absMax = a;
        }
        if (absMax === 0) absMax = 1;

        // Build per-pixel min/max buckets
        const numBars = Math.floor(w);
        const step = data.length / numBars;

        // Gradient for fill
        const gradient = ctx.createLinearGradient(0, 0, w, 0);
        gradient.addColorStop(0, 'rgba(99, 102, 241, 0.6)');
        gradient.addColorStop(0.5, 'rgba(249, 115, 22, 0.6)');
        gradient.addColorStop(1, 'rgba(16, 185, 129, 0.6)');

        const strokeGrad = ctx.createLinearGradient(0, 0, w, 0);
        strokeGrad.addColorStop(0, '#6366f1');
        strokeGrad.addColorStop(0.5, '#f97316');
        strokeGrad.addColorStop(1, '#10b981');

        // Draw filled waveform (mirror style)
        ctx.beginPath();
        ctx.moveTo(0, h / 2);

        // Top half (positive envelope)
        for (let i = 0; i < numBars; i++) {
            const start = Math.floor(i * step);
            const end = Math.min(Math.floor((i + 1) * step), data.length);
            let max = 0;
            for (let j = start; j < end; j++) {
                const a = Math.abs(data[j]) / absMax;
                if (a > max) max = a;
            }
            const yTop = h / 2 - max * (h / 2 - 4);
            ctx.lineTo(i, yTop);
        }

        // Come back along bottom half (negative envelope, mirrored)
        for (let i = numBars - 1; i >= 0; i--) {
            const start = Math.floor(i * step);
            const end = Math.min(Math.floor((i + 1) * step), data.length);
            let max = 0;
            for (let j = start; j < end; j++) {
                const a = Math.abs(data[j]) / absMax;
                if (a > max) max = a;
            }
            const yBot = h / 2 + max * (h / 2 - 4);
            ctx.lineTo(i, yBot);
        }

        ctx.closePath();
        ctx.fillStyle = gradient;
        ctx.fill();

        // Stroke the top edge for crisp line
        ctx.beginPath();
        for (let i = 0; i < numBars; i++) {
            const start = Math.floor(i * step);
            const end = Math.min(Math.floor((i + 1) * step), data.length);
            let max = 0;
            for (let j = start; j < end; j++) {
                const a = Math.abs(data[j]) / absMax;
                if (a > max) max = a;
            }
            const yTop = h / 2 - max * (h / 2 - 4);
            if (i === 0) ctx.moveTo(i, yTop);
            else ctx.lineTo(i, yTop);
        }
        ctx.strokeStyle = strokeGrad;
        ctx.lineWidth = 1;
        ctx.stroke();

        // Mirror stroke on bottom
        ctx.beginPath();
        for (let i = 0; i < numBars; i++) {
            const start = Math.floor(i * step);
            const end = Math.min(Math.floor((i + 1) * step), data.length);
            let max = 0;
            for (let j = start; j < end; j++) {
                const a = Math.abs(data[j]) / absMax;
                if (a > max) max = a;
            }
            const yBot = h / 2 + max * (h / 2 - 4);
            if (i === 0) ctx.moveTo(i, yBot);
            else ctx.lineTo(i, yBot);
        }
        ctx.strokeStyle = strokeGrad;
        ctx.lineWidth = 1;
        ctx.stroke();
    }

    // ===== IMAGE: Analyze Button =====
    
    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        imageUploadSection.style.display = 'none';
        resultsSection.style.display = 'none';
        loadingSection.style.display = 'flex';

        try {
            const formData = new FormData();
            formData.append('file', selectedFile);

            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                displayResults(data);
            } else {
                alert('Error: ' + (data.error || 'Unknown error occurred'));
                resetToUpload();
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Failed to analyze image. Please try again.');
            resetToUpload();
        }
    });

    // ===== AUDIO: Analyze Button =====

    analyzeAudioBtn.addEventListener('click', async () => {
        if (!selectedAudioFile) return;

        audioUploadSection.style.display = 'none';
        audioResultsSection.style.display = 'none';
        loadingSection.style.display = 'flex';

        try {
            const formData = new FormData();
            formData.append('file', selectedAudioFile);

            const response = await fetch('/predict_audio', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                displayAudioResults(data);
            } else {
                alert('Error: ' + (data.error || 'Audio classification failed'));
                resetToAudioUpload();
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Failed to classify audio. Please try again.');
            resetToAudioUpload();
        }
    });

    // ===== Try Again Buttons =====
    
    tryAgainBtn.addEventListener('click', resetToUpload);
    tryAgainAudioBtn.addEventListener('click', resetToAudioUpload);

    function resetToUpload() {
        selectedFile = null;
        fileInput.value = '';
        previewImage.src = '';
        uploadContent.style.display = 'block';
        previewContainer.style.display = 'none';
        analyzeBtn.disabled = true;
        
        loadingSection.style.display = 'none';
        resultsSection.style.display = 'none';
        imageUploadSection.style.display = 'block';
        
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    function resetToAudioUpload() {
        selectedAudioFile = null;
        audioFileInput.value = '';
        audioPlayer.src = '';
        audioUploadContent.style.display = 'block';
        audioPreviewContainer.style.display = 'none';
        analyzeAudioBtn.disabled = true;

        loadingSection.style.display = 'none';
        audioResultsSection.style.display = 'none';
        audioUploadSection.style.display = 'block';

        window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    // ===== Display Audio Results =====

    function displayAudioResults(data) {
        loadingSection.style.display = 'none';
        audioResultsSection.style.display = 'block';

        const predictions = data.predictions;
        const top = predictions[0];

        // Species icon and label
        const speciesIcon = document.getElementById('speciesIcon');
        const speciesLabel = document.getElementById('speciesLabel');
        speciesIcon.textContent = top.class === 'dogs' ? '🐕' : '🐱';
        speciesLabel.textContent = top.label;

        // Confidence ring
        const confidenceText = document.getElementById('confidenceText');
        const ring = document.getElementById('confidenceRing');
        const pct = top.confidence;
        confidenceText.textContent = `${pct.toFixed(1)}%`;

        const circumference = 2 * Math.PI * 52;
        const offset = circumference - (pct / 100) * circumference;
        ring.style.strokeDasharray = circumference;
        ring.style.strokeDashoffset = offset;

        // Color the ring based on class
        ring.style.stroke = top.class === 'dogs' ? '#6366f1' : '#f97316';

        // Predictions bars
        const barContainer = document.getElementById('audioPredictionsList');
        barContainer.innerHTML = predictions.map(pred => `
            <div class="audio-pred-item">
                <div class="audio-pred-header">
                    <span class="audio-pred-icon">${pred.class === 'dogs' ? '🐕' : '🐱'}</span>
                    <span class="audio-pred-label">${pred.label}</span>
                    <span class="audio-pred-pct">${pred.confidence.toFixed(1)}%</span>
                </div>
                <div class="audio-pred-bar">
                    <div class="audio-pred-fill ${pred.class}" style="width: ${pred.confidence}%"></div>
                </div>
            </div>
        `).join('');

        // Waveform in results
        if (data.waveform && data.waveform.length > 0) {
            const canvas = document.getElementById('resultWaveform');
            // Wait a tick for the canvas to be visible & sized by CSS
            requestAnimationFrame(() => {
                fitCanvasToDisplay(canvas);
                const ctx = canvas.getContext('2d');
                drawWaveform(ctx, canvas, new Float32Array(data.waveform));
            });
        }

        // Audio player in results
        if (data.audio_path) {
            document.getElementById('resultAudioPlayer').src = data.audio_path;
        }

        audioResultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    // ===== Display Image Results =====
    
    function displayResults(data) {
        loadingSection.style.display = 'none';
        resultsSection.style.display = 'block';

        const predictions = data.predictions;
        const topPrediction = predictions[0];

        // Multi-dog warning disabled - confidence-based detection is unreliable
        // Low confidence can mean: puppies, unusual angles, poor lighting, mixed breeds, etc.
        // Not necessarily multiple dogs. A proper multi-dog detector would need object detection.
        const multiDogWarning = document.getElementById('multiDogWarning');
        multiDogWarning.style.display = 'none';

        // Set result image
        document.getElementById('resultImage').src = data.image_path;

        // Set top breed
        document.getElementById('topBreedName').textContent = topPrediction.breed;
        document.getElementById('topConfidence').textContent = 
            `${topPrediction.confidence.toFixed(1)}% Confidence`;

        // Predictions list
        renderPredictionsList(predictions);

        // Get breed info
        const info = topPrediction.info;

        if (info) {
            // Physical attributes
            renderAttributes(info);

            // Behavioral traits
            renderTraits(info.traits);

            // Capability scores
            renderCapabilityScores(info.capability_scores);

            // Breed info
            renderBreedInfo(info);
        } else {
            // No behavioral data available
            document.getElementById('attributesGrid').innerHTML = 
                '<p style="color: var(--text-secondary); grid-column: span 2;">No data available for this breed</p>';
            document.getElementById('traitsChart').innerHTML = 
                '<p style="color: var(--text-secondary);">No behavioral data available</p>';
            document.getElementById('capabilityMeters').innerHTML = 
                '<p style="color: var(--text-secondary);">No capability data available</p>';
            document.getElementById('infoContent').innerHTML = 
                '<p style="color: var(--text-secondary);">No additional information available for this breed</p>';
        }

        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    function renderPredictionsList(predictions) {
        const container = document.getElementById('predictionsList');
        container.innerHTML = predictions.map((pred, idx) => `
            <div class="prediction-item">
                <span class="prediction-rank ${idx === 0 ? 'rank-1' : ''}">${idx + 1}</span>
                <div class="prediction-info">
                    <span class="prediction-breed">${pred.breed}</span>
                    <div class="prediction-bar">
                        <div class="prediction-fill" style="width: ${pred.confidence}%"></div>
                    </div>
                </div>
                <span class="prediction-percent">${pred.confidence.toFixed(1)}%</span>
            </div>
        `).join('');
    }

    function renderAttributes(info) {
        const container = document.getElementById('attributesGrid');
        const attributes = [];

        if (info.height) {
            attributes.push({
                icon: 'fa-ruler-vertical',
                label: 'Height',
                value: info.height
            });
        }
        if (info.weight) {
            attributes.push({
                icon: 'fa-weight-scale',
                label: 'Weight',
                value: info.weight
            });
        }
        if (info.lifespan) {
            attributes.push({
                icon: 'fa-heart',
                label: 'Lifespan',
                value: info.lifespan
            });
        }
        if (info.group) {
            attributes.push({
                icon: 'fa-layer-group',
                label: 'Group',
                value: info.group
            });
        }

        if (attributes.length === 0) {
            container.innerHTML = '<p style="color: var(--text-secondary); grid-column: span 2;">No physical data available</p>';
            return;
        }

        container.innerHTML = attributes.map(attr => `
            <div class="attribute-item">
                <div class="attribute-icon"><i class="fas ${attr.icon}"></i></div>
                <span class="attribute-label">${attr.label}</span>
                <span class="attribute-value">${attr.value}</span>
            </div>
        `).join('');
    }

    function renderTraits(traits) {
        const container = document.getElementById('traitsChart');
        
        if (!traits || Object.keys(traits).length === 0) {
            container.innerHTML = '<p style="color: var(--text-secondary);">No behavioral data available</p>';
            return;
        }

        container.innerHTML = Object.entries(traits).slice(0, 8).map(([label, value]) => `
            <div class="trait-item">
                <span class="trait-label">${label}</span>
                <div class="trait-bars">
                    ${[1, 2, 3, 4, 5].map(i => 
                        `<div class="trait-dot ${i <= value ? 'active' : ''}"></div>`
                    ).join('')}
                </div>
                <span class="trait-value">${value}/5</span>
            </div>
        `).join('');
    }

    function renderCapabilityScores(scores) {
        const container = document.getElementById('capabilityMeters');
        
        if (!scores || Object.keys(scores).length === 0) {
            container.innerHTML = '<p style="color: var(--text-secondary);">No capability data available</p>';
            return;
        }

        const iconMap = {
            'Working Suitability': 'fa-briefcase',
            'Family Companion': 'fa-home',
            'Guard Dog Potential': 'fa-shield-alt'
        };

        const classMap = {
            'Working Suitability': 'working',
            'Family Companion': 'family',
            'Guard Dog Potential': 'guard'
        };

        container.innerHTML = Object.entries(scores).map(([label, value]) => `
            <div class="capability-item">
                <div class="capability-header">
                    <span class="capability-label">
                        <i class="fas ${iconMap[label] || 'fa-star'}"></i>
                        ${label}
                    </span>
                    <span class="capability-score">${value}%</span>
                </div>
                <div class="capability-bar">
                    <div class="capability-fill ${classMap[label] || ''}" style="width: ${value}%"></div>
                </div>
            </div>
        `).join('');
    }

    function renderBreedInfo(info) {
        const container = document.getElementById('infoContent');
        let html = '';

        if (info.temperament) {
            const traits = info.temperament.split(',').map(t => t.trim());
            html += `
                <div class="info-row">
                    <span class="info-label">Temperament</span>
                    <div class="temperament-tags">
                        ${traits.map(t => `<span class="temp-tag">${t}</span>`).join('')}
                    </div>
                </div>
            `;
        }

        if (info.description) {
            html += `
                <div class="info-row">
                    <span class="info-label">Description</span>
                    <span class="info-text">${info.description}</span>
                </div>
            `;
        }

        if (!html) {
            html = '<p style="color: var(--text-secondary);">No additional information available for this breed</p>';
        }

        container.innerHTML = html;
    }
});
