class HandsOff {
    constructor() {
        // DOM elements
        this.video = document.getElementById('webcam');
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.alertOverlay = document.getElementById('alert');
        this.statusIndicator = document.getElementById('status');
        this.statusDot = this.statusIndicator.querySelector('.status-dot');
        this.statusText = this.statusIndicator.querySelector('.status-text');
        this.durationText = document.getElementById('duration');

        // Controls
        this.startBtn = document.getElementById('startBtn');
        this.stopBtn = document.getElementById('stopBtn');
        this.soundToggle = document.getElementById('soundToggle');
        this.thresholdSlider = document.getElementById('threshold');
        this.thresholdValue = document.getElementById('thresholdValue');
        this.sensitivitySlider = document.getElementById('sensitivity');
        this.sensitivityValue = document.getElementById('sensitivityValue');

        // Settings
        this.touchThreshold = 1.0; // seconds
        this.proximityThreshold = 0.08;
        this.enableSound = true;

        // State
        this.isRunning = false;
        this.touchStartTime = null;
        this.isTouching = false;
        this.alertActive = false;
        this.camera = null;

        // MediaPipe
        this.hands = null;
        this.faceDetection = null;
        this.lastHandResults = null;
        this.lastFaceResults = null;

        // Audio
        this.audioContext = null;
        this.alarmOscillator = null;
        this.alarmGain = null;
        this.alarmInterval = null;

        this.setupEventListeners();
    }

    setupEventListeners() {
        this.startBtn.addEventListener('click', () => this.start());
        this.stopBtn.addEventListener('click', () => this.stop());
        this.soundToggle.addEventListener('change', (e) => {
            this.enableSound = e.target.checked;
            if (!this.enableSound) {
                this.stopAlarm();
            }
        });
        this.thresholdSlider.addEventListener('input', (e) => {
            this.touchThreshold = parseFloat(e.target.value);
            this.thresholdValue.textContent = e.target.value;
        });
        this.sensitivitySlider.addEventListener('input', (e) => {
            const value = parseInt(e.target.value);
            const labels = ['Low', 'Medium', 'High'];
            const thresholds = [0.12, 0.08, 0.05];
            this.proximityThreshold = thresholds[value - 1];
            this.sensitivityValue.textContent = labels[value - 1];
        });
    }

    async start() {
        try {
            // Setup audio context
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();

            // Get webcam
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480 },
                audio: false
            });

            this.video.srcObject = stream;
            await this.video.play();

            // Set canvas size
            this.canvas.width = this.video.videoWidth;
            this.canvas.height = this.video.videoHeight;

            // Initialize MediaPipe
            await this.initMediaPipe();

            // Update UI
            this.isRunning = true;
            this.startBtn.disabled = true;
            this.stopBtn.disabled = false;
            this.updateStatus('Clear', false);

        } catch (error) {
            console.error('Error starting:', error);
            alert('Error accessing webcam. Please grant camera permissions and try again.');
        }
    }

    async initMediaPipe() {
        // Initialize hands detector
        this.hands = new Hands({
            locateFile: (file) => {
                return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
            }
        });

        this.hands.setOptions({
            maxNumHands: 2,
            modelComplexity: 1,
            minDetectionConfidence: 0.7,
            minTrackingConfidence: 0.5
        });

        this.hands.onResults((results) => {
            this.lastHandResults = results;
            this.processFrame();
        });

        // Initialize face detector
        this.faceDetection = new FaceDetection({
            locateFile: (file) => {
                return `https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/${file}`;
            }
        });

        this.faceDetection.setOptions({
            model: 'short',
            minDetectionConfidence: 0.7
        });

        this.faceDetection.onResults((results) => {
            this.lastFaceResults = results;
        });

        // Start camera
        this.camera = new Camera(this.video, {
            onFrame: async () => {
                await this.hands.send({image: this.video});
                await this.faceDetection.send({image: this.video});
            },
            width: 640,
            height: 480
        });

        this.camera.start();
    }

    processFrame() {
        if (!this.isRunning) return;

        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        let touchingNow = false;

        // Draw face detection
        if (this.lastFaceResults && this.lastFaceResults.detections) {
            for (const detection of this.lastFaceResults.detections) {
                const bbox = detection.boundingBox;

                // Draw face bounding box
                this.ctx.strokeStyle = '#00FF00';
                this.ctx.lineWidth = 2;
                this.ctx.strokeRect(
                    bbox.xCenter * this.canvas.width - (bbox.width * this.canvas.width) / 2,
                    bbox.yCenter * this.canvas.height - (bbox.height * this.canvas.height) / 2,
                    bbox.width * this.canvas.width,
                    bbox.height * this.canvas.height
                );

                // Check if hands are near face
                if (this.lastHandResults && this.lastHandResults.multiHandLandmarks) {
                    for (const landmarks of this.lastHandResults.multiHandLandmarks) {
                        // Draw hand landmarks
                        this.drawHandLandmarks(landmarks);

                        // Check proximity to face
                        if (this.isHandNearFace(landmarks, bbox)) {
                            touchingNow = true;
                        }
                    }
                }
            }
        }

        // Update touch state
        this.updateTouchState(touchingNow);
    }

    drawHandLandmarks(landmarks) {
        // Draw connections
        const connections = [
            [0, 1], [1, 2], [2, 3], [3, 4], // Thumb
            [0, 5], [5, 6], [6, 7], [7, 8], // Index
            [0, 9], [9, 10], [10, 11], [11, 12], // Middle
            [0, 13], [13, 14], [14, 15], [15, 16], // Ring
            [0, 17], [17, 18], [18, 19], [19, 20], // Pinky
            [5, 9], [9, 13], [13, 17] // Palm
        ];

        this.ctx.strokeStyle = '#FF0000';
        this.ctx.lineWidth = 2;

        for (const [start, end] of connections) {
            const startPoint = landmarks[start];
            const endPoint = landmarks[end];

            this.ctx.beginPath();
            this.ctx.moveTo(startPoint.x * this.canvas.width, startPoint.y * this.canvas.height);
            this.ctx.lineTo(endPoint.x * this.canvas.width, endPoint.y * this.canvas.height);
            this.ctx.stroke();
        }

        // Draw landmarks
        this.ctx.fillStyle = '#00FF00';
        for (const landmark of landmarks) {
            this.ctx.beginPath();
            this.ctx.arc(
                landmark.x * this.canvas.width,
                landmark.y * this.canvas.height,
                3,
                0,
                2 * Math.PI
            );
            this.ctx.fill();
        }
    }

    isHandNearFace(handLandmarks, faceBbox) {
        const margin = this.proximityThreshold;
        const expandedBox = {
            xMin: faceBbox.xCenter - (faceBbox.width / 2) - margin,
            xMax: faceBbox.xCenter + (faceBbox.width / 2) + margin,
            yMin: faceBbox.yCenter - (faceBbox.height / 2) - margin,
            yMax: faceBbox.yCenter + (faceBbox.height / 2) + margin
        };

        // Check key hand points (wrist + fingertips)
        const keyPoints = [0, 4, 8, 12, 16, 20];

        for (const idx of keyPoints) {
            const landmark = handLandmarks[idx];
            if (landmark.x >= expandedBox.xMin && landmark.x <= expandedBox.xMax &&
                landmark.y >= expandedBox.yMin && landmark.y <= expandedBox.yMax) {
                return true;
            }
        }

        return false;
    }

    updateTouchState(touchingNow) {
        const currentTime = Date.now() / 1000;

        if (touchingNow) {
            if (!this.isTouching) {
                // Just started touching
                this.touchStartTime = currentTime;
                this.isTouching = true;
                this.alertActive = false;
                this.updateStatus('TOUCHING', true);
            } else {
                // Still touching - check duration
                const duration = currentTime - this.touchStartTime;

                if (duration >= this.touchThreshold && !this.alertActive) {
                    this.alertActive = true;
                    console.log(`ALERT: Hands on face for ${duration.toFixed(1)}s`);
                    this.startAlarm();
                }

                if (this.alertActive) {
                    this.alertOverlay.classList.add('active');
                    this.durationText.textContent = `Duration: ${duration.toFixed(1)}s`;
                }
            }
        } else {
            // Not touching
            if (this.isTouching) {
                const duration = currentTime - this.touchStartTime;
                console.log(`Touch ended after ${duration.toFixed(1)}s`);
                this.stopAlarm();
            }

            this.isTouching = false;
            this.touchStartTime = null;
            this.alertActive = false;
            this.alertOverlay.classList.remove('active');
            this.updateStatus('Clear', false);
        }
    }

    updateStatus(text, isTouching) {
        this.statusText.textContent = text;
        if (isTouching) {
            this.statusDot.classList.add('touching');
        } else {
            this.statusDot.classList.remove('touching');
        }
    }

    startAlarm() {
        if (!this.enableSound || this.alarmOscillator) return;

        // Create annoying alternating beep alarm
        const frequencies = [1200, 800]; // Hz
        let freqIndex = 0;

        this.alarmGain = this.audioContext.createGain();
        this.alarmGain.connect(this.audioContext.destination);
        this.alarmGain.gain.value = 0.3;

        const playBeep = () => {
            if (!this.enableSound || !this.alertActive) return;

            const oscillator = this.audioContext.createOscillator();
            const gain = this.audioContext.createGain();

            oscillator.connect(gain);
            gain.connect(this.alarmGain);

            oscillator.type = 'sine';
            oscillator.frequency.value = frequencies[freqIndex];

            // Amplitude modulation envelope
            gain.gain.setValueAtTime(0.3, this.audioContext.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.01, this.audioContext.currentTime + 0.3);

            oscillator.start();
            oscillator.stop(this.audioContext.currentTime + 0.3);

            freqIndex = (freqIndex + 1) % frequencies.length;
        };

        // Start repeating beeps
        playBeep();
        this.alarmInterval = setInterval(playBeep, 300);
    }

    stopAlarm() {
        if (this.alarmInterval) {
            clearInterval(this.alarmInterval);
            this.alarmInterval = null;
        }
        if (this.alarmGain) {
            this.alarmGain.disconnect();
            this.alarmGain = null;
        }
        this.alarmOscillator = null;
    }

    stop() {
        this.isRunning = false;

        // Stop camera
        if (this.camera) {
            this.camera.stop();
        }

        // Stop webcam stream
        if (this.video.srcObject) {
            this.video.srcObject.getTracks().forEach(track => track.stop());
            this.video.srcObject = null;
        }

        // Stop alarm
        this.stopAlarm();

        // Clean up MediaPipe
        if (this.hands) {
            this.hands.close();
        }
        if (this.faceDetection) {
            this.faceDetection.close();
        }

        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Reset state
        this.isTouching = false;
        this.touchStartTime = null;
        this.alertActive = false;
        this.alertOverlay.classList.remove('active');

        // Update UI
        this.startBtn.disabled = false;
        this.stopBtn.disabled = true;
        this.updateStatus('Camera Off', false);
    }
}

// Initialize app when page loads
window.addEventListener('DOMContentLoaded', () => {
    new HandsOff();
});
