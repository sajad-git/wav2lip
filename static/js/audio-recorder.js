/**
 * Audio Recorder - Handles audio recording with Persian language support
 */

class AudioRecorder {
    constructor() {
        this.mediaRecorder = null;
        this.audioStream = null;
        this.audioChunks = [];
        this.isRecording = false;
        this.recordButton = null;
        this.recordingStatus = null;
        this.recordingTimer = null;
        this.startTime = null;
        this.websocket_client = null;
        
        this.initializeRecorder();
    }
    
    /**
     * Initialize audio recorder
     */
    async initializeRecorder() {
        try {
            await this.setupAudioConstraints();
            this.setupUI();
            console.log('✅ Audio recorder initialized');
        } catch (error) {
            console.error('❌ Failed to initialize audio recorder:', error);
            this.showError('Microphone access denied or not available');
        }
    }
    
    /**
     * Set WebSocket client reference
     * @param {AvatarWebSocketClient} client - WebSocket client instance
     */
    setWebSocketClient(client) {
        this.websocket_client = client;
    }
    
    /**
     * Setup audio constraints optimized for speech
     */
    async setupAudioConstraints() {
        const constraints = {
            audio: {
                sampleRate: 16000,          // Optimal for speech recognition
                channelCount: 1,            // Mono audio
                echoCancellation: true,     // Reduce echo
                noiseSuppression: true,     // Reduce background noise
                autoGainControl: true,      // Normalize volume
                suppressLocalAudioPlayback: false
            }
        };
        
        try {
            this.audioStream = await navigator.mediaDevices.getUserMedia(constraints);
            console.log('🎤 Microphone access granted');
        } catch (error) {
            console.error('❌ Microphone access failed:', error);
            throw new Error('Microphone access required for audio recording');
        }
    }
    
    /**
     * Setup UI elements
     */
    setupUI() {
        this.recordButton = document.getElementById('record-button');
        this.recordingStatus = document.getElementById('recording-status');
        this.recordingTimer = document.getElementById('recording-timer');
        
        if (this.recordButton) {
            this.recordButton.addEventListener('click', this.toggleRecording.bind(this));
        }
        
        // Initialize UI state
        this.updateUI();
    }
    
    /**
     * Toggle recording state
     */
    async toggleRecording() {
        if (this.isRecording) {
            await this.stopRecording();
        } else {
            await this.startRecording();
        }
    }
    
    /**
     * Start audio recording
     */
    async startRecording() {
        try {
            // Clear previous recording data
            this.audioChunks = [];
            
            // Create new MediaRecorder
            this.mediaRecorder = new MediaRecorder(this.audioStream, {
                mimeType: this.getSupportedMimeType()
            });
            
            // Setup event handlers
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                }
            };
            
            this.mediaRecorder.onstop = () => {
                this.processRecordedAudio();
            };
            
            this.mediaRecorder.onerror = (error) => {
                console.error('❌ MediaRecorder error:', error);
                this.showError('Recording error occurred');
            };
            
            // Start recording
            this.mediaRecorder.start(1000); // Collect data every second
            this.isRecording = true;
            this.startTime = Date.now();
            
            // Update UI
            this.updateUI();
            this.startTimer();
            
            console.log('🎤 Recording started');
            
        } catch (error) {
            console.error('❌ Failed to start recording:', error);
            this.showError('Failed to start recording');
        }
    }
    
    /**
     * Stop audio recording
     */
    async stopRecording() {
        if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
            this.mediaRecorder.stop();
            this.isRecording = false;
            
            // Update UI
            this.updateUI();
            this.stopTimer();
            
            console.log('🛑 Recording stopped');
        }
    }
    
    /**
     * Process recorded audio
     */
    async processRecordedAudio() {
        try {
            if (this.audioChunks.length === 0) {
                console.warn('⚠️ No audio data recorded');
                return;
            }
            
            // Create audio blob
            const audioBlob = new Blob(this.audioChunks, { 
                type: this.getSupportedMimeType() 
            });
            
            console.log(`📊 Audio recorded: ${audioBlob.size} bytes, ${this.getRecordingDuration()}s`);
            
            // Validate audio duration
            const duration = this.getRecordingDuration();
            if (duration < 0.5) {
                this.showError('Recording too short (minimum 0.5 seconds)');
                return;
            }
            
            if (duration > 30) {
                this.showError('Recording too long (maximum 30 seconds)');
                return;
            }
            
            // Convert to wav format if needed
            const processedBlob = await this.convertToWav(audioBlob);
            
            // Send to server via WebSocket
            if (this.websocket_client) {
                const metadata = {
                    duration: duration,
                    sample_rate: 16000,
                    format: 'wav',
                    language: 'fa',
                    timestamp: Date.now()
                };
                
                this.websocket_client.send_audio_data(processedBlob, metadata);
                this.showSuccess('Audio sent for processing');
            } else {
                console.warn('⚠️ No WebSocket connection available');
                this.showError('No connection to server');
            }
            
        } catch (error) {
            console.error('❌ Failed to process audio:', error);
            this.showError('Failed to process recorded audio');
        }
    }
    
    /**
     * Convert audio to WAV format
     * @param {Blob} audioBlob - Original audio blob
     * @returns {Promise<Blob>} WAV audio blob
     */
    async convertToWav(audioBlob) {
        try {
            // If already WAV, return as is
            if (audioBlob.type.includes('wav')) {
                return audioBlob;
            }
            
            // Create audio context for conversion
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const arrayBuffer = await audioBlob.arrayBuffer();
            const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
            
            // Convert to WAV format
            const wavBuffer = this.audioBufferToWav(audioBuffer);
            return new Blob([wavBuffer], { type: 'audio/wav' });
            
        } catch (error) {
            console.warn('⚠️ Failed to convert to WAV, using original format:', error);
            return audioBlob;
        }
    }
    
    /**
     * Convert AudioBuffer to WAV format
     * @param {AudioBuffer} audioBuffer - Audio buffer
     * @returns {ArrayBuffer} WAV format buffer
     */
    audioBufferToWav(audioBuffer) {
        const numChannels = 1; // Force mono
        const sampleRate = 16000; // Force 16kHz
        const length = audioBuffer.length;
        const buffer = new ArrayBuffer(44 + length * 2);
        const view = new DataView(buffer);
        
        // WAV header
        this.writeString(view, 0, 'RIFF');
        view.setUint32(4, 36 + length * 2, true);
        this.writeString(view, 8, 'WAVE');
        this.writeString(view, 12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, numChannels, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * 2, true);
        view.setUint16(32, 2, true);
        view.setUint16(34, 16, true);
        this.writeString(view, 36, 'data');
        view.setUint32(40, length * 2, true);
        
        // Convert audio data
        const channelData = audioBuffer.getChannelData(0);
        let offset = 44;
        for (let i = 0; i < length; i++) {
            const sample = Math.max(-1, Math.min(1, channelData[i]));
            view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
            offset += 2;
        }
        
        return buffer;
    }
    
    /**
     * Write string to DataView
     * @param {DataView} view - DataView instance
     * @param {number} offset - Byte offset
     * @param {string} string - String to write
     */
    writeString(view, offset, string) {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    }
    
    /**
     * Get supported MIME type
     * @returns {string} Supported MIME type
     */
    getSupportedMimeType() {
        const types = [
            'audio/webm;codecs=opus',
            'audio/webm',
            'audio/ogg;codecs=opus',
            'audio/ogg',
            'audio/wav',
            'audio/mp4'
        ];
        
        for (const type of types) {
            if (MediaRecorder.isTypeSupported(type)) {
                return type;
            }
        }
        
        return 'audio/webm'; // Fallback
    }
    
    /**
     * Get recording duration in seconds
     * @returns {number} Duration in seconds
     */
    getRecordingDuration() {
        if (!this.startTime) return 0;
        return (Date.now() - this.startTime) / 1000;
    }
    
    /**
     * Start recording timer
     */
    startTimer() {
        if (this.timerInterval) {
            clearInterval(this.timerInterval);
        }
        
        this.timerInterval = setInterval(() => {
            if (this.recordingTimer) {
                const duration = this.getRecordingDuration();
                this.recordingTimer.textContent = this.formatDuration(duration);
            }
        }, 100);
    }
    
    /**
     * Stop recording timer
     */
    stopTimer() {
        if (this.timerInterval) {
            clearInterval(this.timerInterval);
            this.timerInterval = null;
        }
    }
    
    /**
     * Format duration for display
     * @param {number} seconds - Duration in seconds
     * @returns {string} Formatted duration
     */
    formatDuration(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }
    
    /**
     * Update UI state
     */
    updateUI() {
        if (this.recordButton) {
            if (this.isRecording) {
                this.recordButton.textContent = '🛑 Stop Recording';
                this.recordButton.classList.add('recording');
            } else {
                this.recordButton.textContent = '🎤 Start Recording';
                this.recordButton.classList.remove('recording');
            }
        }
        
        if (this.recordingStatus) {
            if (this.isRecording) {
                this.recordingStatus.textContent = 'Recording...';
                this.recordingStatus.classList.add('active');
            } else {
                this.recordingStatus.textContent = 'Ready to record';
                this.recordingStatus.classList.remove('active');
            }
        }
        
        if (this.recordingTimer && !this.isRecording) {
            this.recordingTimer.textContent = '0:00';
        }
    }
    
    /**
     * Show success message
     * @param {string} message - Success message
     */
    showSuccess(message) {
        this.showMessage(message, 'success');
    }
    
    /**
     * Show error message
     * @param {string} message - Error message
     */
    showError(message) {
        this.showMessage(message, 'error');
    }
    
    /**
     * Show message
     * @param {string} message - Message text
     * @param {string} type - Message type
     */
    showMessage(message, type = 'info') {
        // Create or find message container
        let messageContainer = document.getElementById('audio-messages');
        if (!messageContainer) {
            messageContainer = document.createElement('div');
            messageContainer.id = 'audio-messages';
            messageContainer.className = 'audio-messages';
            
            // Insert after record button
            const recordButton = document.getElementById('record-button');
            if (recordButton) {
                recordButton.parentNode.insertBefore(messageContainer, recordButton.nextSibling);
            } else {
                document.body.appendChild(messageContainer);
            }
        }
        
        // Create message element
        const messageElement = document.createElement('div');
        messageElement.className = `audio-message ${type}`;
        messageElement.innerHTML = `
            <span class="message-text">${message}</span>
            <button class="message-close" onclick="this.parentElement.remove()">×</button>
        `;
        
        messageContainer.appendChild(messageElement);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (messageElement.parentElement) {
                messageElement.remove();
            }
        }, 5000);
    }
    
    /**
     * Check if browser supports audio recording
     * @returns {boolean} Support status
     */
    static isSupported() {
        return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia && window.MediaRecorder);
    }
    
    /**
     * Request microphone permissions
     * @returns {Promise<boolean>} Permission granted
     */
    static async requestPermissions() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            stream.getTracks().forEach(track => track.stop());
            return true;
        } catch (error) {
            console.error('❌ Microphone permission denied:', error);
            return false;
        }
    }
    
    /**
     * Cleanup resources
     */
    cleanup() {
        if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
            this.mediaRecorder.stop();
        }
        
        if (this.audioStream) {
            this.audioStream.getTracks().forEach(track => track.stop());
        }
        
        this.stopTimer();
    }
}

// Initialize audio recorder when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    if (AudioRecorder.isSupported()) {
        window.audioRecorder = new AudioRecorder();
    } else {
        console.error('❌ Audio recording not supported in this browser');
        
        // Show fallback message
        const recordButton = document.getElementById('record-button');
        if (recordButton) {
            recordButton.disabled = true;
            recordButton.textContent = 'Audio recording not supported';
        }
    }
}); 