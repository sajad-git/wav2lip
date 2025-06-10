/**
 * Buffer Manager - Handles streaming buffer management for smooth playback
 */

class BufferManager {
    constructor() {
        this.audio_buffer = new StreamingBuffer('audio');
        this.video_buffer = new StreamingBuffer('video');
        this.sync_controller = new SyncController();
        this.buffer_monitor = new BufferMonitor();
        
        this.initializeBufferManager();
    }
    
    /**
     * Initialize buffer manager
     */
    initializeBufferManager() {
        this.setupBufferMonitoring();
        this.setupSyncControl();
        console.log('✅ Buffer manager initialized');
    }
    
    /**
     * Setup buffer monitoring
     */
    setupBufferMonitoring() {
        // Monitor buffer levels every 100ms
        setInterval(() => {
            this.updateBufferStatus();
        }, 100);
    }
    
    /**
     * Setup sync control
     */
    setupSyncControl() {
        this.sync_controller.setBuffers(this.audio_buffer, this.video_buffer);
    }
    
    /**
     * Add audio chunk to buffer
     * @param {Object} chunk - Audio chunk data
     */
    addAudioChunk(chunk) {
        this.audio_buffer.addChunk(chunk);
        this.checkSyncRequirement();
    }
    
    /**
     * Add video chunk to buffer
     * @param {Object} chunk - Video chunk data
     */
    addVideoChunk(chunk) {
        this.video_buffer.addChunk(chunk);
        this.checkSyncRequirement();
    }
    
    /**
     * Get next audio chunk
     * @returns {Object|null} Next audio chunk
     */
    getNextAudioChunk() {
        return this.audio_buffer.getNextChunk();
    }
    
    /**
     * Get next video chunk
     * @returns {Object|null} Next video chunk
     */
    getNextVideoChunk() {
        return this.video_buffer.getNextChunk();
    }
    
    /**
     * Check if sync is required
     */
    checkSyncRequirement() {
        if (this.sync_controller.needsSync()) {
            this.sync_controller.performSync();
        }
    }
    
    /**
     * Update buffer status
     */
    updateBufferStatus() {
        const status = {
            audio: this.audio_buffer.getStatus(),
            video: this.video_buffer.getStatus(),
            sync: this.sync_controller.getSyncStatus()
        };
        
        this.buffer_monitor.updateStatus(status);
        this.reportBufferStatus(status);
    }
    
    /**
     * Report buffer status to UI
     * @param {Object} status - Buffer status
     */
    reportBufferStatus(status) {
        const bufferInfoElement = document.getElementById('buffer-info');
        if (bufferInfoElement) {
            bufferInfoElement.innerHTML = `
                <div class="buffer-status">
                    <div class="buffer-item">
                        <span class="buffer-label">Audio:</span>
                        <div class="buffer-bar">
                            <div class="buffer-fill" style="width: ${status.audio.level}%"></div>
                        </div>
                        <span class="buffer-value">${status.audio.level.toFixed(0)}%</span>
                    </div>
                    <div class="buffer-item">
                        <span class="buffer-label">Video:</span>
                        <div class="buffer-bar">
                            <div class="buffer-fill" style="width: ${status.video.level}%"></div>
                        </div>
                        <span class="buffer-value">${status.video.level.toFixed(0)}%</span>
                    </div>
                    <div class="buffer-sync">
                        <span class="sync-label">Sync:</span>
                        <span class="sync-status ${status.sync.inSync ? 'in-sync' : 'out-of-sync'}">
                            ${status.sync.inSync ? '✅' : '⚠️'} 
                            ${status.sync.drift.toFixed(1)}ms
                        </span>
                    </div>
                </div>
            `;
        }
    }
    
    /**
     * Clear all buffers
     */
    clearBuffers() {
        this.audio_buffer.clear();
        this.video_buffer.clear();
        this.sync_controller.reset();
        console.log('🧹 Buffers cleared');
    }
    
    /**
     * Get buffer health status
     * @returns {Object} Buffer health information
     */
    getBufferHealth() {
        return {
            audio: this.audio_buffer.getHealth(),
            video: this.video_buffer.getHealth(),
            overall: this.calculateOverallHealth()
        };
    }
    
    /**
     * Calculate overall buffer health
     * @returns {string} Health status
     */
    calculateOverallHealth() {
        const audioHealth = this.audio_buffer.getHealth();
        const videoHealth = this.video_buffer.getHealth();
        
        if (audioHealth === 'healthy' && videoHealth === 'healthy') {
            return 'healthy';
        } else if (audioHealth === 'critical' || videoHealth === 'critical') {
            return 'critical';
        } else {
            return 'warning';
        }
    }
}

/**
 * Streaming Buffer - Base buffer implementation
 */
class StreamingBuffer {
    constructor(type) {
        this.type = type;
        this.chunks = [];
        this.max_size = type === 'video' ? 10 : 15; // Video needs less buffering
        this.current_index = 0;
        this.target_level = 70; // Target buffer level percentage
        this.critical_level = 20; // Critical buffer level
    }
    
    /**
     * Add chunk to buffer
     * @param {Object} chunk - Chunk data
     */
    addChunk(chunk) {
        // Insert chunk in correct position based on timestamp
        let insertIndex = this.chunks.length;
        for (let i = 0; i < this.chunks.length; i++) {
            if (chunk.timestamp < this.chunks[i].timestamp) {
                insertIndex = i;
                break;
            }
        }
        
        this.chunks.splice(insertIndex, 0, chunk);
        
        // Maintain buffer size
        while (this.chunks.length > this.max_size) {
            const removedChunk = this.chunks.shift();
            if (this.current_index > 0) {
                this.current_index--;
            }
            console.log(`🗑️ Removed old ${this.type} chunk: ${removedChunk.chunk_id}`);
        }
        
        console.log(`📦 Added ${this.type} chunk: ${chunk.chunk_id} (${this.chunks.length}/${this.max_size})`);
    }
    
    /**
     * Get next chunk from buffer
     * @returns {Object|null} Next chunk
     */
    getNextChunk() {
        if (this.current_index >= this.chunks.length) {
            return null;
        }
        
        const chunk = this.chunks[this.current_index];
        this.current_index++;
        
        return chunk;
    }
    
    /**
     * Peek at next chunk without consuming it
     * @returns {Object|null} Next chunk
     */
    peekNextChunk() {
        if (this.current_index >= this.chunks.length) {
            return null;
        }
        
        return this.chunks[this.current_index];
    }
    
    /**
     * Get buffer status
     * @returns {Object} Buffer status
     */
    getStatus() {
        const remaining = this.chunks.length - this.current_index;
        const level = (remaining / this.max_size) * 100;
        
        return {
            level: Math.max(0, level),
            remaining: remaining,
            total: this.chunks.length,
            capacity: this.max_size,
            current_index: this.current_index
        };
    }
    
    /**
     * Get buffer health
     * @returns {string} Health status
     */
    getHealth() {
        const status = this.getStatus();
        
        if (status.level >= this.target_level) {
            return 'healthy';
        } else if (status.level >= this.critical_level) {
            return 'warning';
        } else {
            return 'critical';
        }
    }
    
    /**
     * Clear buffer
     */
    clear() {
        this.chunks = [];
        this.current_index = 0;
    }
    
    /**
     * Get buffer size
     * @returns {number} Number of chunks in buffer
     */
    size() {
        return this.chunks.length;
    }
    
    /**
     * Check if buffer is empty
     * @returns {boolean} Empty status
     */
    isEmpty() {
        return this.current_index >= this.chunks.length;
    }
    
    /**
     * Get current playback position
     * @returns {number} Current position
     */
    getCurrentPosition() {
        return this.current_index;
    }
    
    /**
     * Seek to specific position
     * @param {number} position - Target position
     */
    seekTo(position) {
        this.current_index = Math.max(0, Math.min(position, this.chunks.length));
    }
}

/**
 * Sync Controller - Handles audio/video synchronization
 */
class SyncController {
    constructor() {
        this.audio_buffer = null;
        this.video_buffer = null;
        this.sync_threshold = 100; // 100ms sync threshold
        this.last_sync_check = 0;
        this.sync_drift = 0;
    }
    
    /**
     * Set buffer references
     * @param {StreamingBuffer} audioBuffer - Audio buffer
     * @param {StreamingBuffer} videoBuffer - Video buffer
     */
    setBuffers(audioBuffer, videoBuffer) {
        this.audio_buffer = audioBuffer;
        this.video_buffer = videoBuffer;
    }
    
    /**
     * Check if sync is needed
     * @returns {boolean} Sync requirement
     */
    needsSync() {
        if (!this.audio_buffer || !this.video_buffer) {
            return false;
        }
        
        const now = Date.now();
        if (now - this.last_sync_check < 500) { // Check every 500ms
            return false;
        }
        
        this.last_sync_check = now;
        this.calculateSyncDrift();
        
        return Math.abs(this.sync_drift) > this.sync_threshold;
    }
    
    /**
     * Calculate sync drift between audio and video
     */
    calculateSyncDrift() {
        const audioChunk = this.audio_buffer.peekNextChunk();
        const videoChunk = this.video_buffer.peekNextChunk();
        
        if (!audioChunk || !videoChunk) {
            this.sync_drift = 0;
            return;
        }
        
        this.sync_drift = videoChunk.timestamp - audioChunk.timestamp;
    }
    
    /**
     * Perform synchronization
     */
    performSync() {
        if (this.sync_drift > this.sync_threshold) {
            // Video is ahead, advance audio
            this.advanceBuffer(this.audio_buffer, this.sync_drift);
        } else if (this.sync_drift < -this.sync_threshold) {
            // Audio is ahead, advance video
            this.advanceBuffer(this.video_buffer, Math.abs(this.sync_drift));
        }
        
        console.log(`🔄 Sync performed: drift was ${this.sync_drift.toFixed(1)}ms`);
    }
    
    /**
     * Advance buffer to sync
     * @param {StreamingBuffer} buffer - Buffer to advance
     * @param {number} driftMs - Drift in milliseconds
     */
    advanceBuffer(buffer, driftMs) {
        // Simple approach: skip chunks until we're close to sync
        let advanced = 0;
        while (advanced < driftMs && !buffer.isEmpty()) {
            const chunk = buffer.getNextChunk();
            if (chunk) {
                advanced += chunk.duration || 40; // Assume 40ms chunk duration
            }
        }
    }
    
    /**
     * Get sync status
     * @returns {Object} Sync status
     */
    getSyncStatus() {
        return {
            inSync: Math.abs(this.sync_drift) <= this.sync_threshold,
            drift: this.sync_drift,
            threshold: this.sync_threshold
        };
    }
    
    /**
     * Reset sync controller
     */
    reset() {
        this.sync_drift = 0;
        this.last_sync_check = 0;
    }
}

/**
 * Buffer Monitor - Monitors buffer performance and health
 */
class BufferMonitor {
    constructor() {
        this.metrics = {
            underruns: 0,
            overruns: 0,
            sync_corrections: 0,
            average_buffer_level: 0
        };
        this.history = [];
        this.max_history = 100;
    }
    
    /**
     * Update buffer status
     * @param {Object} status - Current buffer status
     */
    updateStatus(status) {
        // Record status in history
        this.history.push({
            timestamp: Date.now(),
            audio_level: status.audio.level,
            video_level: status.video.level,
            sync_drift: status.sync.drift
        });
        
        // Maintain history size
        if (this.history.length > this.max_history) {
            this.history.shift();
        }
        
        // Update metrics
        this.updateMetrics(status);
    }
    
    /**
     * Update performance metrics
     * @param {Object} status - Current status
     */
    updateMetrics(status) {
        // Track underruns (buffer too low)
        if (status.audio.level < 20 || status.video.level < 20) {
            this.metrics.underruns++;
        }
        
        // Track overruns (buffer too high)
        if (status.audio.level > 90 || status.video.level > 90) {
            this.metrics.overruns++;
        }
        
        // Calculate average buffer level
        if (this.history.length > 0) {
            const totalAudio = this.history.reduce((sum, entry) => sum + entry.audio_level, 0);
            const totalVideo = this.history.reduce((sum, entry) => sum + entry.video_level, 0);
            this.metrics.average_buffer_level = (totalAudio + totalVideo) / (this.history.length * 2);
        }
    }
    
    /**
     * Get performance metrics
     * @returns {Object} Performance metrics
     */
    getMetrics() {
        return { ...this.metrics };
    }
    
    /**
     * Reset metrics
     */
    resetMetrics() {
        this.metrics = {
            underruns: 0,
            overruns: 0,
            sync_corrections: 0,
            average_buffer_level: 0
        };
        this.history = [];
    }
}

// Export for use in main application
window.BufferManager = BufferManager; 