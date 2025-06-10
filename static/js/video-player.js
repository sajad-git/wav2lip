/**
 * Sequential Video Player - Handles video chunk playback with smooth streaming
 */

class SequentialVideoPlayer {
    constructor() {
        this.canvas = null;
        this.context = null;
        this.chunk_buffer = new ChunkBuffer();
        this.playback_state = new PlaybackState();
        this.timing_controller = new TimingController();
        this.is_playing = false;
        this.current_chunk = null;
        this.frame_index = 0;
        
        this.initializePlayer();
    }
    
    /**
     * Initialize video player
     */
    initializePlayer() {
        this.canvas = document.getElementById('video-canvas');
        if (!this.canvas) {
            console.error('❌ Video canvas not found');
            return;
        }
        
        this.context = this.canvas.getContext('2d');
        if (!this.context) {
            console.error('❌ Failed to get canvas context');
            return;
        }
        
        // Set default canvas size
        this.canvas.width = 512;
        this.canvas.height = 512;
        
        // Setup controls
        this.setupControls();
        
        console.log('✅ Video player initialized');
    }
    
    /**
     * Setup video controls
     */
    setupControls() {
        const playButton = document.getElementById('play-button');
        const pauseButton = document.getElementById('pause-button');
        const stopButton = document.getElementById('stop-button');
        
        if (playButton) {
            playButton.addEventListener('click', this.play.bind(this));
        }
        
        if (pauseButton) {
            pauseButton.addEventListener('click', this.pause.bind(this));
        }
        
        if (stopButton) {
            stopButton.addEventListener('click', this.stop.bind(this));
        }
    }
    
    /**
     * Display video chunk with proper timing
     * @param {Object} chunk - Video chunk with frames and timing
     */
    display_video_chunk(chunk) {
        try {
            // Validate chunk
            if (!chunk || !chunk.video_data || !chunk.metadata) {
                console.warn('⚠️ Invalid video chunk received');
                return;
            }
            
            console.log(`🎬 Displaying chunk: ${chunk.chunk_id}`);
            
            // Parse video data
            const frames = this.parseVideoData(chunk.video_data);
            if (!frames || frames.length === 0) {
                console.warn('⚠️ No frames in video chunk');
                return;
            }
            
            // Create video chunk object
            const videoChunk = {
                chunk_id: chunk.chunk_id,
                frames: frames,
                metadata: chunk.metadata,
                timestamp: chunk.timestamp || Date.now()
            };
            
            // Add to buffer
            this.chunk_buffer.addChunk(videoChunk);
            
            // Start playback if not already playing
            if (!this.is_playing) {
                this.startPlayback();
            }
            
        } catch (error) {
            console.error('❌ Error displaying video chunk:', error);
        }
    }
    
    /**
     * Parse video data into frames
     * @param {ArrayBuffer} videoData - Binary video data
     * @returns {Array} Array of frame data
     */
    parseVideoData(videoData) {
        try {
            // For now, assume the video data contains individual frame images
            // In a real implementation, this would decode video frames
            const frames = [];
            
            // Simple implementation: assume data contains multiple JPEG frames
            const view = new Uint8Array(videoData);
            let offset = 0;
            
            while (offset < view.length) {
                // Look for JPEG markers (0xFFD8)
                const jpegStart = this.findJpegStart(view, offset);
                if (jpegStart === -1) break;
                
                const jpegEnd = this.findJpegEnd(view, jpegStart);
                if (jpegEnd === -1) break;
                
                // Extract frame data
                const frameData = view.slice(jpegStart, jpegEnd + 2);
                const blob = new Blob([frameData], { type: 'image/jpeg' });
                const url = URL.createObjectURL(blob);
                
                frames.push({
                    data: url,
                    timestamp: Date.now() + (frames.length * 40) // 25 FPS = 40ms per frame
                });
                
                offset = jpegEnd + 2;
            }
            
            return frames;
            
        } catch (error) {
            console.error('❌ Error parsing video data:', error);
            return [];
        }
    }
    
    /**
     * Find JPEG start marker
     * @param {Uint8Array} data - Video data
     * @param {number} offset - Start offset
     * @returns {number} JPEG start position or -1
     */
    findJpegStart(data, offset) {
        for (let i = offset; i < data.length - 1; i++) {
            if (data[i] === 0xFF && data[i + 1] === 0xD8) {
                return i;
            }
        }
        return -1;
    }
    
    /**
     * Find JPEG end marker
     * @param {Uint8Array} data - Video data
     * @param {number} offset - Start offset
     * @returns {number} JPEG end position or -1
     */
    findJpegEnd(data, offset) {
        for (let i = offset; i < data.length - 1; i++) {
            if (data[i] === 0xFF && data[i + 1] === 0xD9) {
                return i;
            }
        }
        return -1;
    }
    
    /**
     * Start video playback
     */
    startPlayback() {
        if (this.is_playing) return;
        
        this.is_playing = true;
        this.playback_state.setState('playing');
        this.playNextFrame();
        
        console.log('▶️ Video playback started');
    }
    
    /**
     * Play next frame
     */
    async playNextFrame() {
        if (!this.is_playing) return;
        
        try {
            // Get next frame from buffer
            const frame = this.chunk_buffer.getNextFrame();
            
            if (frame) {
                await this.renderFrame(frame);
                
                // Schedule next frame
                const frameDelay = this.timing_controller.getFrameDelay();
                setTimeout(() => {
                    this.playNextFrame();
                }, frameDelay);
                
            } else {
                // No more frames, check if more chunks are expected
                if (this.chunk_buffer.isComplete()) {
                    this.stopPlayback();
                } else {
                    // Wait for more chunks
                    setTimeout(() => {
                        this.playNextFrame();
                    }, 10);
                }
            }
            
        } catch (error) {
            console.error('❌ Error playing frame:', error);
            setTimeout(() => {
                this.playNextFrame();
            }, 40); // Continue with next frame
        }
    }
    
    /**
     * Render frame to canvas
     * @param {Object} frame - Frame data
     */
    async renderFrame(frame) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            
            img.onload = () => {
                try {
                    // Clear canvas
                    this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);
                    
                    // Calculate aspect ratio and positioning
                    const aspectRatio = img.width / img.height;
                    const canvasAspect = this.canvas.width / this.canvas.height;
                    
                    let drawWidth, drawHeight, drawX, drawY;
                    
                    if (aspectRatio > canvasAspect) {
                        // Image is wider than canvas
                        drawWidth = this.canvas.width;
                        drawHeight = this.canvas.width / aspectRatio;
                        drawX = 0;
                        drawY = (this.canvas.height - drawHeight) / 2;
                    } else {
                        // Image is taller than canvas
                        drawWidth = this.canvas.height * aspectRatio;
                        drawHeight = this.canvas.height;
                        drawX = (this.canvas.width - drawWidth) / 2;
                        drawY = 0;
                    }
                    
                    // Draw image
                    this.context.drawImage(img, drawX, drawY, drawWidth, drawHeight);
                    
                    // Update playback info
                    this.updatePlaybackInfo(frame);
                    
                    // Cleanup
                    URL.revokeObjectURL(frame.data);
                    
                    resolve();
                    
                } catch (error) {
                    reject(error);
                }
            };
            
            img.onerror = () => {
                reject(new Error('Failed to load frame image'));
            };
            
            img.src = frame.data;
        });
    }
    
    /**
     * Update playback information display
     * @param {Object} frame - Current frame
     */
    updatePlaybackInfo(frame) {
        const infoElement = document.getElementById('playback-info');
        if (infoElement) {
            const chunkInfo = this.chunk_buffer.getCurrentChunkInfo();
            infoElement.innerHTML = `
                <div class="playback-stats">
                    <span>Chunk: ${chunkInfo.chunk_id}</span>
                    <span>Frame: ${chunkInfo.frame_index + 1}/${chunkInfo.total_frames}</span>
                    <span>Buffer: ${this.chunk_buffer.getBufferLevel()}%</span>
                    <span>FPS: ${this.timing_controller.getCurrentFPS().toFixed(1)}</span>
                </div>
            `;
        }
    }
    
    /**
     * Maintain smooth playback
     */
    maintain_smooth_playback() {
        // Monitor buffer levels and adjust playback
        const bufferLevel = this.chunk_buffer.getBufferLevel();
        
        if (bufferLevel < 20) {
            // Buffer running low, slow down playback slightly
            this.timing_controller.adjustFrameRate(0.9);
        } else if (bufferLevel > 80) {
            // Buffer high, speed up playback slightly
            this.timing_controller.adjustFrameRate(1.1);
        } else {
            // Normal playback speed
            this.timing_controller.resetFrameRate();
        }
    }
    
    /**
     * Handle chunk buffer management
     */
    handle_chunk_buffer_management() {
        // Clean up old chunks
        this.chunk_buffer.cleanup();
        
        // Report buffer status to server if needed
        const bufferStatus = this.chunk_buffer.getBufferStatus();
        this.reportBufferStatus(bufferStatus);
    }
    
    /**
     * Report buffer status to server
     * @param {Object} status - Buffer status
     */
    reportBufferStatus(status) {
        // This would normally send buffer status via WebSocket
        // to help server optimize chunk delivery
        console.log('📊 Buffer status:', status);
    }
    
    /**
     * Play video
     */
    play() {
        if (!this.is_playing) {
            this.startPlayback();
        }
    }
    
    /**
     * Pause video
     */
    pause() {
        this.is_playing = false;
        this.playback_state.setState('paused');
        console.log('⏸️ Video playback paused');
    }
    
    /**
     * Stop video
     */
    stop() {
        this.stopPlayback();
    }
    
    /**
     * Stop video playback
     */
    stopPlayback() {
        this.is_playing = false;
        this.playback_state.setState('stopped');
        this.chunk_buffer.clear();
        this.clearCanvas();
        console.log('⏹️ Video playback stopped');
    }
    
    /**
     * Clear canvas
     */
    clearCanvas() {
        if (this.context) {
            this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);
            
            // Draw placeholder
            this.context.fillStyle = '#333';
            this.context.fillRect(0, 0, this.canvas.width, this.canvas.height);
            
            this.context.fillStyle = '#666';
            this.context.font = '24px Arial';
            this.context.textAlign = 'center';
            this.context.fillText('No Video', this.canvas.width / 2, this.canvas.height / 2);
        }
    }
    
    /**
     * Set canvas size
     * @param {number} width - Canvas width
     * @param {number} height - Canvas height
     */
    setCanvasSize(width, height) {
        if (this.canvas) {
            this.canvas.width = width;
            this.canvas.height = height;
        }
    }
    
    /**
     * Get playback state
     * @returns {string} Current playback state
     */
    getPlaybackState() {
        return this.playback_state.getState();
    }
    
    /**
     * Check if playing
     * @returns {boolean} Playing status
     */
    isPlaying() {
        return this.is_playing;
    }
}

/**
 * Chunk Buffer - Manages video chunk buffering
 */
class ChunkBuffer {
    constructor() {
        this.chunks = [];
        this.current_chunk_index = 0;
        this.current_frame_index = 0;
        this.buffer_size = 5; // Maximum chunks to buffer
        this.is_complete = false;
    }
    
    /**
     * Add chunk to buffer
     * @param {Object} chunk - Video chunk
     */
    addChunk(chunk) {
        // Insert chunk in correct position (by chunk_id or timestamp)
        let insertIndex = this.chunks.length;
        for (let i = 0; i < this.chunks.length; i++) {
            if (chunk.chunk_id < this.chunks[i].chunk_id) {
                insertIndex = i;
                break;
            }
        }
        
        this.chunks.splice(insertIndex, 0, chunk);
        
        // Limit buffer size
        while (this.chunks.length > this.buffer_size) {
            const removedChunk = this.chunks.shift();
            console.log(`🗑️ Removed old chunk: ${removedChunk.chunk_id}`);
        }
        
        console.log(`📦 Added chunk to buffer: ${chunk.chunk_id} (${this.chunks.length} chunks)`);
    }
    
    /**
     * Get next frame from buffer
     * @returns {Object|null} Next frame or null
     */
    getNextFrame() {
        if (this.current_chunk_index >= this.chunks.length) {
            return null;
        }
        
        const currentChunk = this.chunks[this.current_chunk_index];
        if (!currentChunk || !currentChunk.frames) {
            this.current_chunk_index++;
            this.current_frame_index = 0;
            return this.getNextFrame();
        }
        
        if (this.current_frame_index >= currentChunk.frames.length) {
            // Move to next chunk
            this.current_chunk_index++;
            this.current_frame_index = 0;
            return this.getNextFrame();
        }
        
        const frame = currentChunk.frames[this.current_frame_index];
        this.current_frame_index++;
        
        return frame;
    }
    
    /**
     * Get current chunk info
     * @returns {Object} Chunk information
     */
    getCurrentChunkInfo() {
        const currentChunk = this.chunks[this.current_chunk_index];
        return {
            chunk_id: currentChunk?.chunk_id || 'none',
            frame_index: this.current_frame_index,
            total_frames: currentChunk?.frames?.length || 0
        };
    }
    
    /**
     * Get buffer level percentage
     * @returns {number} Buffer level (0-100)
     */
    getBufferLevel() {
        const totalFrames = this.chunks.reduce((sum, chunk) => sum + (chunk.frames?.length || 0), 0);
        const remainingFrames = this.chunks.slice(this.current_chunk_index).reduce((sum, chunk, index) => {
            if (index === 0) {
                return sum + Math.max(0, (chunk.frames?.length || 0) - this.current_frame_index);
            }
            return sum + (chunk.frames?.length || 0);
        }, 0);
        
        return totalFrames > 0 ? (remainingFrames / totalFrames) * 100 : 0;
    }
    
    /**
     * Get buffer status
     * @returns {Object} Buffer status
     */
    getBufferStatus() {
        return {
            chunks_buffered: this.chunks.length,
            buffer_level: this.getBufferLevel(),
            current_chunk: this.current_chunk_index,
            current_frame: this.current_frame_index
        };
    }
    
    /**
     * Check if buffer is complete
     * @returns {boolean} Complete status
     */
    isComplete() {
        return this.is_complete;
    }
    
    /**
     * Mark buffer as complete
     */
    markComplete() {
        this.is_complete = true;
    }
    
    /**
     * Clear buffer
     */
    clear() {
        // Cleanup blob URLs
        this.chunks.forEach(chunk => {
            if (chunk.frames) {
                chunk.frames.forEach(frame => {
                    if (frame.data && frame.data.startsWith('blob:')) {
                        URL.revokeObjectURL(frame.data);
                    }
                });
            }
        });
        
        this.chunks = [];
        this.current_chunk_index = 0;
        this.current_frame_index = 0;
        this.is_complete = false;
    }
    
    /**
     * Cleanup old chunks
     */
    cleanup() {
        // Remove chunks that have been fully played
        const chunksToRemove = this.current_chunk_index;
        if (chunksToRemove > 0) {
            const removedChunks = this.chunks.splice(0, chunksToRemove);
            removedChunks.forEach(chunk => {
                if (chunk.frames) {
                    chunk.frames.forEach(frame => {
                        if (frame.data && frame.data.startsWith('blob:')) {
                            URL.revokeObjectURL(frame.data);
                        }
                    });
                }
            });
            
            this.current_chunk_index = 0;
        }
    }
}

/**
 * Playback State - Manages playback state
 */
class PlaybackState {
    constructor() {
        this.state = 'stopped'; // stopped, playing, paused
        this.start_time = null;
    }
    
    /**
     * Set playback state
     * @param {string} state - New state
     */
    setState(state) {
        this.state = state;
        if (state === 'playing') {
            this.start_time = Date.now();
        }
    }
    
    /**
     * Get current state
     * @returns {string} Current state
     */
    getState() {
        return this.state;
    }
    
    /**
     * Get playback duration
     * @returns {number} Duration in milliseconds
     */
    getPlaybackDuration() {
        return this.start_time ? Date.now() - this.start_time : 0;
    }
}

/**
 * Timing Controller - Manages frame timing
 */
class TimingController {
    constructor() {
        this.target_fps = 25;
        this.current_fps = 25;
        this.frame_rate_multiplier = 1.0;
        this.last_frame_time = 0;
    }
    
    /**
     * Get frame delay in milliseconds
     * @returns {number} Frame delay
     */
    getFrameDelay() {
        const baseDelay = 1000 / this.target_fps;
        return baseDelay / this.frame_rate_multiplier;
    }
    
    /**
     * Adjust frame rate
     * @param {number} multiplier - Rate multiplier
     */
    adjustFrameRate(multiplier) {
        this.frame_rate_multiplier = Math.max(0.5, Math.min(2.0, multiplier));
    }
    
    /**
     * Reset frame rate to normal
     */
    resetFrameRate() {
        this.frame_rate_multiplier = 1.0;
    }
    
    /**
     * Get current FPS
     * @returns {number} Current FPS
     */
    getCurrentFPS() {
        return this.target_fps * this.frame_rate_multiplier;
    }
}

// Initialize video player when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.videoPlayer = new SequentialVideoPlayer();
}); 