/**
 * Avatar WebSocket Client - Handles WebSocket communication with avatar selection
 */

class AvatarWebSocketClient {
    constructor() {
        this.websocket = null;
        this.connection_state = 'disconnected';
        this.message_queue = [];
        this.selected_avatar_id = null;
        this.reconnection_manager = new ReconnectionManager();
        this.event_handlers = {};
        
        this.setupEventHandlers();
    }
    
    setupEventHandlers() {
        this.event_handlers = {
            open: [],
            close: [],
            error: [],
            message: [],
            avatar_selected: [],
            chunk_received: []
        };
    }
    
    /**
     * Connect to WebSocket server
     * @param {string} url - WebSocket server URL
     * @returns {Promise<boolean>} Connection success
     */
    async connect_to_server(url) {
        return new Promise((resolve, reject) => {
            try {
                this.websocket = new WebSocket(url);
                
                this.websocket.onopen = (event) => {
                    console.log('🔗 WebSocket connected');
                    this.connection_state = 'connected';
                    this.startHeartbeat();
                    this.requestAvailableAvatars();
                    this.triggerEvent('open', event);
                    resolve(true);
                };
                
                this.websocket.onclose = (event) => {
                    console.log('🔌 WebSocket disconnected');
                    this.connection_state = 'disconnected';
                    this.stopHeartbeat();
                    this.triggerEvent('close', event);
                    
                    // Attempt reconnection if not intentional
                    if (!event.wasClean) {
                        this.reconnection_manager.attemptReconnection(() => {
                            return this.connect_to_server(url);
                        });
                    }
                };
                
                this.websocket.onerror = (error) => {
                    console.error('❌ WebSocket error:', error);
                    this.connection_state = 'error';
                    this.triggerEvent('error', error);
                    reject(error);
                };
                
                this.websocket.onmessage = (event) => {
                    this.handleMessage(event);
                };
                
                // Connection timeout
                setTimeout(() => {
                    if (this.connection_state !== 'connected') {
                        this.websocket.close();
                        reject(new Error('Connection timeout'));
                    }
                }, 10000);
                
            } catch (error) {
                console.error('❌ Failed to create WebSocket:', error);
                reject(error);
            }
        });
    }
    
    /**
     * Select avatar for processing
     * @param {string} avatar_id - Avatar identifier
     * @returns {Promise<boolean>} Selection success
     */
    async select_avatar(avatar_id) {
        return new Promise((resolve, reject) => {
            if (!this.isConnected()) {
                reject(new Error('WebSocket not connected'));
                return;
            }
            
            const message = {
                type: 'avatar_selection',
                avatar_id: avatar_id,
                timestamp: Date.now()
            };
            
            // Set up response handler
            const responseHandler = (data) => {
                if (data.type === 'avatar_selection_response' && data.avatar_id === avatar_id) {
                    this.removeEventListener('message', responseHandler);
                    if (data.success) {
                        this.selected_avatar_id = avatar_id;
                        this.triggerEvent('avatar_selected', {avatar_id, data});
                        resolve(true);
                    } else {
                        reject(new Error(data.error || 'Avatar selection failed'));
                    }
                }
            };
            
            this.addEventListener('message', responseHandler);
            this.sendMessage(message);
            
            // Timeout after 5 seconds
            setTimeout(() => {
                this.removeEventListener('message', responseHandler);
                reject(new Error('Avatar selection timeout'));
            }, 5000);
        });
    }
    
    /**
     * Send audio data for processing
     * @param {Blob} audio_blob - Audio data
     * @param {Object} metadata - Audio metadata
     */
    send_audio_data(audio_blob, metadata = {}) {
        if (!this.isConnected()) {
            console.error('❌ Cannot send audio: WebSocket not connected');
            return;
        }
        
        if (!this.selected_avatar_id) {
            console.error('❌ Cannot send audio: No avatar selected');
            return;
        }
        
        const reader = new FileReader();
        reader.onload = () => {
            const message = {
                type: 'audio_data',
                audio_data: Array.from(new Uint8Array(reader.result)),
                avatar_id: this.selected_avatar_id,
                metadata: {
                    ...metadata,
                    timestamp: Date.now(),
                    sample_rate: metadata.sample_rate || 16000,
                    format: metadata.format || 'wav'
                }
            };
            
            this.sendMessage(message);
        };
        
        reader.readAsArrayBuffer(audio_blob);
    }
    
    /**
     * Send text data for processing
     * @param {string} text - Text content
     * @param {Object} options - Processing options
     */
    send_text_data(text, options = {}) {
        if (!this.isConnected()) {
            console.error('❌ Cannot send text: WebSocket not connected');
            return;
        }
        
        if (!this.selected_avatar_id) {
            console.error('❌ Cannot send text: No avatar selected');
            return;
        }
        
        const message = {
            type: 'text_data',
            text: text,
            avatar_id: this.selected_avatar_id,
            options: {
                ...options,
                language: options.language || 'fa',
                voice: options.voice || 'alloy'
            },
            timestamp: Date.now()
        };
        
        this.sendMessage(message);
    }
    
    /**
     * Handle incoming video chunk
     * @param {ArrayBuffer} chunk_data - Binary video chunk data
     */
    handle_video_chunk(chunk_data) {
        try {
            // Parse chunk metadata from first bytes
            const view = new DataView(chunk_data);
            const metadata_length = view.getUint32(0, true);
            const metadata_bytes = chunk_data.slice(4, 4 + metadata_length);
            const video_data = chunk_data.slice(4 + metadata_length);
            
            const metadata = JSON.parse(new TextDecoder().decode(metadata_bytes));
            
            const chunk = {
                chunk_id: metadata.chunk_id,
                video_data: video_data,
                metadata: metadata,
                timestamp: Date.now()
            };
            
            this.triggerEvent('chunk_received', chunk);
            
        } catch (error) {
            console.error('❌ Error processing video chunk:', error);
        }
    }
    
    /**
     * Handle incoming WebSocket messages
     * @param {MessageEvent} event - WebSocket message event
     */
    handleMessage(event) {
        try {
            if (event.data instanceof ArrayBuffer) {
                // Binary data (video chunks)
                this.handle_video_chunk(event.data);
            } else {
                // Text data (JSON messages)
                const data = JSON.parse(event.data);
                this.processMessage(data);
            }
        } catch (error) {
            console.error('❌ Error handling message:', error);
        }
    }
    
    /**
     * Process JSON messages
     * @param {Object} data - Parsed message data
     */
    processMessage(data) {
        switch (data.type) {
            case 'available_avatars':
                this.handleAvailableAvatars(data.avatars);
                break;
                
            case 'avatar_selection_response':
                // Handled by select_avatar method
                break;
                
            case 'processing_status':
                this.handleProcessingStatus(data);
                break;
                
            case 'error':
                this.handleError(data);
                break;
                
            case 'heartbeat_response':
                // Heartbeat acknowledged
                break;
                
            default:
                console.log('📨 Unknown message type:', data.type);
        }
        
        this.triggerEvent('message', data);
    }
    
    /**
     * Handle available avatars list
     * @param {Array} avatars - List of available avatars
     */
    handleAvailableAvatars(avatars) {
        console.log('👥 Available avatars:', avatars);
        this.triggerEvent('avatars_received', avatars);
    }
    
    /**
     * Handle processing status updates
     * @param {Object} status - Processing status data
     */
    handleProcessingStatus(status) {
        console.log('⚙️ Processing status:', status);
        this.triggerEvent('status_update', status);
    }
    
    /**
     * Handle error messages
     * @param {Object} error - Error data
     */
    handleError(error) {
        console.error('❌ Server error:', error);
        this.triggerEvent('error', error);
    }
    
    /**
     * Request available avatars from server
     */
    requestAvailableAvatars() {
        const message = {
            type: 'request_avatars',
            timestamp: Date.now()
        };
        this.sendMessage(message);
    }
    
    /**
     * Send message to server
     * @param {Object} message - Message to send
     */
    sendMessage(message) {
        if (!this.isConnected()) {
            this.message_queue.push(message);
            return;
        }
        
        try {
            this.websocket.send(JSON.stringify(message));
        } catch (error) {
            console.error('❌ Error sending message:', error);
            this.message_queue.push(message);
        }
    }
    
    /**
     * Start heartbeat mechanism
     */
    startHeartbeat() {
        this.heartbeat_interval = setInterval(() => {
            if (this.isConnected()) {
                this.sendMessage({
                    type: 'heartbeat',
                    timestamp: Date.now()
                });
            }
        }, 30000); // 30 seconds
    }
    
    /**
     * Stop heartbeat mechanism
     */
    stopHeartbeat() {
        if (this.heartbeat_interval) {
            clearInterval(this.heartbeat_interval);
            this.heartbeat_interval = null;
        }
    }
    
    /**
     * Check if WebSocket is connected
     * @returns {boolean} Connection status
     */
    isConnected() {
        return this.websocket && this.websocket.readyState === WebSocket.OPEN;
    }
    
    /**
     * Add event listener
     * @param {string} event - Event name
     * @param {Function} handler - Event handler
     */
    addEventListener(event, handler) {
        if (this.event_handlers[event]) {
            this.event_handlers[event].push(handler);
        }
    }
    
    /**
     * Remove event listener
     * @param {string} event - Event name
     * @param {Function} handler - Event handler to remove
     */
    removeEventListener(event, handler) {
        if (this.event_handlers[event]) {
            const index = this.event_handlers[event].indexOf(handler);
            if (index > -1) {
                this.event_handlers[event].splice(index, 1);
            }
        }
    }
    
    /**
     * Trigger event
     * @param {string} event - Event name
     * @param {any} data - Event data
     */
    triggerEvent(event, data) {
        if (this.event_handlers[event]) {
            this.event_handlers[event].forEach(handler => {
                try {
                    handler(data);
                } catch (error) {
                    console.error(`❌ Error in ${event} handler:`, error);
                }
            });
        }
    }
    
    /**
     * Disconnect WebSocket
     */
    disconnect() {
        if (this.websocket) {
            this.stopHeartbeat();
            this.websocket.close(1000, 'Client disconnect');
            this.connection_state = 'disconnected';
        }
    }
}

/**
 * Reconnection Manager - Handles automatic reconnection
 */
class ReconnectionManager {
    constructor() {
        this.max_retry_attempts = 5;
        this.retry_delay = 1000; // Start with 1 second
        this.exponential_backoff = true;
        this.connection_timeout = 10000;
        this.current_attempts = 0;
    }
    
    /**
     * Attempt reconnection with exponential backoff
     * @param {Function} reconnect_function - Function to call for reconnection
     */
    async attemptReconnection(reconnect_function) {
        if (this.current_attempts >= this.max_retry_attempts) {
            console.error('❌ Max reconnection attempts reached');
            return false;
        }
        
        this.current_attempts++;
        const delay = this.exponential_backoff 
            ? this.retry_delay * Math.pow(2, this.current_attempts - 1)
            : this.retry_delay;
        
        console.log(`🔄 Attempting reconnection ${this.current_attempts}/${this.max_retry_attempts} in ${delay}ms`);
        
        setTimeout(async () => {
            try {
                const success = await reconnect_function();
                if (success) {
                    console.log('✅ Reconnection successful');
                    this.current_attempts = 0;
                } else {
                    this.attemptReconnection(reconnect_function);
                }
            } catch (error) {
                console.error('❌ Reconnection failed:', error);
                this.attemptReconnection(reconnect_function);
            }
        }, delay);
    }
    
    /**
     * Reset reconnection attempts
     */
    reset() {
        this.current_attempts = 0;
    }
} 