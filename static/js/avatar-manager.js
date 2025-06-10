/**
 * Avatar Manager - Handles avatar selection and management
 */

class AvatarManager {
    constructor() {
        this.avatar_list = [];
        this.selected_avatar = null;
        this.api_client = new APIClient();
        this.ui_manager = new UIManager();
        this.websocket_client = null;
        
        this.initializeAvatarManager();
    }
    
    /**
     * Initialize avatar manager
     */
    async initializeAvatarManager() {
        try {
            await this.load_available_avatars();
            this.setupEventListeners();
            console.log('✅ Avatar manager initialized');
        } catch (error) {
            console.error('❌ Failed to initialize avatar manager:', error);
        }
    }
    
    /**
     * Set WebSocket client reference
     * @param {AvatarWebSocketClient} client - WebSocket client instance
     */
    setWebSocketClient(client) {
        this.websocket_client = client;
        
        // Listen for avatar-related events
        if (this.websocket_client) {
            this.websocket_client.addEventListener('avatars_received', this.handleAvatarsReceived.bind(this));
        }
    }
    
    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Refresh button
        const refreshButton = document.getElementById('refresh-avatars-btn');
        if (refreshButton) {
            refreshButton.addEventListener('click', this.refresh_avatar_list.bind(this));
        }
        
        // Upload new avatar button
        const uploadButton = document.getElementById('upload-new-avatar-btn');
        if (uploadButton) {
            uploadButton.addEventListener('click', this.showUploadDialog.bind(this));
        }
        
        // Avatar grid clicks
        document.addEventListener('click', (event) => {
            if (event.target.closest('.avatar-item')) {
                const avatarItem = event.target.closest('.avatar-item');
                const avatarId = avatarItem.dataset.avatarId;
                if (avatarId) {
                    this.selectAvatarFromGrid(avatarId);
                }
            }
        });
        
        // Delete avatar clicks
        document.addEventListener('click', (event) => {
            if (event.target.closest('.delete-avatar-btn')) {
                event.stopPropagation();
                const avatarId = event.target.closest('.avatar-item').dataset.avatarId;
                if (avatarId) {
                    this.confirmDeleteAvatar(avatarId);
                }
            }
        });
    }
    
    /**
     * Load available avatars from server
     * @returns {Promise<Array>} Avatar list
     */
    async load_available_avatars() {
        try {
            console.log('📥 Loading available avatars...');
            
            const avatars = await this.api_client.getAvatarList();
            this.avatar_list = avatars;
            
            // Update UI with loaded avatars
            this.ui_manager.render_avatar_grid(this.avatar_list);
            
            console.log(`✅ Loaded ${avatars.length} avatars`);
            return avatars;
            
        } catch (error) {
            console.error('❌ Failed to load avatars:', error);
            this.ui_manager.showError('Failed to load avatars');
            return [];
        }
    }
    
    /**
     * Select avatar for processing
     * @param {string} avatar_id - Avatar identifier
     * @returns {Promise<boolean>} Selection success
     */
    async select_avatar(avatar_id) {
        try {
            console.log('👤 Selecting avatar:', avatar_id);
            
            // Find avatar in list
            const avatar = this.avatar_list.find(a => a.avatar_id === avatar_id);
            if (!avatar) {
                throw new Error('Avatar not found');
            }
            
            // Check if avatar is ready for processing
            if (!avatar.processing_ready) {
                throw new Error('Avatar is not ready for processing');
            }
            
            // Select via WebSocket if available
            if (this.websocket_client) {
                const success = await this.websocket_client.select_avatar(avatar_id);
                if (success) {
                    this.selected_avatar = avatar;
                    this.ui_manager.highlight_selected_avatar(avatar_id);
                    this.updateSelectedAvatarInfo(avatar);
                    console.log('✅ Avatar selected successfully');
                    return true;
                }
            } else {
                // Direct selection without WebSocket
                this.selected_avatar = avatar;
                this.ui_manager.highlight_selected_avatar(avatar_id);
                this.updateSelectedAvatarInfo(avatar);
                return true;
            }
            
        } catch (error) {
            console.error('❌ Failed to select avatar:', error);
            this.ui_manager.showError(`Failed to select avatar: ${error.message}`);
            return false;
        }
    }
    
    /**
     * Select avatar from grid click
     * @param {string} avatar_id - Avatar identifier
     */
    async selectAvatarFromGrid(avatar_id) {
        await this.select_avatar(avatar_id);
    }
    
    /**
     * Delete avatar with confirmation
     * @param {string} avatar_id - Avatar identifier
     */
    async confirmDeleteAvatar(avatar_id) {
        const avatar = this.avatar_list.find(a => a.avatar_id === avatar_id);
        if (!avatar) return;
        
        const confirmed = confirm(`Are you sure you want to delete avatar "${avatar.name}"? This action cannot be undone.`);
        if (confirmed) {
            await this.delete_avatar(avatar_id);
        }
    }
    
    /**
     * Delete avatar
     * @param {string} avatar_id - Avatar identifier
     * @returns {Promise<boolean>} Deletion success
     */
    async delete_avatar(avatar_id) {
        try {
            console.log('🗑️ Deleting avatar:', avatar_id);
            
            const success = await this.api_client.deleteAvatar(avatar_id);
            
            if (success) {
                // Remove from local list
                this.avatar_list = this.avatar_list.filter(a => a.avatar_id !== avatar_id);
                
                // Clear selection if deleted avatar was selected
                if (this.selected_avatar?.avatar_id === avatar_id) {
                    this.selected_avatar = null;
                    this.updateSelectedAvatarInfo(null);
                }
                
                // Update UI
                this.ui_manager.render_avatar_grid(this.avatar_list);
                this.ui_manager.showSuccess('Avatar deleted successfully');
                
                console.log('✅ Avatar deleted successfully');
                return true;
            }
            
        } catch (error) {
            console.error('❌ Failed to delete avatar:', error);
            this.ui_manager.showError(`Failed to delete avatar: ${error.message}`);
            return false;
        }
    }
    
    /**
     * Refresh avatar list from server
     */
    async refresh_avatar_list() {
        console.log('🔄 Refreshing avatar list...');
        
        try {
            await this.load_available_avatars();
            
            // Maintain selection if avatar still exists
            if (this.selected_avatar) {
                const stillExists = this.avatar_list.find(a => a.avatar_id === this.selected_avatar.avatar_id);
                if (stillExists) {
                    this.ui_manager.highlight_selected_avatar(this.selected_avatar.avatar_id);
                } else {
                    this.selected_avatar = null;
                    this.updateSelectedAvatarInfo(null);
                }
            }
            
            this.ui_manager.showSuccess('Avatar list refreshed');
            
        } catch (error) {
            console.error('❌ Failed to refresh avatars:', error);
            this.ui_manager.showError('Failed to refresh avatar list');
        }
    }
    
    /**
     * Handle avatars received from WebSocket
     * @param {Array} avatars - Avatar list from WebSocket
     */
    handleAvatarsReceived(avatars) {
        console.log('📨 Received avatars from WebSocket:', avatars.length);
        this.avatar_list = avatars;
        this.ui_manager.render_avatar_grid(this.avatar_list);
    }
    
    /**
     * Show upload dialog
     */
    showUploadDialog() {
        // Navigate to upload page or show modal
        window.location.href = '/avatar-registration';
    }
    
    /**
     * Update selected avatar info display
     * @param {Object|null} avatar - Selected avatar or null
     */
    updateSelectedAvatarInfo(avatar) {
        const infoContainer = document.getElementById('selected-avatar-info');
        if (!infoContainer) return;
        
        if (avatar) {
            infoContainer.innerHTML = `
                <div class="selected-avatar-details">
                    <h3>Selected Avatar</h3>
                    <div class="avatar-preview">
                        <img src="/avatar/${avatar.avatar_id}/thumbnail" alt="${avatar.name}" 
                             onerror="this.src='/static/assets/default-avatar.png'" />
                    </div>
                    <div class="avatar-info">
                        <p><strong>Name:</strong> ${avatar.name}</p>
                        <p><strong>Quality:</strong> ${(avatar.face_quality_score * 100).toFixed(1)}%</p>
                        <p><strong>Format:</strong> ${avatar.file_format.toUpperCase()}</p>
                        <p><strong>Status:</strong> ${avatar.processing_ready ? '✅ Ready' : '⚠️ Processing'}</p>
                        <p><strong>Used:</strong> ${avatar.usage_count} times</p>
                    </div>
                </div>
            `;
            infoContainer.style.display = 'block';
        } else {
            infoContainer.innerHTML = `
                <div class="no-selection">
                    <p>No avatar selected</p>
                    <p>Choose an avatar from the grid above</p>
                </div>
            `;
        }
    }
    
    /**
     * Get currently selected avatar
     * @returns {Object|null} Selected avatar
     */
    getSelectedAvatar() {
        return this.selected_avatar;
    }
    
    /**
     * Check if an avatar is selected
     * @returns {boolean} Selection status
     */
    hasSelectedAvatar() {
        return this.selected_avatar !== null;
    }
}

/**
 * API Client - Handles server communication
 */
class APIClient {
    constructor() {
        this.base_url = '';
    }
    
    /**
     * Get avatar list from server
     * @returns {Promise<Array>} Avatar list
     */
    async getAvatarList() {
        const response = await fetch('/avatar/list');
        if (!response.ok) {
            throw new Error(`Failed to fetch avatars: ${response.status}`);
        }
        return await response.json();
    }
    
    /**
     * Get avatar info
     * @param {string} avatar_id - Avatar identifier
     * @returns {Promise<Object>} Avatar info
     */
    async getAvatarInfo(avatar_id) {
        const response = await fetch(`/avatar/${avatar_id}/info`);
        if (!response.ok) {
            throw new Error(`Failed to fetch avatar info: ${response.status}`);
        }
        return await response.json();
    }
    
    /**
     * Delete avatar
     * @param {string} avatar_id - Avatar identifier
     * @returns {Promise<boolean>} Deletion success
     */
    async deleteAvatar(avatar_id) {
        const response = await fetch(`/avatar/${avatar_id}`, {
            method: 'DELETE',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ user_id: 'default_user' })
        });
        
        if (!response.ok) {
            throw new Error(`Failed to delete avatar: ${response.status}`);
        }
        
        const result = await response.json();
        return result.success || false;
    }
}

/**
 * UI Manager - Handles user interface updates
 */
class UIManager {
    constructor() {
        this.avatar_grid = document.getElementById('avatar-grid');
        this.message_container = document.getElementById('message-container');
    }
    
    /**
     * Render avatar grid
     * @param {Array} avatars - Avatar list
     */
    render_avatar_grid(avatars) {
        if (!this.avatar_grid) {
            console.warn('⚠️ Avatar grid container not found');
            return;
        }
        
        if (avatars.length === 0) {
            this.avatar_grid.innerHTML = `
                <div class="no-avatars">
                    <p>No avatars available</p>
                    <button onclick="window.location.href='/avatar-registration'" class="upload-btn">
                        Upload Your First Avatar
                    </button>
                </div>
            `;
            return;
        }
        
        const gridHTML = avatars.map(avatar => `
            <div class="avatar-item" data-avatar-id="${avatar.avatar_id}">
                <div class="avatar-thumbnail">
                    <img src="/avatar/${avatar.avatar_id}/thumbnail" 
                         alt="${avatar.name}"
                         onerror="this.src='/static/assets/default-avatar.png'" />
                    <div class="avatar-overlay">
                        <div class="avatar-actions">
                            <button class="delete-avatar-btn" title="Delete Avatar">🗑️</button>
                        </div>
                    </div>
                </div>
                <div class="avatar-details">
                    <h4 class="avatar-name">${avatar.name}</h4>
                    <div class="avatar-meta">
                        <span class="quality-score" style="color: ${this.getQualityColor(avatar.face_quality_score)}">
                            ${(avatar.face_quality_score * 100).toFixed(0)}%
                        </span>
                        <span class="format-badge">${avatar.file_format.toUpperCase()}</span>
                    </div>
                    <div class="avatar-status">
                        ${avatar.processing_ready ? 
                            '<span class="status-ready">✅ Ready</span>' : 
                            '<span class="status-processing">⚠️ Processing</span>'
                        }
                    </div>
                </div>
            </div>
        `).join('');
        
        this.avatar_grid.innerHTML = gridHTML;
    }
    
    /**
     * Highlight selected avatar
     * @param {string} avatar_id - Avatar identifier
     */
    highlight_selected_avatar(avatar_id) {
        // Remove previous selection
        document.querySelectorAll('.avatar-item.selected').forEach(item => {
            item.classList.remove('selected');
        });
        
        // Add selection to current avatar
        const avatarItem = document.querySelector(`[data-avatar-id="${avatar_id}"]`);
        if (avatarItem) {
            avatarItem.classList.add('selected');
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
     * @param {string} type - Message type (success, error, warning)
     */
    showMessage(message, type = 'info') {
        if (!this.message_container) {
            // Create message container if it doesn't exist
            this.message_container = document.createElement('div');
            this.message_container.id = 'message-container';
            this.message_container.className = 'message-container';
            document.body.appendChild(this.message_container);
        }
        
        const messageElement = document.createElement('div');
        messageElement.className = `message ${type}`;
        messageElement.innerHTML = `
            <span class="message-text">${message}</span>
            <button class="message-close" onclick="this.parentElement.remove()">×</button>
        `;
        
        this.message_container.appendChild(messageElement);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (messageElement.parentElement) {
                messageElement.remove();
            }
        }, 5000);
    }
    
    /**
     * Get quality color based on score
     * @param {number} score - Quality score (0-1)
     * @returns {string} Color code
     */
    getQualityColor(score) {
        if (score >= 0.8) return '#00aa00';      // Green - excellent
        if (score >= 0.6) return '#ccaa00';      // Yellow - good
        if (score >= 0.4) return '#cc6600';      // Orange - fair
        return '#cc0000';                        // Red - poor
    }
}

// Initialize avatar manager when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.avatarManager = new AvatarManager();
}); 