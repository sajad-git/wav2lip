/**
 * Avatar Uploader - Handles avatar upload and registration
 */

class AvatarUploader {
    constructor() {
        this.upload_area = null;
        this.file_validator = new FileValidator();
        this.upload_progress = new ProgressTracker();
        this.preview_manager = new PreviewManager();
        this.current_file = null;
        
        this.initialize_upload_interface();
    }
    
    /**
     * Initialize drag-and-drop upload interface
     */
    initialize_upload_interface() {
        this.upload_area = document.getElementById('upload-area');
        if (!this.upload_area) {
            console.error('❌ Upload area not found');
            return;
        }
        
        // Setup drag and drop events
        this.upload_area.addEventListener('dragover', this.handleDragOver.bind(this));
        this.upload_area.addEventListener('dragleave', this.handleDragLeave.bind(this));
        this.upload_area.addEventListener('drop', this.handleDrop.bind(this));
        this.upload_area.addEventListener('click', this.openFileDialog.bind(this));
        
        // Setup file input
        const fileInput = document.getElementById('file-input');
        if (fileInput) {
            fileInput.addEventListener('change', this.handleFileSelect.bind(this));
        }
        
        // Setup upload button
        const uploadButton = document.getElementById('upload-button');
        if (uploadButton) {
            uploadButton.addEventListener('click', this.handleUpload.bind(this));
            uploadButton.disabled = true;
        }
        
        console.log('✅ Avatar uploader initialized');
    }
    
    /**
     * Handle drag over event
     * @param {DragEvent} event - Drag event
     */
    handleDragOver(event) {
        event.preventDefault();
        this.upload_area.classList.add('dragover');
    }
    
    /**
     * Handle drag leave event
     * @param {DragEvent} event - Drag event
     */
    handleDragLeave(event) {
        event.preventDefault();
        this.upload_area.classList.remove('dragover');
    }
    
    /**
     * Handle file drop
     * @param {DragEvent} event - Drop event
     */
    handleDrop(event) {
        event.preventDefault();
        this.upload_area.classList.remove('dragover');
        
        const files = event.dataTransfer.files;
        this.handle_file_drop(files);
    }
    
    /**
     * Open file dialog
     */
    openFileDialog() {
        const fileInput = document.getElementById('file-input');
        if (fileInput) {
            fileInput.click();
        }
    }
    
    /**
     * Handle file selection from input
     * @param {Event} event - File input change event
     */
    handleFileSelect(event) {
        const files = event.target.files;
        this.handle_file_drop(files);
    }
    
    /**
     * Process dropped/selected files
     * @param {FileList} files - Selected files
     */
    handle_file_drop(files) {
        if (files.length === 0) return;
        
        const file = files[0];
        console.log('📁 File selected:', file.name);
        
        // Validate file
        const validation = this.file_validator.validate_avatar_file(file);
        
        if (!validation.is_valid) {
            this.showValidationErrors(validation.errors);
            return;
        }
        
        if (validation.warnings.length > 0) {
            this.showValidationWarnings(validation.warnings);
        }
        
        this.current_file = file;
        this.preview_avatar_with_face_detection(file);
        this.enableUploadButton();
    }
    
    /**
     * Preview avatar with face detection
     * @param {File} file - Avatar file
     */
    async preview_avatar_with_face_detection(file) {
        try {
            console.log('🔍 Generating preview with face detection...');
            
            // Show loading state
            this.preview_manager.showLoadingState();
            
            // Create preview for image/video
            if (file.type.startsWith('image/')) {
                await this.preview_manager.previewImage(file);
            } else if (file.type.startsWith('video/')) {
                await this.preview_manager.previewVideo(file);
            }
            
            // Send to server for face detection preview
            const formData = new FormData();
            formData.append('file', file);
            formData.append('preview_only', 'true');
            
            const response = await fetch('/avatar/preview', {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                const result = await response.json();
                this.preview_manager.showFaceDetectionResults(result);
            } else {
                console.warn('⚠️ Face detection preview failed');
                this.preview_manager.showPreviewOnly();
            }
            
        } catch (error) {
            console.error('❌ Error generating preview:', error);
            this.preview_manager.showError('Preview generation failed');
        }
    }
    
    /**
     * Handle upload button click
     */
    async handleUpload() {
        if (!this.current_file) {
            console.error('❌ No file selected');
            return;
        }
        
        const metadata = this.getAvatarMetadata();
        if (!this.validateMetadata(metadata)) {
            return;
        }
        
        try {
            const result = await this.upload_avatar(this.current_file, metadata);
            this.handleUploadResult(result);
        } catch (error) {
            console.error('❌ Upload failed:', error);
            this.showUploadError(error.message);
        }
    }
    
    /**
     * Upload avatar to server
     * @param {File} file - Avatar file
     * @param {Object} metadata - Avatar metadata
     * @returns {Promise<Object>} Upload result
     */
    async upload_avatar(file, metadata) {
        console.log('📤 Uploading avatar:', file.name);
        
        // Show upload progress
        this.upload_progress.show();
        
        const formData = new FormData();
        formData.append('file', file);
        formData.append('avatar_name', metadata.name);
        formData.append('user_id', metadata.user_id);
        
        if (metadata.description) {
            formData.append('description', metadata.description);
        }
        
        if (metadata.tags && metadata.tags.length > 0) {
            formData.append('tags', JSON.stringify(metadata.tags));
        }
        
        // Create XMLHttpRequest for progress tracking
        return new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();
            
            // Track upload progress
            xhr.upload.onprogress = (event) => {
                if (event.lengthComputable) {
                    const progress = (event.loaded / event.total) * 100;
                    this.upload_progress.updateProgress(progress);
                }
            };
            
            xhr.onload = () => {
                this.upload_progress.hide();
                
                if (xhr.status === 200) {
                    try {
                        const result = JSON.parse(xhr.responseText);
                        resolve(result);
                    } catch (error) {
                        reject(new Error('Invalid response format'));
                    }
                } else {
                    reject(new Error(`Upload failed: ${xhr.status}`));
                }
            };
            
            xhr.onerror = () => {
                this.upload_progress.hide();
                reject(new Error('Network error during upload'));
            };
            
            xhr.open('POST', '/avatar/register');
            xhr.send(formData);
        });
    }
    
    /**
     * Get avatar metadata from form
     * @returns {Object} Avatar metadata
     */
    getAvatarMetadata() {
        return {
            name: document.getElementById('avatar-name')?.value || '',
            user_id: document.getElementById('user-id')?.value || 'default_user',
            description: document.getElementById('avatar-description')?.value || '',
            tags: this.getTagsFromInput()
        };
    }
    
    /**
     * Get tags from input field
     * @returns {Array<string>} Tags array
     */
    getTagsFromInput() {
        const tagsInput = document.getElementById('avatar-tags');
        if (!tagsInput || !tagsInput.value) return [];
        
        return tagsInput.value
            .split(',')
            .map(tag => tag.trim())
            .filter(tag => tag.length > 0);
    }
    
    /**
     * Validate metadata
     * @param {Object} metadata - Avatar metadata
     * @returns {boolean} Validation result
     */
    validateMetadata(metadata) {
        const errors = [];
        
        if (!metadata.name || metadata.name.trim().length === 0) {
            errors.push('Avatar name is required');
        }
        
        if (metadata.name && metadata.name.length > 50) {
            errors.push('Avatar name must be less than 50 characters');
        }
        
        if (metadata.description && metadata.description.length > 200) {
            errors.push('Description must be less than 200 characters');
        }
        
        if (errors.length > 0) {
            this.showValidationErrors(errors);
            return false;
        }
        
        return true;
    }
    
    /**
     * Handle upload result
     * @param {Object} result - Upload result from server
     */
    handleUploadResult(result) {
        if (result.registration_status === 'success') {
            this.showUploadSuccess(result);
            this.resetForm();
        } else {
            this.showUploadError(result.errors?.join(', ') || 'Upload failed');
        }
    }
    
    /**
     * Show upload success message
     * @param {Object} result - Upload result
     */
    showUploadSuccess(result) {
        const successDiv = document.getElementById('upload-success');
        if (successDiv) {
            successDiv.innerHTML = `
                <div class="success-message">
                    <h3>✅ Avatar Registered Successfully!</h3>
                    <p><strong>Avatar ID:</strong> ${result.avatar_id}</p>
                    <p><strong>Quality Score:</strong> ${(result.quality_assessment?.face_quality_score * 100).toFixed(1)}%</p>
                    <p><strong>Processing Time:</strong> ${result.processing_time?.toFixed(2)}s</p>
                    ${result.warnings?.length > 0 ? 
                        `<div class="warnings">
                            <h4>⚠️ Warnings:</h4>
                            <ul>${result.warnings.map(w => `<li>${w}</li>`).join('')}</ul>
                        </div>` : ''
                    }
                </div>
            `;
            successDiv.style.display = 'block';
            
            // Hide after 10 seconds
            setTimeout(() => {
                successDiv.style.display = 'none';
            }, 10000);
        }
    }
    
    /**
     * Show upload error message
     * @param {string} message - Error message
     */
    showUploadError(message) {
        const errorDiv = document.getElementById('upload-error');
        if (errorDiv) {
            errorDiv.innerHTML = `
                <div class="error-message">
                    <h3>❌ Upload Failed</h3>
                    <p>${message}</p>
                </div>
            `;
            errorDiv.style.display = 'block';
            
            // Hide after 10 seconds
            setTimeout(() => {
                errorDiv.style.display = 'none';
            }, 10000);
        }
    }
    
    /**
     * Show validation errors
     * @param {Array<string>} errors - Validation errors
     */
    showValidationErrors(errors) {
        const errorDiv = document.getElementById('validation-error');
        if (errorDiv) {
            errorDiv.innerHTML = `
                <div class="error-message">
                    <h3>❌ Validation Errors</h3>
                    <ul>${errors.map(error => `<li>${error}</li>`).join('')}</ul>
                </div>
            `;
            errorDiv.style.display = 'block';
        }
    }
    
    /**
     * Show validation warnings
     * @param {Array<string>} warnings - Validation warnings
     */
    showValidationWarnings(warnings) {
        const warningDiv = document.getElementById('validation-warning');
        if (warningDiv) {
            warningDiv.innerHTML = `
                <div class="warning-message">
                    <h3>⚠️ Warnings</h3>
                    <ul>${warnings.map(warning => `<li>${warning}</li>`).join('')}</ul>
                </div>
            `;
            warningDiv.style.display = 'block';
        }
    }
    
    /**
     * Enable upload button
     */
    enableUploadButton() {
        const uploadButton = document.getElementById('upload-button');
        if (uploadButton) {
            uploadButton.disabled = false;
            uploadButton.textContent = 'Upload Avatar';
        }
    }
    
    /**
     * Reset form after successful upload
     */
    resetForm() {
        this.current_file = null;
        
        // Reset file input
        const fileInput = document.getElementById('file-input');
        if (fileInput) {
            fileInput.value = '';
        }
        
        // Reset form fields
        const form = document.getElementById('avatar-form');
        if (form) {
            form.reset();
        }
        
        // Reset preview
        this.preview_manager.reset();
        
        // Disable upload button
        const uploadButton = document.getElementById('upload-button');
        if (uploadButton) {
            uploadButton.disabled = true;
            uploadButton.textContent = 'Select File First';
        }
        
        // Hide upload area highlight
        this.upload_area.classList.remove('file-selected');
    }
}

/**
 * File Validator - Validates avatar files
 */
class FileValidator {
    constructor() {
        this.supported_formats = new Set([
            'image/jpeg', 'image/jpg', 'image/png', 'image/gif',
            'video/mp4', 'video/mov', 'video/avi', 'video/webm'
        ]);
        this.max_file_size = 50 * 1024 * 1024; // 50MB
        this.min_dimensions = [64, 64];
        this.max_dimensions = [1920, 1080];
    }
    
    /**
     * Validate avatar file
     * @param {File} file - File to validate
     * @returns {Object} Validation result
     */
    validate_avatar_file(file) {
        const result = {
            is_valid: true,
            errors: [],
            warnings: []
        };
        
        // Check file size
        if (file.size > this.max_file_size) {
            result.errors.push(`File size (${this.formatFileSize(file.size)}) exceeds maximum allowed size (${this.formatFileSize(this.max_file_size)})`);
            result.is_valid = false;
        }
        
        // Check file format
        if (!this.supported_formats.has(file.type)) {
            result.errors.push(`File format '${file.type}' is not supported. Supported formats: ${Array.from(this.supported_formats).join(', ')}`);
            result.is_valid = false;
        }
        
        // Check file name
        if (file.name.length > 100) {
            result.warnings.push('File name is very long and will be truncated');
        }
        
        // Size warnings
        if (file.size > 10 * 1024 * 1024) { // 10MB
            result.warnings.push('Large file size may result in slower upload and processing');
        }
        
        return result;
    }
    
    /**
     * Format file size for display
     * @param {number} bytes - File size in bytes
     * @returns {string} Formatted size
     */
    formatFileSize(bytes) {
        const units = ['B', 'KB', 'MB', 'GB'];
        let size = bytes;
        let unitIndex = 0;
        
        while (size >= 1024 && unitIndex < units.length - 1) {
            size /= 1024;
            unitIndex++;
        }
        
        return `${size.toFixed(1)} ${units[unitIndex]}`;
    }
}

/**
 * Progress Tracker - Tracks upload progress
 */
class ProgressTracker {
    constructor() {
        this.progress_container = null;
        this.progress_bar = null;
        this.progress_text = null;
        
        this.initializeProgressElements();
    }
    
    initializeProgressElements() {
        this.progress_container = document.getElementById('upload-progress');
        this.progress_bar = document.getElementById('progress-bar');
        this.progress_text = document.getElementById('progress-text');
    }
    
    /**
     * Show progress indicator
     */
    show() {
        if (this.progress_container) {
            this.progress_container.style.display = 'block';
        }
    }
    
    /**
     * Hide progress indicator
     */
    hide() {
        if (this.progress_container) {
            this.progress_container.style.display = 'none';
        }
    }
    
    /**
     * Update progress
     * @param {number} progress - Progress percentage (0-100)
     */
    updateProgress(progress) {
        if (this.progress_bar) {
            this.progress_bar.style.width = `${progress}%`;
        }
        
        if (this.progress_text) {
            this.progress_text.textContent = `${Math.round(progress)}%`;
        }
    }
}

/**
 * Preview Manager - Handles avatar preview
 */
class PreviewManager {
    constructor() {
        this.preview_container = null;
        this.preview_image = null;
        this.preview_video = null;
        this.face_detection_overlay = null;
        
        this.initializePreviewElements();
    }
    
    initializePreviewElements() {
        this.preview_container = document.getElementById('avatar-preview');
        this.preview_image = document.getElementById('preview-image');
        this.preview_video = document.getElementById('preview-video');
        this.face_detection_overlay = document.getElementById('face-detection-overlay');
    }
    
    /**
     * Show loading state
     */
    showLoadingState() {
        if (this.preview_container) {
            this.preview_container.innerHTML = `
                <div class="loading-state">
                    <div class="spinner"></div>
                    <p>Generating preview...</p>
                </div>
            `;
            this.preview_container.style.display = 'block';
        }
    }
    
    /**
     * Preview image file
     * @param {File} file - Image file
     */
    async previewImage(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            
            reader.onload = (event) => {
                if (this.preview_container) {
                    this.preview_container.innerHTML = `
                        <div class="image-preview">
                            <img id="preview-image" src="${event.target.result}" alt="Avatar Preview" />
                            <div id="face-detection-overlay"></div>
                        </div>
                    `;
                    
                    // Re-initialize elements
                    this.initializePreviewElements();
                }
                resolve();
            };
            
            reader.onerror = reject;
            reader.readAsDataURL(file);
        });
    }
    
    /**
     * Preview video file
     * @param {File} file - Video file
     */
    async previewVideo(file) {
        return new Promise((resolve, reject) => {
            const url = URL.createObjectURL(file);
            
            if (this.preview_container) {
                this.preview_container.innerHTML = `
                    <div class="video-preview">
                        <video id="preview-video" src="${url}" controls muted></video>
                        <div id="face-detection-overlay"></div>
                    </div>
                `;
                
                // Re-initialize elements
                this.initializePreviewElements();
            }
            
            resolve();
        });
    }
    
    /**
     * Show face detection results
     * @param {Object} results - Face detection results
     */
    showFaceDetectionResults(results) {
        if (!this.face_detection_overlay) return;
        
        const faces = results.face_detection_summary?.bounding_boxes || [];
        const quality = results.quality_assessment?.face_quality_score || 0;
        
        let overlayHTML = `
            <div class="detection-info">
                <span class="face-count">Faces: ${faces.length}</span>
                <span class="quality-score" style="color: ${this.getQualityColor(quality)}">
                    Quality: ${(quality * 100).toFixed(1)}%
                </span>
            </div>
        `;
        
        // Add bounding boxes
        faces.forEach((box, index) => {
            overlayHTML += `
                <div class="face-box" style="
                    left: ${box[0]}px;
                    top: ${box[1]}px;
                    width: ${box[2] - box[0]}px;
                    height: ${box[3] - box[1]}px;
                    border: 2px solid ${this.getQualityColor(quality)};
                ">
                    <span class="face-label">Face ${index + 1}</span>
                </div>
            `;
        });
        
        this.face_detection_overlay.innerHTML = overlayHTML;
    }
    
    /**
     * Show preview without face detection
     */
    showPreviewOnly() {
        if (this.face_detection_overlay) {
            this.face_detection_overlay.innerHTML = `
                <div class="detection-info">
                    <span class="no-detection">Face detection unavailable</span>
                </div>
            `;
        }
    }
    
    /**
     * Show error in preview
     * @param {string} message - Error message
     */
    showError(message) {
        if (this.preview_container) {
            this.preview_container.innerHTML = `
                <div class="error-state">
                    <p>❌ ${message}</p>
                </div>
            `;
        }
    }
    
    /**
     * Reset preview
     */
    reset() {
        if (this.preview_container) {
            this.preview_container.innerHTML = '';
            this.preview_container.style.display = 'none';
        }
    }
    
    /**
     * Get color based on quality score
     * @param {number} quality - Quality score (0-1)
     * @returns {string} Color code
     */
    getQualityColor(quality) {
        if (quality >= 0.8) return '#00ff00';      // Green - excellent
        if (quality >= 0.6) return '#ffff00';      // Yellow - good
        if (quality >= 0.4) return '#ff8800';      // Orange - fair
        return '#ff0000';                          // Red - poor
    }
}

// Initialize avatar uploader when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.avatarUploader = new AvatarUploader();
}); 