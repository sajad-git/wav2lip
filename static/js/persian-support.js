/**
 * Persian Support - Utilities for Persian language support and RTL layout
 */

class PersianSupport {
    constructor() {
        this.persian_digits = '۰۱۲۳۴۵۶۷۸۹';
        this.english_digits = '0123456789';
        this.rtl_enabled = false;
        
        this.initializePersianSupport();
    }
    
    /**
     * Initialize Persian language support
     */
    initializePersianSupport() {
        this.setupRTLSupport();
        this.setupPersianInput();
        this.setupPersianKeyboard();
        console.log('✅ Persian support initialized');
    }
    
    /**
     * Setup RTL (Right-to-Left) support
     */
    setupRTLSupport() {
        // Auto-detect Persian text and apply RTL
        document.addEventListener('input', (event) => {
            if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') {
                this.handleTextDirection(event.target);
            }
        });
        
        // Setup RTL toggle button
        const rtlToggle = document.getElementById('rtl-toggle');
        if (rtlToggle) {
            rtlToggle.addEventListener('click', this.toggleRTL.bind(this));
        }
    }
    
    /**
     * Handle text direction based on content
     * @param {HTMLElement} element - Input element
     */
    handleTextDirection(element) {
        const text = element.value;
        const hasPersianText = this.containsPersianText(text);
        
        if (hasPersianText) {
            element.style.direction = 'rtl';
            element.style.textAlign = 'right';
            element.classList.add('persian-text');
        } else {
            element.style.direction = 'ltr';
            element.style.textAlign = 'left';
            element.classList.remove('persian-text');
        }
    }
    
    /**
     * Check if text contains Persian characters
     * @param {string} text - Text to check
     * @returns {boolean} Contains Persian text
     */
    containsPersianText(text) {
        const persianRange = /[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]/;
        return persianRange.test(text);
    }
    
    /**
     * Setup Persian input handling
     */
    setupPersianInput() {
        // Handle Persian number conversion
        document.addEventListener('input', (event) => {
            const element = event.target;
            if (element.classList.contains('persian-numbers')) {
                this.convertToPersianNumbers(element);
            }
        });
        
        // Handle text processing for Persian
        document.addEventListener('blur', (event) => {
            const element = event.target;
            if (element.classList.contains('persian-text') || this.containsPersianText(element.value)) {
                this.processPersianText(element);
            }
        });
    }
    
    /**
     * Convert English digits to Persian digits
     * @param {HTMLElement} element - Input element
     */
    convertToPersianNumbers(element) {
        let text = element.value;
        for (let i = 0; i < this.english_digits.length; i++) {
            const englishDigit = this.english_digits[i];
            const persianDigit = this.persian_digits[i];
            text = text.replace(new RegExp(englishDigit, 'g'), persianDigit);
        }
        element.value = text;
    }
    
    /**
     * Convert Persian digits to English digits
     * @param {string} text - Text with Persian digits
     * @returns {string} Text with English digits
     */
    convertToEnglishNumbers(text) {
        for (let i = 0; i < this.persian_digits.length; i++) {
            const persianDigit = this.persian_digits[i];
            const englishDigit = this.english_digits[i];
            text = text.replace(new RegExp(persianDigit, 'g'), englishDigit);
        }
        return text;
    }
    
    /**
     * Process Persian text for optimization
     * @param {HTMLElement} element - Text element
     */
    processPersianText(element) {
        let text = element.value;
        
        // Normalize Persian text
        text = this.normalizePersianText(text);
        
        // Fix Persian punctuation
        text = this.fixPersianPunctuation(text);
        
        // Update element value
        element.value = text;
    }
    
    /**
     * Normalize Persian text
     * @param {string} text - Persian text
     * @returns {string} Normalized text
     */
    normalizePersianText(text) {
        // Normalize Persian characters
        const normalizations = {
            'ي': 'ی',  // Arabic Yeh to Persian Yeh
            'ك': 'ک',  // Arabic Kaf to Persian Kaf
            'ئ': 'ی',  // Hamza on Yeh to Yeh (in some contexts)
            'ء': '',   // Remove standalone Hamza in some contexts
        };
        
        for (const [from, to] of Object.entries(normalizations)) {
            text = text.replace(new RegExp(from, 'g'), to);
        }
        
        return text;
    }
    
    /**
     * Fix Persian punctuation spacing
     * @param {string} text - Persian text
     * @returns {string} Fixed text
     */
    fixPersianPunctuation(text) {
        // Add space before Persian question mark and exclamation
        text = text.replace(/([^\s])([؟!])/g, '$1 $2');
        
        // Fix comma spacing
        text = text.replace(/([^\s])(،)/g, '$1$2 ');
        
        // Fix period spacing
        text = text.replace(/([^\s])(\.)/g, '$1$2 ');
        
        // Remove extra spaces
        text = text.replace(/\s+/g, ' ').trim();
        
        return text;
    }
    
    /**
     * Setup Persian keyboard shortcuts
     */
    setupPersianKeyboard() {
        document.addEventListener('keydown', (event) => {
            // Handle Persian keyboard shortcuts
            this.handlePersianShortcuts(event);
        });
    }
    
    /**
     * Handle Persian keyboard shortcuts
     * @param {KeyboardEvent} event - Keyboard event
     */
    handlePersianShortcuts(event) {
        // Ctrl + Shift + P: Toggle Persian keyboard
        if (event.ctrlKey && event.shiftKey && event.key === 'P') {
            event.preventDefault();
            this.togglePersianKeyboard();
        }
        
        // Ctrl + Shift + R: Toggle RTL
        if (event.ctrlKey && event.shiftKey && event.key === 'R') {
            event.preventDefault();
            this.toggleRTL();
        }
        
        // Ctrl + Shift + N: Convert numbers
        if (event.ctrlKey && event.shiftKey && event.key === 'N') {
            event.preventDefault();
            this.toggleNumberFormat();
        }
    }
    
    /**
     * Toggle Persian keyboard mode
     */
    togglePersianKeyboard() {
        const activeElement = document.activeElement;
        if (activeElement && (activeElement.tagName === 'INPUT' || activeElement.tagName === 'TEXTAREA')) {
            activeElement.classList.toggle('persian-keyboard');
            
            // Show notification
            this.showNotification(
                activeElement.classList.contains('persian-keyboard') 
                    ? 'Persian keyboard enabled' 
                    : 'Persian keyboard disabled'
            );
        }
    }
    
    /**
     * Toggle RTL layout
     */
    toggleRTL() {
        this.rtl_enabled = !this.rtl_enabled;
        document.body.classList.toggle('rtl-layout', this.rtl_enabled);
        
        // Update all text inputs
        const textInputs = document.querySelectorAll('input[type="text"], textarea');
        textInputs.forEach(input => {
            if (this.rtl_enabled || this.containsPersianText(input.value)) {
                input.style.direction = 'rtl';
                input.style.textAlign = 'right';
            } else {
                input.style.direction = 'ltr';
                input.style.textAlign = 'left';
            }
        });
        
        this.showNotification(this.rtl_enabled ? 'RTL layout enabled' : 'LTR layout enabled');
    }
    
    /**
     * Toggle number format between Persian and English
     */
    toggleNumberFormat() {
        const activeElement = document.activeElement;
        if (activeElement && (activeElement.tagName === 'INPUT' || activeElement.tagName === 'TEXTAREA')) {
            const text = activeElement.value;
            
            if (this.containsPersianDigits(text)) {
                activeElement.value = this.convertToEnglishNumbers(text);
                this.showNotification('Converted to English numbers');
            } else {
                this.convertToPersianNumbers(activeElement);
                this.showNotification('Converted to Persian numbers');
            }
        }
    }
    
    /**
     * Check if text contains Persian digits
     * @param {string} text - Text to check
     * @returns {boolean} Contains Persian digits
     */
    containsPersianDigits(text) {
        return /[۰-۹]/.test(text);
    }
    
    /**
     * Show notification to user
     * @param {string} message - Notification message
     */
    showNotification(message) {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = 'persian-notification';
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #333;
            color: white;
            padding: 10px 15px;
            border-radius: 5px;
            z-index: 10000;
            font-size: 14px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
            animation: slideInRight 0.3s ease-out;
        `;
        
        document.body.appendChild(notification);
        
        // Remove after 3 seconds
        setTimeout(() => {
            notification.style.animation = 'slideOutRight 0.3s ease-in';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 3000);
    }
    
    /**
     * Optimize text for Persian TTS
     * @param {string} text - Persian text
     * @returns {string} Optimized text
     */
    optimizeForTTS(text) {
        // Convert English numbers to Persian words
        text = this.convertNumbersToWords(text);
        
        // Fix pronunciation issues
        text = this.fixPronunciation(text);
        
        // Add proper pauses
        text = this.addPauses(text);
        
        return text;
    }
    
    /**
     * Convert numbers to Persian words
     * @param {string} text - Text with numbers
     * @returns {string} Text with word numbers
     */
    convertNumbersToWords(text) {
        const numberWords = {
            '0': 'صفر', '1': 'یک', '2': 'دو', '3': 'سه', '4': 'چهار',
            '5': 'پنج', '6': 'شش', '7': 'هفت', '8': 'هشت', '9': 'نه',
            '10': 'ده', '11': 'یازده', '12': 'دوازده', '13': 'سیزده',
            '14': 'چهارده', '15': 'پانزده', '16': 'شانزده', '17': 'هفده',
            '18': 'هجده', '19': 'نوزده', '20': 'بیست'
        };
        
        // Replace simple numbers
        for (const [num, word] of Object.entries(numberWords)) {
            text = text.replace(new RegExp(`\\b${num}\\b`, 'g'), word);
        }
        
        return text;
    }
    
    /**
     * Fix common Persian pronunciation issues
     * @param {string} text - Persian text
     * @returns {string} Fixed text
     */
    fixPronunciation(text) {
        // Add common pronunciation fixes for TTS
        const fixes = {
            'می‌خواهم': 'میخوام',
            'چطور': 'چطوری',
            'خیلی': 'خیلی خوب'
        };
        
        for (const [from, to] of Object.entries(fixes)) {
            text = text.replace(new RegExp(from, 'g'), to);
        }
        
        return text;
    }
    
    /**
     * Add appropriate pauses for natural speech
     * @param {string} text - Persian text
     * @returns {string} Text with pauses
     */
    addPauses(text) {
        // Add short pause after commas
        text = text.replace(/،/g, '، ');
        
        // Add longer pause after periods
        text = text.replace(/\./g, '. ');
        
        // Add pause after question marks
        text = text.replace(/؟/g, '؟ ');
        
        return text;
    }
    
    /**
     * Create Persian text input with proper styling
     * @param {HTMLElement} container - Container element
     * @param {Object} options - Input options
     */
    createPersianInput(container, options = {}) {
        const input = document.createElement(options.multiline ? 'textarea' : 'input');
        
        if (!options.multiline) {
            input.type = 'text';
        }
        
        // Apply Persian styling
        input.style.cssText = `
            direction: rtl;
            text-align: right;
            font-family: 'Vazir', 'Tahoma', sans-serif;
            font-size: 16px;
            padding: 10px;
            border: 2px solid #ddd;
            border-radius: 5px;
            width: 100%;
            box-sizing: border-box;
        `;
        
        input.className = 'persian-input';
        input.placeholder = options.placeholder || 'متن خود را وارد کنید...';
        
        // Add event listeners
        input.addEventListener('input', () => {
            this.handleTextDirection(input);
        });
        
        container.appendChild(input);
        return input;
    }
    
    /**
     * Format Persian text for display
     * @param {string} text - Persian text
     * @returns {string} Formatted text
     */
    formatPersianText(text) {
        // Apply Persian typography rules
        text = this.normalizePersianText(text);
        text = this.fixPersianPunctuation(text);
        
        // Apply proper line breaks for Persian text
        text = this.applyPersianLineBreaks(text);
        
        return text;
    }
    
    /**
     * Apply Persian line break rules
     * @param {string} text - Persian text
     * @returns {string} Text with proper line breaks
     */
    applyPersianLineBreaks(text) {
        // Don't break after certain characters
        const noBreakAfter = ['و', 'در', 'از', 'به', 'با'];
        
        noBreakAfter.forEach(word => {
            text = text.replace(new RegExp(`${word} `, 'g'), `${word}\u00A0`); // Non-breaking space
        });
        
        return text;
    }
}

// Initialize Persian support when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.persianSupport = new PersianSupport();
    
    // Add Persian support CSS
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideInRight {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        
        @keyframes slideOutRight {
            from { transform: translateX(0); opacity: 1; }
            to { transform: translateX(100%); opacity: 0; }
        }
        
        .persian-text {
            direction: rtl !important;
            text-align: right !important;
            font-family: 'Vazir', 'Tahoma', sans-serif !important;
        }
        
        .rtl-layout {
            direction: rtl;
        }
        
        .rtl-layout .container {
            direction: rtl;
        }
        
        .persian-keyboard {
            border-color: #4CAF50 !important;
            box-shadow: 0 0 5px rgba(76, 175, 80, 0.3) !important;
        }
        
        .persian-input {
            direction: rtl;
            text-align: right;
            font-family: 'Vazir', 'Tahoma', sans-serif;
        }
        
        .persian-notification {
            font-family: 'Vazir', 'Tahoma', sans-serif;
        }
    `;
    document.head.appendChild(style);
});

// Export for use in other modules
window.PersianSupport = PersianSupport; 