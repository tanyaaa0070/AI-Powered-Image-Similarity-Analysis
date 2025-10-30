// DOM elements
const uploadArea1 = document.getElementById('uploadArea1');
const uploadArea2 = document.getElementById('uploadArea2');
const fileInput1 = document.getElementById('fileInput1');
const fileInput2 = document.getElementById('fileInput2');
const imagePreviews = document.getElementById('imagePreviews');
const preview1 = document.getElementById('preview1');
const preview2 = document.getElementById('preview2');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultsSection = document.getElementById('resultsSection');
const overallScore = document.getElementById('overallScore');
const objectScore = document.getElementById('objectScore');
const textureScore = document.getElementById('textureScore');
const colorScore = document.getElementById('colorScore');
const lowLevelProgress = document.getElementById('lowLevelProgress');
const midLevelProgress = document.getElementById('midLevelProgress');
const highLevelProgress = document.getElementById('highLevelProgress');
const lowLevelText = document.getElementById('lowLevelText');
const midLevelText = document.getElementById('midLevelText');
const highLevelText = document.getElementById('highLevelText');
const objectsList = document.getElementById('objectsList');
const processingTime = document.getElementById('processingTime');

// Store uploaded files
let uploadedFiles = {
    image1: null,
    image2: null
};

// Initialize event listeners
function init() {
    // File input event listeners
    fileInput1.addEventListener('change', (e) => handleFileSelect(e, 1));
    fileInput2.addEventListener('change', (e) => handleFileSelect(e, 2));
    
    // Drag and drop functionality for area 1
    setupDragAndDrop(uploadArea1, 1);
    setupDragAndDrop(uploadArea2, 2);
}

// Set up drag and drop for an upload area
function setupDragAndDrop(uploadArea, imageNumber) {
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });
    
    uploadArea.addEventListener('dragleave', function() {
        uploadArea.classList.remove('drag-over');
    });
    
    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        
        if (e.dataTransfer.files.length) {
            const file = e.dataTransfer.files[0];
            handleFileUpload(file, imageNumber);
        }
    });
    
    // Click to upload
    uploadArea.addEventListener('click', function() {
        const fileInput = imageNumber === 1 ? fileInput1 : fileInput2;
        fileInput.click();
    });
}

// Handle file selection from input
function handleFileSelect(event, imageNumber) {
    const file = event.target.files[0];
    if (file) {
        handleFileUpload(file, imageNumber);
    }
}

// Process uploaded file
function handleFileUpload(file, imageNumber) {
    // Validate file type
    if (!file.type.match('image.*')) {
        showError('Please upload an image file (PNG, JPG, JPEG, GIF, BMP)');
        return;
    }
    
    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
        showError('File size must be less than 10MB');
        return;
    }
    
    // Store file
    uploadedFiles[`image${imageNumber}`] = file;
    
    // Create preview
    const reader = new FileReader();
    reader.onload = function(e) {
        const preview = imageNumber === 1 ? preview1 : preview2;
        preview.src = e.target.result;
        
        // Show image previews if both images are uploaded
        if (uploadedFiles.image1 && uploadedFiles.image2) {
            imagePreviews.style.display = 'flex';
            resultsSection.style.display = 'none';
        }
    };
    reader.readAsDataURL(file);
}

// Analyze images
async function analyzeImages() {
    if (!uploadedFiles.image1 || !uploadedFiles.image2) {
        showError('Please upload two images first.');
        return;
    }
    
    // Show loading state
    analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
    analyzeBtn.disabled = true;
    document.body.classList.add('loading');
    
    try {
        // Create FormData
        const formData = new FormData();
        formData.append('image1', uploadedFiles.image1);
        formData.append('image2', uploadedFiles.image2);
        
        // Send request to backend
        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayResults(data);
        } else {
            showError(data.error || 'Analysis failed. Please try again.');
        }
    } catch (error) {
        console.error('Error:', error);
        showError('Network error. Please check your connection and try again.');
    } finally {
        // Reset button
        analyzeBtn.innerHTML = '<i class="fas fa-search"></i> Analyze Similarity';
        analyzeBtn.disabled = false;
        document.body.classList.remove('loading');
    }
}

// Display analysis results
function displayResults(data) {
    // Update overall score
    overallScore.textContent = `${data.overall_similarity}%`;
    
    // Update score circle
    const scoreCircle = document.querySelector('.score-circle');
    scoreCircle.style.background = `conic-gradient(var(--success-color) 0% ${data.overall_similarity}%, rgba(255, 255, 255, 0.2) ${data.overall_similarity}% 100%)`;
    
    // Update breakdown scores
    objectScore.textContent = `${data.breakdown.object_similarity}%`;
    textureScore.textContent = `${data.breakdown.texture_similarity}%`;
    colorScore.textContent = `${data.breakdown.color_similarity}%`;
    
    // Update layer contributions
    lowLevelProgress.style.width = `${data.layer_contributions.low_level}%`;
    midLevelProgress.style.width = `${data.layer_contributions.mid_level}%`;
    highLevelProgress.style.width = `${data.layer_contributions.high_level}%`;
    
    lowLevelText.textContent = `${data.layer_contributions.low_level}% contribution`;
    midLevelText.textContent = `${data.layer_contributions.mid_level}% contribution`;
    highLevelText.textContent = `${data.layer_contributions.high_level}% contribution`;
    
    // Update detected objects
    objectsList.innerHTML = '';
    data.detected_objects.forEach(obj => {
        const objectTag = document.createElement('div');
        objectTag.className = 'object-tag';
        objectTag.innerHTML = `
            ${obj.name} <span class="object-confidence">${obj.confidence}%</span>
        `;
        objectsList.appendChild(objectTag);
    });
    
    // Update processing time
    processingTime.textContent = data.processing_time;
    
    // Show results section
    resultsSection.style.display = 'block';
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// Load demo scenario
async function loadDemoScenario(scenario) {
    try {
        // Show loading state
        analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading Demo...';
        analyzeBtn.disabled = true;
        
        const response = await fetch(`/demo/${scenario}`);
        const data = await response.json();
        
        if (data.success) {
            // Update previews with placeholder images
            preview1.src = 'https://via.placeholder.com/400x300/4a6fa5/ffffff?text=Demo+Image+1';
            preview2.src = 'https://via.placeholder.com/400x300/6b8cbc/ffffff?text=Demo+Image+2';
            
            imagePreviews.style.display = 'flex';
            displayResults(data);
        } else {
            showError(data.error || 'Demo scenario failed to load.');
        }
    } catch (error) {
        console.error('Demo error:', error);
        showError('Failed to load demo scenario.');
    } finally {
        analyzeBtn.innerHTML = '<i class="fas fa-search"></i> Analyze Similarity';
        analyzeBtn.disabled = false;
    }
}

// Show error message
function showError(message) {
    // Remove existing error messages
    const existingErrors = document.querySelectorAll('.error-message');
    existingErrors.forEach(error => error.remove());
    
    // Create error message
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.innerHTML = `<i class="fas fa-exclamation-circle"></i> ${message}`;
    
    // Insert after the first card
    const firstCard = document.querySelector('.card');
    firstCard.parentNode.insertBefore(errorDiv, firstCard.nextSibling);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        errorDiv.remove();
    }, 5000);
}

// Show success message
function showSuccess(message) {
    // Remove existing success messages
    const existingSuccess = document.querySelectorAll('.success-message');
    existingSuccess.forEach(success => success.remove());
    
    // Create success message
    const successDiv = document.createElement('div');
    successDiv.className = 'success-message';
    successDiv.innerHTML = `<i class="fas fa-check-circle"></i> ${message}`;
    
    // Insert after the first card
    const firstCard = document.querySelector('.card');
    firstCard.parentNode.insertBefore(successDiv, firstCard.nextSibling);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        successDiv.remove();
    }, 5000);
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', init);

// Enhanced error handling in analyzeImages function
async function analyzeImages() {
    if (!uploadedFiles.image1 || !uploadedFiles.image2) {
        showError('Please upload two images first.');
        return;
    }
    
    // Show loading state
    analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
    analyzeBtn.disabled = true;
    document.body.classList.add('loading');
    
    try {
        // Create FormData
        const formData = new FormData();
        formData.append('image1', uploadedFiles.image1);
        formData.append('image2', uploadedFiles.image2);
        
        // Send request to backend
        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || `HTTP error! status: ${response.status}`);
        }
        
        if (data.success) {
            displayResults(data);
            showSuccess('Analysis completed successfully!');
        } else {
            showError(data.error || 'Analysis failed. Please try again.');
        }
    } catch (error) {
        console.error('Error:', error);
        if (error.name === 'TypeError' && error.message.includes('Failed to fetch')) {
            showError('Network error. Please check your connection and try again.');
        } else {
            showError(error.message || 'An unexpected error occurred. Please try again.');
        }
    } finally {
        // Reset button
        analyzeBtn.innerHTML = '<i class="fas fa-search"></i> Analyze Similarity';
        analyzeBtn.disabled = false;
        document.body.classList.remove('loading');
    }
}