document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    const form = document.getElementById('transaction-form');
    const resultSection = document.getElementById('result-section');
    const loading = document.getElementById('loading');
    
    // Prediction elements
    const ifResult = document.getElementById('if-result');
    const ifIcon = document.getElementById('if-icon');
    const ifPrediction = document.getElementById('if-prediction');
    
    const aeResult = document.getElementById('ae-result');
    const aeIcon = document.getElementById('ae-icon');
    const aePrediction = document.getElementById('ae-prediction');
    
    // MSE elements
    const mseBar = document.getElementById('mse-bar');
    const thresholdMarker = document.getElementById('threshold-marker');
    const mseValue = document.getElementById('mse-value');
    const thresholdValue = document.getElementById('threshold-value');
    
    // Transaction details
    const transactionJson = document.getElementById('transaction-json');
    
    // Create error message element
    const errorContainer = document.createElement('div');
    errorContainer.className = 'error-container';
    errorContainer.style.display = 'none';
    errorContainer.style.backgroundColor = '#f8d7da';
    errorContainer.style.color = '#721c24';
    errorContainer.style.padding = '15px';
    errorContainer.style.marginBottom = '20px';
    errorContainer.style.borderRadius = '5px';
    errorContainer.style.border = '1px solid #f5c6cb';
    document.querySelector('main').insertBefore(errorContainer, resultSection);
    
    // Handle form submission
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Hide any previous error messages
        errorContainer.style.display = 'none';
        
        // Hide results section if it was previously shown
        resultSection.style.display = 'none';
        
        // Show loading spinner
        loading.style.display = 'flex';
        
        // Get form data
        const formData = {
            amount: document.getElementById('amount').value,
            category: document.getElementById('category').value,
            gender: document.getElementById('gender').value,
            age: document.getElementById('age').value
        };
        
        // Send data to backend
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(errorData => {
                    throw new Error(errorData.error || 'Server error: ' + response.status);
                });
            }
            return response.json();
        })
        .then(data => {
            // Hide loading spinner
            loading.style.display = 'none';
            
            // Check if there's an error in the response
            if (data.error) {
                // Display error message
                errorContainer.textContent = 'Error: ' + data.error;
                errorContainer.style.display = 'block';
                return;
            }
            
            // Display results section
            resultSection.style.display = 'block';
            
            // Update Isolation Forest prediction
            updatePrediction(
                data.isolation_forest_prediction, 
                ifResult, 
                ifIcon, 
                ifPrediction
            );
            
            // Update Autoencoder prediction
            updatePrediction(
                data.autoencoder_prediction, 
                aeResult, 
                aeIcon, 
                aePrediction
            );
            
            // Update MSE meter
            updateMseMeter(
                data.reconstruction_error, 
                data.threshold, 
                mseBar, 
                thresholdMarker, 
                mseValue, 
                thresholdValue
            );
            
            // Update transaction details
            transactionJson.textContent = JSON.stringify(data.input_data, null, 2);
            
            // Scroll to results
            resultSection.scrollIntoView({ behavior: 'smooth' });
        })
        .catch(error => {
            // Hide loading spinner
            loading.style.display = 'none';
            
            // Show error message
            console.error('Error:', error);
            errorContainer.textContent = error.message || 'An unexpected error occurred. Please try again.';
            errorContainer.style.display = 'block';
            errorContainer.scrollIntoView({ behavior: 'smooth' });
        });
    });
    
    // Function to update prediction display
    function updatePrediction(prediction, resultElement, iconElement, predictionBox) {
        resultElement.textContent = prediction;
        
        if (prediction === 'Normal') {
            iconElement.innerHTML = '<i class="fas fa-check-circle"></i>';
            predictionBox.className = 'prediction-box normal';
        } else {
            iconElement.innerHTML = '<i class="fas fa-exclamation-triangle"></i>';
            predictionBox.className = 'prediction-box fraud';
        }
    }
    
    // Function to update MSE meter
    function updateMseMeter(mse, threshold, barElement, markerElement, valueElement, thresholdElement) {
        // Calculate max value for scaling (3x threshold or MSE, whichever is larger)
        const maxValue = Math.max(threshold * 3, mse * 1.2);
        
        // Update MSE bar width (as percentage of max)
        const percentage = (mse / maxValue) * 100;
        barElement.style.width = `${percentage}%`;
        
        // Update threshold marker position
        const thresholdPercentage = (threshold / maxValue) * 100;
        markerElement.style.left = `${thresholdPercentage}%`;
        
        // Update text values
        valueElement.textContent = `MSE: ${mse.toFixed(6)}`;
        thresholdElement.textContent = `Threshold: ${threshold.toFixed(6)}`;
        
        // Change bar color based on MSE value compared to threshold
        if (mse <= threshold) {
            barElement.style.background = 'var(--success-color)';
        } else {
            barElement.style.background = 'var(--danger-color)';
        }
    }
});
