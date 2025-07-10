document.addEventListener('DOMContentLoaded', function() {
    const queryForm = document.getElementById('queryForm');
    const queryInput = document.getElementById('queryInput');
    const submitBtn = document.getElementById('submitBtn');
    const loadingSection = document.getElementById('loadingSection');
    const resultsSection = document.getElementById('resultsSection');
    const errorSection = document.getElementById('errorSection');
    const answerContent = document.getElementById('answerContent');
    const citationsContent = document.getElementById('citationsContent');
    const errorMessage = document.getElementById('errorMessage');
    const status = document.getElementById('status');

    // Check service health on page load
    checkHealth();

    // Handle example query clicks
    document.addEventListener('click', function(e) {
        if (e.target.closest('.example-query')) {
            e.preventDefault();
            const button = e.target.closest('.example-query');
            const query = button.getAttribute('data-query');
            queryInput.value = query;
            queryInput.focus();
            
            // Scroll to the query form
            queryForm.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    });

    queryForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const query = queryInput.value.trim();
        if (!query) {
            showError('Please enter a query');
            return;
        }

        // Show loading state
        showLoading();
        
        try {
            const response = await fetch('/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: query })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || 'Failed to process query');
            }

            // Show results
            showResults(data);

        } catch (error) {
            console.error('Error:', error);
            showError(error.message);
        }
    });

    async function checkHealth() {
        try {
            const response = await fetch('/health');
            const data = await response.json();
            
            if (data.status === 'healthy') {
                status.textContent = `Ready (${data.documents_count} documents)`;
                status.className = 'navbar-text';
                status.innerHTML = `<i class="fas fa-circle text-success me-1"></i>Ready (${data.documents_count} documents)`;
            } else {
                status.innerHTML = `<i class="fas fa-circle text-warning me-1"></i>Service Issues`;
            }
        } catch (error) {
            status.innerHTML = `<i class="fas fa-circle text-danger me-1"></i>Offline`;
        }
    }

    function showLoading() {
        hideAllSections();
        loadingSection.style.display = 'block';
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
    }

    function showResults(data) {
        hideAllSections();
        
        // Display answer
        answerContent.textContent = data.answer;
        
        // Display citations
        citationsContent.innerHTML = '';
        if (data.citations && data.citations.length > 0) {
            data.citations.forEach((citation, index) => {
                const citationDiv = document.createElement('div');
                citationDiv.className = 'citation-item';
                citationDiv.innerHTML = `
                    <div class="citation-text">"${citation.text}"</div>
                    <div class="citation-source">Source: ${citation.source}</div>
                `;
                citationsContent.appendChild(citationDiv);
            });
        } else {
            citationsContent.innerHTML = '<p class="text-muted">No citations available</p>';
        }
        
        resultsSection.style.display = 'block';
        resetSubmitButton();
    }

    function showError(message) {
        hideAllSections();
        errorMessage.textContent = message;
        errorSection.style.display = 'block';
        resetSubmitButton();
    }

    function hideAllSections() {
        loadingSection.style.display = 'none';
        resultsSection.style.display = 'none';
        errorSection.style.display = 'none';
    }

    function resetSubmitButton() {
        submitBtn.disabled = false;
        submitBtn.innerHTML = '<i class="fas fa-search me-2"></i>Submit Query';
    }
});
