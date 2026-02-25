# Create a virtual environment
py -3.12 -m venv .venv
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to create virtual environment"
    exit $LASTEXITCODE
}

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to activate virtual environment. You may need to run 'Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser'."
    exit $LASTEXITCODE
}

# Upgrade pip
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to install requirements"
    exit $LASTEXITCODE
}

# Download spaCy model
python -m spacy download en_core_web_sm
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to download spaCy model"
    exit $LASTEXITCODE
}

# Generate synthetic train classifier
python train_classifier.py --synthetic
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to generate synthetic classifier"
    exit $LASTEXITCODE
}

Write-Host "Setup completed successfully."
