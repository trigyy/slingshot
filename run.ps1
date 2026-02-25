# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Check if environment variables are set in .env
if (-not (Test-Path .env)) {
    Write-Warning "No .env file found. Make sure GROQ_API_KEY, DEEPGRAM_API_KEY, and ELEVENLABS_API_KEY are configured."
}

# Run the application
python main.py
