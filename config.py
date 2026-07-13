# config.py
# Configuration and constants for the application

GROQ_API_KEY = "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"

# CSS Styles
CUSTOM_CSS = """
    <style>
    .main-header {
        font-size: 3rem;
        color: #00ff88;
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #1a1a2e, #16213e, #0f3460);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4fc3f7;
        padding: 1rem 0;
    }
    .metric-card {
        background: #1e1e3f;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #00ff88;
    }
    .stButton>button {
        background: linear-gradient(90deg, #00ff88, #00b4d8);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-weight: bold;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        transition: 0.3s;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
    }
    .user-message {
        background: #1e1e3f;
        border-left: 4px solid #00ff88;
    }
    .ai-message {
        background: #16213e;
        border-left: 4px solid #00b4d8;
    }
    .prediction-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #00ff88;
        text-align: center;
        margin: 1rem 0;
    }
    .prediction-value {
        font-size: 3rem;
        font-weight: bold;
        color: #00ff88;
    }
    .prediction-label {
        font-size: 1.5rem;
        color: #4fc3f7;
        margin-top: 0.5rem;
    }
    .prediction-confidence {
        font-size: 1rem;
        color: #aaa;
        margin-top: 0.5rem;
    }
    </style>
"""