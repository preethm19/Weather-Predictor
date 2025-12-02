# India Weather Assistant

An intelligent weather prediction application powered by Artificial Neural Networks (ANN) and Large Language Models (LLM). Features an interactive web interface with real-time probability visualization, AI-powered chat assistant, and smart auto-fill capabilities that detect locations and atmospheric conditions from natural language conversations.

## 🌟 Key Features

- **ANN Weather Prediction**: Machine learning model trained on climatologically-inspired synthetic data for Indian weather patterns
- **Interactive Web UI**: Modern interface with form-based predictions and dynamic probability bar charts
- **AI Chat Assistant**: Gemini-powered conversational interface for weather discussions
- **Smart Auto-Fill**: Chat automatically detects Indian locations, months, and weather conditions to update prediction forms
- **Session Management**: Persistent chat conversations with context awareness
- **RESTful APIs**: Comprehensive API endpoints for predictions, chat, and form updates
- **Location Intelligence**: Maps 100+ Indian cities and regions to weather prediction zones
- **Real-time Form Updates**: Instant synchronization between chat suggestions and prediction inputs

## 🏗️ Project Structure

```
.
├── app.py                          # Main Flask application with ANN prediction and AI chat
├── train.py                        # Data generation and ANN model training script
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment configuration template
├── AUTO_FILL_FEATURE_GUIDE.md       # Detailed auto-fill feature documentation
├── README.md                        # This file
├── chat/                           # Chat provider abstraction layer
│   ├── __init__.py
│   └── provider.py                 # Supports Gemini, Llama (extensible)
├── data/                           # Generated synthetic weather datasets
│   └── synthetic_weather_india.csv
├── models/                         # Trained ML artifacts
│   ├── weather_ann.keras           # ANN model (TensorFlow)
│   ├── feature_transformer.joblib  # Preprocessing pipeline
│   └── label_encoder.joblib        # Weather condition encoder
├── static/css/                     # Frontend styling
│   └── main.css
├── templates/                      # Jinja2 HTML templates
│   └── index.html                  # Main application interface
└── tests/                          # Test suite (in development)
    ├── test_chat_suggestions.py    # Chat suggestion functionality tests
    └── test.py                     # General application tests
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Google Gemini API key ([Get here](https://aistudio.google.com/app/apikey))

### Installation

1. **Clone and navigate to the project**
   ```bash
   git clone <repository-url>
   cd weather-predictor
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Unix/Mac:
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your GOOGLE_API_KEY
   ```

5. **Train the ANN model** (generates training data and model artifacts)
   ```bash
   python train.py
   ```

6. **Start the application**
   ```bash
   python app.py
   ```

7. **Open your browser** to [http://127.0.0.1:5000](http://127.0.0.1:5000)

## 🎯 Usage

### Web Interface
- **Weather Prediction Form**: Input atmospheric conditions to get ANN predictions with probability visualizations
- **AI Chat Assistant**: Chat about weather conditions and automatically populate the prediction form
- **Real-time Updates**: Form fields update instantly based on chat conversations

### Auto-Fill Magic ✨
The chat assistant intelligently detects:
- **Locations**: "Delhi weather?" → Automatically sets Region to "North"
- **Time Periods**: "June monsoon?" → Sets Month to 6
- **Conditions**: "30°C and 80% humidity" → Updates temperature and humidity fields

### API Endpoints

#### Weather Prediction
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "region": "South",
    "month": 7,
    "temperature_c": 29.5,
    "humidity_pct": 82,
    "pressure_hpa": 1002,
    "wind_speed_kph": 14,
    "precip_mm": 18,
    "cloud_cover_pct": 78
  }'
```

**Response:**
```json
{
  "condition": "Rain",
  "confidence": 0.734,
  "probabilities": {
    "Sunny": 0.120,
    "Cloudy": 0.182,
    "Rain": 0.734,
    "Storm": 0.010,
    "Fog": 0.002
  }
}
```

#### Chat API
```bash
curl -X POST http://127.0.0.1:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "How is the weather in Mumbai during December?",
    "session_id": "optional-session-id"
  }'
```

**Response:**
```json
{
  "session_id": "generated-or-provided",
  "response": "Mumbai in December typically has pleasant weather with temperatures around 28-32°C...",
  "model": "gemini",
  "suggestions": {
    "region": "West",
    "month": "12",
    "conditions": {
      "temperature_c": 30,
      "humidity_pct": 65,
      "precip_mm": 5
    }
  }
}
```

#### Form Update API
```bash
curl -X POST http://127.0.0.1:5000/api/update-form \
  -H "Content-Type: application/json" \
  -d '{
    "region": "North",
    "month": "6",
    "conditions": {
      "temperature_c": 35,
      "humidity_pct": 40
    }
  }'
```

## ⚙️ Configuration

### Environment Variables
Create a `.env` file (copy from `.env.example`):

```env
# Required
GOOGLE_API_KEY=your-gemini-api-key-here

# Optional
CHAT_PROVIDER=gemini  # Options: gemini, llama
GEMINI_MODEL=gemini-2.0-flash  # Specific Gemini model
FLASK_ENV=development
FLASK_DEBUG=True
```

### Supported Locations
The system recognizes 100+ Indian locations mapped to 5 weather regions:

- **North**: Delhi, Punjab, Himachal, Kashmir, Uttar Pradesh, etc.
- **South**: Tamil Nadu, Kerala, Karnataka, Andhra Pradesh, etc.
- **East**: West Bengal, Bihar, Odisha, Assam, etc.
- **West**: Maharashtra, Gujarat, Rajasthan, Goa, etc.
- **Central**: Madhya Pradesh, Chhattisgarh

## 🧪 Testing

Run the test suite:
```bash
python test_chat_suggestions.py
```

Tests cover:
- ✅ Location suggestion parsing
- ✅ Month extraction from text
- ✅ Atmospheric condition detection
- ✅ Form validation ranges
- ✅ API response formats

## 🏛️ Architecture

### Backend Architecture
- **Flask Web Framework**: RESTful API endpoints
- **TensorFlow/Keras**: ANN implementation for weather prediction
- **Scikit-learn**: Data preprocessing pipeline
- **Gemini API**: Conversational AI with structured output parsing
- **Session Management**: In-memory chat history (production-ready for Redis)

### Frontend Architecture
- **Vanilla JavaScript**: DOM manipulation and API interactions
- **Chart.js**: Real-time probability visualization
- **CSS3**: Responsive design with modern styling
- **Jinja2 Templates**: Server-side rendering with dynamic data

### AI Pipeline
1. **User Input** → Natural language processing
2. **LLM Response** → Structured suggestion extraction
3. **Context Mapping** → Location→region, text→conditions
4. **Form Updates** → Real-time interface synchronization
5. **Prediction** → ANN inference with visualization

## 🔧 Development

### Adding New Chat Providers
Extend `chat/provider.py` with new provider classes:
```python
class NewProvider(ChatProvider):
    def chat(self, messages, stream=False, max_tokens=None, temperature=0.7):
        # Implement chat logic
        pass
```

### Customizing Suggestions
Modify system prompts in `app.py` to change AI behavior for location/condition detection.

### Extending Location Database
Add to `LOCATION_REGION_MAP` in `app.py` to support more Indian locations.

## 🚀 Future Enhancements

- **Real Weather APIs**: Integrate OpenWeatherMap for actual weather data validation
- **User Accounts**: Persistent sessions with prediction history
- **Advanced Visualizations**: Weather trend charts and historical comparisons
- **Mobile App**: React Native companion application
- **Multi-language Support**: Hindi and regional language interfaces
- **Weather Alerts**: Integration with disaster management systems

## 📚 Dependencies

### Core Dependencies
- `flask>=3.0.0`: Web framework
- `tensorflow>=2.13.0`: Machine learning framework
- `google-generativeai>=0.8.0`: Gemini AI integration
- `scikit-learn>=1.3.0`: Data preprocessing
- `pandas>=2.1.0`: Data manipulation
- `numpy>=1.24.0`: Numerical computing

### Development Dependencies
- `python-dotenv>=1.0.0`: Environment management
- `requests>=2.30.0`: HTTP client for future APIs

## 📄 License

This project is developed for educational purposes demonstrating modern AI and ML integration patterns.