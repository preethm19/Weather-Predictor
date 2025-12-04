"""Flask application exposing the ANN weather condition predictor."""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()
from tensorflow import keras

from chat import get_provider

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "weather_ann.keras"
TRANSFORMER_PATH = BASE_DIR / "models" / "feature_transformer.joblib"
LABEL_ENCODER_PATH = BASE_DIR / "models" / "label_encoder.joblib"

REGIONS = ["North", "South", "East", "West", "Central"]
MONTHS = list(range(1, 13))

# Location mapping for Indian cities/areas to regions
LOCATION_REGION_MAP = {
    # North
    "delhi": "North", "new delhi": "North", "punjab": "North", "chandigarh": "North",
    "himachal": "North", "hp": "North", "haryana": "North", "jk": "North", "kashmir": "North",
    "jammu": "North", "ladakh": "North", "uttarakhand": "North", "uttar pradesh": "North",
    "up": "North", "lucknow": "North", "delhi": "North",
    # South
    "tamil nadu": "South", "tn": "South", "chennai": "South", "bangalore": "South",
    "karnataka": "South", "telangana": "South", "hyderabad": "South", "andhra pradesh": "South",
    "ap": "South", "kerala": "South", "kochi": "South", "trivandrum": "South",
    # East
    "west bengal": "East", "wb": "East", "kolkata": "East", "calcutta": "East",
    "bihar": "East", "jharkhand": "East", "assam": "East", "odisha": "East",
    "orissa": "East", "bhubaneswar": "East",
    # West
    "maharashtra": "West", "mh": "West", "mumbai": "West", "bombay": "West",
    "pune": "West", "goa": "West", "gujarat": "West", "rajasthan": "West",
    "jaipur": "West",
    # Central
    "madhya pradesh": "Central", "mp": "Central", "indore": "Central", "bhopal": "Central",
    "chhattisgarh": "Central"
}

MONTH_MAP = {
    'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
    'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6, 'july': 7, 'jul': 7,
    'august': 8, 'aug': 8, 'september': 9, 'sep': 9, 'october': 10, 'oct': 10,
    'november': 11, 'nov': 11, 'december': 12, 'dec': 12
}

app = Flask(__name__)

# Chat session storage and configuration
CHAT_SESSIONS = {}  # {session_id: {'messages': [...], 'created_at': timestamp}}
MAX_HISTORY_MESSAGES = 12
SYSTEM_PROMPT = """You are a helpful weather assistant for an India-focused Weather Predictor app.

**CRITICAL INSTRUCTIONS FOR FORM AUTO-FILL:**

1. **ALWAYS detect location mentions**: If the user mentions ANY Indian city, area, or region (e.g., Delhi, Mumbai, Chennai, Tamil Nadu, Punjab, etc.), 
   you MUST include: <suggest_location>ExactLocationName</suggest_location>

2. **ALWAYS detect month/time mentions**: If the user mentions any month (e.g., January, Feb, June, December) or time period (summer, monsoon, winter),
   you MUST include: <suggest_month>MonthNumber</suggest_month>

3. **PROVIDE typical atmospheric conditions**: Based on the location and month mentioned, always suggest realistic atmospheric conditions using:
   <suggest_conditions>{"temperature_c": X, "humidity_pct": Y, "pressure_hpa": Z, "wind_speed_kph": W, "precip_mm": P, "cloud_cover_pct": C}</suggest_conditions>
   
   For example:
   - Delhi in January: cool, dry (temp ~15°C, humidity ~45%, pressure ~1000hPa, wind ~10km/h, rain ~0mm, clouds ~20%)
   - Mumbai in August: hot, monsoon (temp ~30°C, humidity ~85%, pressure ~1005hPa, wind ~15km/h, rain ~50mm, clouds ~80%)
   - Bangalore in March: warm, dry (temp ~32°C, humidity ~55%, pressure ~1010hPa, wind ~12km/h, rain ~5mm, clouds ~40%)

4. **EVEN IF user only mentions location WITHOUT month, provide month**: Use current typical month patterns or ask clarification, but ALWAYS include suggestions.

5. **Format strictly**: Use ONLY these XML tags. Keep your main response friendly and informative, but append the tags at the end.

6. **Example response format**:
   "Delhi has pleasant weather in January with cool temperatures and clear skies. The air is crisp and visibility is excellent."
   <suggest_location>Delhi</suggest_location>
   <suggest_month>1</suggest_month>
   <suggest_conditions>{"temperature_c": 15, "humidity_pct": 45, "pressure_hpa": 1000, "wind_speed_kph": 10, "precip_mm": 0, "cloud_cover_pct": 20}</suggest_conditions>"""
CHAT_PROVIDER = os.getenv("CHAT_PROVIDER", "gemini")


def get_session_messages(session_id: str, max_items: int = MAX_HISTORY_MESSAGES) -> List[Dict[str, str]]:
    """Retrieve message history for a session, capped at max_items."""
    if session_id not in CHAT_SESSIONS:
        CHAT_SESSIONS[session_id] = {"messages": []}
    
    messages = CHAT_SESSIONS[session_id]["messages"]
    
    # Ensure system prompt is at the front if not present
    if not messages or messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    
    # Return last max_items messages (keep system prompt at front)
    if len(messages) > max_items:
        return [messages[0]] + messages[-(max_items - 1):]
    
    return messages


def validate_chat_payload(text: str, max_length: int = 2000) -> bool:
    """Simple validation: check length and basic content."""
    if not text or len(text.strip()) == 0:
        return False
    if len(text) > max_length:
        return False
    return True


def extract_suggestions_from_response(response_text: str) -> Dict:
    """Extract location, month, and condition suggestions from LLM response."""
    import re
    suggestions = {}
    
    # Extract location suggestion - handle multi-word locations
    location_match = re.search(r'<suggest_location>\s*(.+?)\s*</suggest_location>', response_text, re.IGNORECASE)
    if location_match:
        location = location_match.group(1).strip().lower()
        # Map location to region
        region = LOCATION_REGION_MAP.get(location)
        if region:
            suggestions['region'] = region
        else:
            # Try partial match for multi-word locations
            for loc_key, region_val in LOCATION_REGION_MAP.items():
                if location in loc_key or loc_key in location:
                    suggestions['region'] = region_val
                    break
    
    # Extract month suggestion
    month_match = re.search(r'<suggest_month>\s*(\d+)\s*</suggest_month>', response_text, re.IGNORECASE)
    if month_match:
        month = int(month_match.group(1))
        if 1 <= month <= 12:
            suggestions['month'] = str(month)
    
    # Extract conditions suggestion (JSON format) - more flexible regex
    conditions_match = re.search(r'<suggest_conditions>\s*(\{[\s\S]*?\})\s*</suggest_conditions>', response_text, re.IGNORECASE)
    if conditions_match:
        try:
            conditions_json = conditions_match.group(1)
            conditions = json.loads(conditions_json)
            # Validate and add conditions
            if isinstance(conditions, dict):
                suggestions['conditions'] = conditions
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing conditions JSON: {e}")
    
    return suggestions


def clean_response_text(response_text: str) -> str:
    """Remove suggestion tags from response before sending to user."""
    import re
    cleaned = re.sub(r'<suggest_\w+>.*?</suggest_\w+>', '', response_text, flags=re.IGNORECASE | re.DOTALL)
    return cleaned.strip()

    
def parse_user_message_for_suggestions(user_message: str) -> Dict:
    """Lightweight server-side parser to extract location, month and numeric conditions from the user's message.

    This runs as a fallback so the form can be auto-filled even if the LLM doesn't emit suggestion tags.
    """
    import re
    suggestions: Dict = {}
    text = (user_message or "").lower()

    # Location detection (direct and partial matches)
    for loc_key, region_val in LOCATION_REGION_MAP.items():
        if loc_key in text:
            suggestions.setdefault('region', region_val)
            # Prefer the first match
            break

    # Month detection by name
    for name, num in MONTH_MAP.items():
        if name in text:
            suggestions.setdefault('month', str(num))
            break

    # Numeric conditions extraction
    conditions = {}

    # Temperature (e.g., 25C, 25 °C, 25 degrees)
    temp_match = re.search(r"(\d{1,2}(?:\.\d+)?)\s*(?:°\s*C|°C|celsius|degrees|degree|°)", text)
    if not temp_match:
        temp_match = re.search(r"(\d{1,2}(?:\.\d+)?)\s*degrees", text)
    if temp_match:
        try:
            t = float(temp_match.group(1))
            if -50 <= t <= 60:
                conditions['temperature_c'] = t
        except ValueError:
            pass

    # Humidity (e.g., 70%, 70 percent)
    hum_match = re.search(r"(\d{1,3}(?:\.\d+)?)\s*%\s*(?:humidity)?", text) or re.search(r"(\d{1,3}(?:\.\d+)?)\s*percent\s*(?:humidity)?", text)
    if hum_match:
        try:
            h = float(hum_match.group(1))
            if 0 <= h <= 100:
                conditions['humidity_pct'] = h
        except ValueError:
            pass

    # Pressure (e.g., 1013 hPa)
    pres_match = re.search(r"(\d{3,4}(?:\.\d+)?)\s*(?:hpa|hectopascals|hpa)", text)
    if pres_match:
        try:
            p = float(pres_match.group(1))
            if 870 <= p <= 1085:
                conditions['pressure_hpa'] = p
        except ValueError:
            pass

    # Wind speed (e.g., 15 km/h, 15 kph)
    wind_match = re.search(r"(\d{1,3}(?:\.\d+)?)\s*(?:km/?h|kph|kmh)", text)
    if wind_match:
        try:
            w = float(wind_match.group(1))
            if 0 <= w <= 200:
                conditions['wind_speed_kph'] = w
        except ValueError:
            pass

    # Precipitation (e.g., 10 mm)
    precip_match = re.search(r"(\d{1,4}(?:\.\d+)?)\s*mm", text)
    if precip_match:
        try:
            pr = float(precip_match.group(1))
            if 0 <= pr <= 500:
                conditions['precip_mm'] = pr
        except ValueError:
            pass

    # Cloud cover (e.g., 50% cloud cover)
    cloud_match = re.search(r"(\d{1,3}(?:\.\d+)?)\s*%\s*(?:cloud|cloud cover|clouds)", text)
    if cloud_match:
        try:
            c = float(cloud_match.group(1))
            if 0 <= c <= 100:
                conditions['cloud_cover_pct'] = c
        except ValueError:
            pass

    if conditions:
        suggestions['conditions'] = conditions

    return suggestions


def load_artifacts() -> Tuple[keras.Model, object, object]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run `python train.py` to generate the model before starting the API."
        )
    model = keras.models.load_model(MODEL_PATH)
    transformer = joblib.load(TRANSFORMER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    return model, transformer, label_encoder


MODEL, TRANSFORMER, LABEL_ENCODER = load_artifacts()


def prepare_dataframe(payload: Dict[str, object]) -> pd.DataFrame:
    """Validate and convert inbound payload to a single-row DataFrame."""
    required_fields = {
        "region": str,
        "month": (int, float),
        "temperature_c": (int, float),
        "humidity_pct": (int, float),
        "pressure_hpa": (int, float),
        "wind_speed_kph": (int, float),
        "precip_mm": (int, float),
        "cloud_cover_pct": (int, float),
    }

    data = {}
    for field, expected_type in required_fields.items():
        if field not in payload:
            raise ValueError(f"Missing field: {field}")
        value = payload[field]
        if isinstance(expected_type, tuple):
            valid = isinstance(value, expected_type)
        else:
            valid = isinstance(value, expected_type)
        if not valid:
            try:
                # Attempt coercion to float/int as appropriate
                if expected_type is str:
                    value = str(value)
                else:
                    value = float(value)
            except Exception as exc:  # pylint: disable=broad-except
                raise ValueError(f"Invalid value for {field}: {payload[field]}") from exc
        data[field] = value

    region = str(data["region"]).title()
    if region not in REGIONS:
        raise ValueError(f"Unsupported region '{data['region']}'. Expected one of {', '.join(REGIONS)}")
    data["region"] = region

    month = int(round(float(data["month"])))
    if month < 1 or month > 12:
        raise ValueError("Month must be between 1 and 12")
    data["month"] = month

    df = pd.DataFrame([data])
    return df


def predict_condition(payload: Dict[str, object]) -> Tuple[str, float, Dict[str, float]]:
    df = prepare_dataframe(payload)
    features = TRANSFORMER.transform(df)
    if hasattr(features, "toarray"):
        features = features.toarray()
    features = features.astype(np.float32)

    probabilities = MODEL.predict(features, verbose=0)[0]
    predicted_idx = int(np.argmax(probabilities))
    condition = LABEL_ENCODER.inverse_transform([predicted_idx])[0]

    probability_map = {
        LABEL_ENCODER.inverse_transform([idx])[0]: float(round(prob, 4))
        for idx, prob in enumerate(probabilities)
    }
    confidence = float(round(probabilities[predicted_idx], 4))
    return condition, confidence, probability_map

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    current_month = datetime.now().month  # Get current month (1-12)
    form_data = {
        "region": REGIONS[0],
        "month": current_month,
        "temperature_c": 30,
        "humidity_pct": 70,
        "pressure_hpa": 1005,
        "wind_speed_kph": 12,
        "precip_mm": 10,
        "cloud_cover_pct": 65,
    }

    if request.method == "POST":
        raw_form = request.form.to_dict()
        form_data.update(raw_form)
        try:
            payload = {
                "region": raw_form.get("region", ""),
                "month": raw_form.get("month", 0),
                "temperature_c": raw_form.get("temperature_c", 0),
                "humidity_pct": raw_form.get("humidity_pct", 0),
                "pressure_hpa": raw_form.get("pressure_hpa", 0),
                "wind_speed_kph": raw_form.get("wind_speed_kph", 0),
                "precip_mm": raw_form.get("precip_mm", 0),
                "cloud_cover_pct": raw_form.get("cloud_cover_pct", 0),
            }
            condition, confidence, probability_map = predict_condition(payload)
            prediction = {
                "condition": condition,
                "confidence": confidence,
                "probabilities": probability_map,
                "labels": json.dumps(list(probability_map.keys())),
                "data": json.dumps(list(probability_map.values())),
            }
        except ValueError as exc:
            error = str(exc)

    return render_template(
        "index.html",
        regions=REGIONS,
        months=MONTHS,
        form_data=form_data,
        prediction=prediction,
        error=error,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        payload = request.get_json(force=True)
        condition, confidence, probability_map = predict_condition(payload)
        return jsonify(
            {
                "condition": condition,
                "confidence": confidence,
                "probabilities": probability_map,
            }
        )
    except (ValueError, KeyError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/update-form", methods=["POST"])
def update_form():
    """Update weather form fields with suggested values."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON payload"}), 400
        
        updates = {}
        
        # Validate and update region
        if "region" in data and data["region"] in REGIONS:
            updates["region"] = data["region"]
        
        # Validate and update month
        if "month" in data:
            month = int(data["month"])
            if 1 <= month <= 12:
                updates["month"] = month
        
        # Validate and update conditions
        if "conditions" in data and isinstance(data["conditions"], dict):
            conditions = data["conditions"]
            # Validate ranges
            if "temperature_c" in conditions:
                temp = float(conditions["temperature_c"])
                if -50 <= temp <= 60:
                    updates["temperature_c"] = temp
            if "humidity_pct" in conditions:
                hum = float(conditions["humidity_pct"])
                if 0 <= hum <= 100:
                    updates["humidity_pct"] = hum
            if "pressure_hpa" in conditions:
                pres = float(conditions["pressure_hpa"])
                if 870 <= pres <= 1085:
                    updates["pressure_hpa"] = pres
            if "wind_speed_kph" in conditions:
                wind = float(conditions["wind_speed_kph"])
                if 0 <= wind <= 200:
                    updates["wind_speed_kph"] = wind
            if "precip_mm" in conditions:
                precip = float(conditions["precip_mm"])
                if 0 <= precip <= 500:
                    updates["precip_mm"] = precip
            if "cloud_cover_pct" in conditions:
                cloud = float(conditions["cloud_cover_pct"])
                if 0 <= cloud <= 100:
                    updates["cloud_cover_pct"] = cloud
        
        return jsonify({"success": True, "updates": updates})
    
    except (ValueError, KeyError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400

@app.route("/api/chat", methods=["POST"])
def chat_api():
    """Non-streaming chat endpoint with weather context."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON payload"}), 400
        
        user_message = data.get("message", "").strip()
        session_id = data.get("session_id") or str(uuid.uuid4())
        # Optional: receive current form values from client for context
        weather_context = data.get("weather_context", {})
        
        if not validate_chat_payload(user_message):
            return jsonify({"error": "Invalid or empty message"}), 400
        
        # Get or initialize session
        if session_id not in CHAT_SESSIONS:
            CHAT_SESSIONS[session_id] = {"messages": []}
        
        # Append user message to history
        CHAT_SESSIONS[session_id]["messages"].append({
            "role": "user",
            "content": user_message
        })
        
        # Get message history for context
        messages = get_session_messages(session_id)

        # Server-side quick suggestions (fallback) parsed from the user's message
        local_suggestions = parse_user_message_for_suggestions(user_message)
        
        # Build enhanced system message with weather data context
        enhanced_system_prompt = SYSTEM_PROMPT
        if weather_context:
            enhanced_system_prompt += "\n\nCurrent weather form values from the user:\n"
            if weather_context.get("region"):
                enhanced_system_prompt += f"- Region: {weather_context['region']}\n"
            if weather_context.get("month"):
                enhanced_system_prompt += f"- Month: {weather_context['month']}\n"
            if weather_context.get("temperature_c") is not None:
                enhanced_system_prompt += f"- Temperature: {weather_context['temperature_c']}°C\n"
            if weather_context.get("humidity_pct") is not None:
                enhanced_system_prompt += f"- Humidity: {weather_context['humidity_pct']}%\n"
            if weather_context.get("pressure_hpa") is not None:
                enhanced_system_prompt += f"- Pressure: {weather_context['pressure_hpa']} hPa\n"
            if weather_context.get("wind_speed_kph") is not None:
                enhanced_system_prompt += f"- Wind Speed: {weather_context['wind_speed_kph']} km/h\n"
            if weather_context.get("precip_mm") is not None:
                enhanced_system_prompt += f"- Precipitation: {weather_context['precip_mm']} mm\n"
            if weather_context.get("cloud_cover_pct") is not None:
                enhanced_system_prompt += f"- Cloud Cover: {weather_context['cloud_cover_pct']}%\n"
            # Include prediction if available
            if weather_context.get("prediction"):
                enhanced_system_prompt += f"\nLast weather prediction: {weather_context['prediction']['condition']} (confidence: {weather_context['prediction']['confidence']*100:.1f}%)\n"
        
        # Update first message (system message) if it exists, otherwise prepend
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = enhanced_system_prompt
        else:
            messages.insert(0, {"role": "system", "content": enhanced_system_prompt})
        
        # Call chat provider
        try:
            provider = get_provider(CHAT_PROVIDER)
            response_text = provider.chat(
                messages=messages,
                stream=False,
                max_tokens=500,
                temperature=0.7
            )
        except Exception as e:
            return jsonify({"error": f"LLM error: {str(e)}"}), 500
        
        # Extract suggestions from response
        llm_suggestions = extract_suggestions_from_response(response_text)

        # Merge local suggestions with LLM suggestions.
        # LLM suggestions take precedence, but we fall back to local values.
        suggestions = {}
        # region
        if 'region' in llm_suggestions:
            suggestions['region'] = llm_suggestions['region']
        elif 'region' in local_suggestions:
            suggestions['region'] = local_suggestions['region']

        # month
        if 'month' in llm_suggestions:
            suggestions['month'] = llm_suggestions['month']
        elif 'month' in local_suggestions:
            suggestions['month'] = local_suggestions['month']

        # conditions (merge dicts, LLM keys override local)
        merged_conditions = {}
        if 'conditions' in local_suggestions:
            merged_conditions.update(local_suggestions['conditions'])
        if 'conditions' in llm_suggestions:
            merged_conditions.update(llm_suggestions['conditions'])
        if merged_conditions:
            suggestions['conditions'] = merged_conditions

        # Clean response (remove suggestion tags before storing and sending)
        cleaned_response = clean_response_text(response_text)
        
        # Store assistant response (cleaned version)
        CHAT_SESSIONS[session_id]["messages"].append({
            "role": "assistant",
            "content": cleaned_response
        })
        
        return jsonify({
            "session_id": session_id,
            "response": cleaned_response,
            "model": CHAT_PROVIDER,
            "suggestions": suggestions
        })
    
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)