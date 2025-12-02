"""Test script for chat suggestions feature."""
import json
import requests

BASE_URL = "http://127.0.0.1:5000/api"

def test_location_suggestion():
    """Test that mentioning a location triggers suggestions."""
    print("\n=== Testing Location Suggestion ===")
    
    payload = {
        "message": "Tell me about the weather in Delhi during January",
        "session_id": "test_session_1",
        "weather_context": {
            "region": "North",
            "month": "1",
            "temperature_c": 15,
            "humidity_pct": 45,
            "pressure_hpa": 1000,
            "wind_speed_kph": 10,
            "precip_mm": 0,
            "cloud_cover_pct": 20
        }
    }
    
    response = requests.post(f"{BASE_URL}/chat", json=payload)
    data = response.json()
    
    print(f"Response: {data['response'][:150]}...")
    print(f"Suggestions: {json.dumps(data.get('suggestions', {}), indent=2)}")
    
    if data.get('suggestions', {}).get('region') == 'North':
        print("✓ Location suggestion working!")
    else:
        print("✗ Location suggestion not detected")

def test_month_suggestion():
    """Test that mentioning a month triggers suggestions."""
    print("\n=== Testing Month Suggestion ===")
    
    payload = {
        "message": "What's the weather like in June?",
        "session_id": "test_session_2",
        "weather_context": {}
    }
    
    response = requests.post(f"{BASE_URL}/chat", json=payload)
    data = response.json()
    
    print(f"Response: {data['response'][:150]}...")
    print(f"Suggestions: {json.dumps(data.get('suggestions', {}), indent=2)}")
    
    if data.get('suggestions', {}).get('month'):
        print("✓ Month suggestion working!")
    else:
        print("✗ Month suggestion not detected")

def test_condition_suggestion():
    """Test that describing conditions triggers suggestions."""
    print("\n=== Testing Atmospheric Condition Suggestion ===")
    
    payload = {
        "message": "What would be the weather if it's 35 degrees, 70% humidity, 1010 hPa pressure?",
        "session_id": "test_session_3",
        "weather_context": {}
    }
    
    response = requests.post(f"{BASE_URL}/chat", json=payload)
    data = response.json()
    
    print(f"Response: {data['response'][:150]}...")
    print(f"Suggestions: {json.dumps(data.get('suggestions', {}), indent=2)}")
    
    if data.get('suggestions', {}).get('conditions'):
        print("✓ Atmospheric condition suggestion working!")
    else:
        print("✗ Atmospheric condition suggestion not detected")

def test_combined():
    """Test combined location and month suggestion."""
    print("\n=== Testing Combined Location + Month Suggestion ===")
    
    payload = {
        "message": "How is the weather in Mumbai during August monsoon season?",
        "session_id": "test_session_4",
        "weather_context": {}
    }
    
    response = requests.post(f"{BASE_URL}/chat", json=payload)
    data = response.json()
    
    print(f"Response: {data['response'][:150]}...")
    print(f"Suggestions: {json.dumps(data.get('suggestions', {}), indent=2)}")
    
    has_location = data.get('suggestions', {}).get('region') is not None
    has_month = data.get('suggestions', {}).get('month') is not None
    
    if has_location or has_month:
        print("✓ Combined suggestions working!")
    else:
        print("✗ Combined suggestions not detected")

if __name__ == "__main__":
    print("Testing Chat Suggestions Feature")
    print("=" * 50)
    
    try:
        test_location_suggestion()
        test_month_suggestion()
        test_condition_suggestion()
        test_combined()
        
        print("\n" + "=" * 50)
        print("Tests completed!")
    except Exception as e:
        print(f"Error: {e}")
