# Auto-Fill Location & Atmospheric Conditions Feature

## Overview
The chat assistant now automatically detects when users mention locations and atmospheric conditions, and suggests form updates that are applied in real-time.

## How It Works

### 1. **User mentions a location** (e.g., "Tell me about Delhi")
   - Assistant recognizes the location and extracts the region
   - Sends `<suggest_location>Delhi</suggest_location>` tag in response
   - Frontend automatically sets the Region dropdown to the mapped region (e.g., "North")

### 2. **User mentions a month/time** (e.g., "How's June weather?")
   - Assistant recognizes the month and sends `<suggest_month>6</suggest_month>` tag
   - Frontend automatically sets the Month dropdown to the value

### 3. **User describes atmospheric conditions** (e.g., "35 degrees, 70% humidity")
   - Assistant recognizes conditions and sends:
     ```
     <suggest_conditions>{
       "temperature_c": 35,
       "humidity_pct": 70,
       "pressure_hpa": 1010,
       ...
     }</suggest_conditions>
     ```
   - Frontend automatically updates the sliders and editable input boxes

## Supported Locations

### North Region
- Delhi, New Delhi, Punjab, Chandigarh, Himachal (HP), Haryana, J&K, Kashmir, Jammu, Ladakh, Uttarakhand, Uttar Pradesh (UP), Lucknow

### South Region
- Tamil Nadu (TN), Chennai, Bangalore, Karnataka, Telangana, Hyderabad, Andhra Pradesh (AP), Kerala, Kochi, Trivandrum

### East Region
- West Bengal (WB), Kolkata, Calcutta, Bihar, Jharkhand, Assam, Odisha, Orissa, Bhubaneswar

### West Region
- Maharashtra (MH), Mumbai, Bombay, Pune, Goa, Gujarat, Rajasthan, Jaipur

### Central Region
- Madhya Pradesh (MP), Indore, Bhopal, Chhattisgarh

## Technical Implementation

### Backend Changes (`app.py`)

1. **Enhanced System Prompt**: Instructions tell the assistant to include suggestion tags when relevant
   
2. **Helper Functions**:
   - `extract_suggestions_from_response()`: Parses suggestion tags from LLM response
   - `clean_response_text()`: Removes tags before displaying to user
   
3. **Location Mapping**: 
   - `LOCATION_REGION_MAP`: Dictionary mapping Indian locations to regions
   - `MONTH_MAP`: Dictionary for month name to number conversion
   
4. **Updated `/api/chat` endpoint**:
   - Returns `suggestions` object along with response
   - Cleans response before storing (no visible tags to user)
   
5. **New `/api/update-form` endpoint**:
   - Validates and applies form updates server-side
   - Ensures all values stay within valid ranges

### Frontend Changes (`templates/index.html`)

1. **`applyFormUpdates()` function**: 
   - Receives suggestions object
   - Updates dropdowns (region, month)
   - Updates sliders and editable inputs (temperature, humidity, pressure, wind, precipitation, cloud cover)
   - Two-way sync maintained with existing slider-to-input sync logic

2. **Enhanced `sendMessage()` function**:
   - Receives `suggestions` in API response
   - Calls `applyFormUpdates()` automatically
   - User sees form update immediately after chatting

## Example Interactions

### Example 1: Location + Time
```
User: "What's the weather like in Chennai during December?"

Assistant Response:
"Chennai in December experiences pleasant weather with temperatures ranging from 24-29°C, 
low humidity around 65%, and minimal rainfall. It's an ideal time to visit!"

Form Auto-Updated:
- Region: South (Chennai mapped to South)
- Month: 12 (December)
- Conditions: Based on typical December patterns
```

### Example 2: Specific Conditions
```
User: "What weather would result in 28°C, 60% humidity, and high pressure?"

Assistant Response:
"With 28°C, 60% humidity, and high pressure, you'd likely see clear skies and sunny conditions..."

Form Auto-Updated:
- Temperature: 28°C
- Humidity: 60%
- Pressure: 1013 hPa (if mentioned)
```

## Data Flow

```
User Input (Chat)
    ↓
Backend: Assistant generates response with suggestion tags
    ↓
Backend: extract_suggestions_from_response() parses tags
    ↓
Backend: clean_response_text() removes tags
    ↓
API Response: {
    "response": "Clean text shown to user",
    "suggestions": {
        "region": "North",
        "month": "6",
        "conditions": {...}
    }
}
    ↓
Frontend: applyFormUpdates() applies all suggestions
    ↓
User sees: Chat response + Form automatically updated
```

## Validation

All form updates are validated:
- **Region**: Must be one of: North, South, East, West, Central
- **Month**: Must be 1-12
- **Temperature**: -50°C to 60°C
- **Humidity**: 0-100%
- **Pressure**: 870-1085 hPa
- **Wind Speed**: 0-200 km/h
- **Precipitation**: 0-500 mm
- **Cloud Cover**: 0-100%

Invalid values are silently rejected.

## Testing

Run the test script:
```bash
python test_chat_suggestions.py
```

All tests verify:
✓ Location suggestions work
✓ Month suggestions work
✓ Atmospheric condition suggestions work
✓ Combined suggestions work

## Future Enhancements

1. **Visual Feedback**: Toast notification when form is auto-updated
2. **Undo Action**: Button to revert auto-applied suggestions
3. **More Locations**: Expand location database with cities/areas
4. **Smart Conditions**: Use historical weather data for intelligent defaults
5. **Confirmation**: Optional modal asking user to confirm suggestions before applying
