import os, pprint
from dotenv import load_dotenv

# Load local .env for convenience when running tests/scripts
load_dotenv()
try:
    import google.generativeai as genai
except Exception as e:
    raise SystemExit("Install google.generativeai and set GOOGLE_API_KEY") from e

api_key = "AIzaSyDF_6_cknO_jF75NET78QOkcNMMOe4FBXo"
if not api_key:
    raise SystemExit("Set GOOGLE_API_KEY in your environment first")

genai.configure(api_key=api_key)
models = genai.list_models()
for m in models:
    print("-----")
    # print best-effort name
    try:
        print("name:", m.name)
    except Exception:
        print("repr:", repr(m))
    # print keys/attrs to help identify supported methods
    try:
        attrs = [a for a in dir(m) if not a.startswith('_')]
        print("attrs:", attrs)
    except Exception as e:
        print("attrs: (error)", e)
    # pretty print full object if possible
    try:
        pprint.pprint(m.__dict__)
    except Exception:
        pass
