import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load the current .env
load_dotenv()

def diagnose():
    api_key = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("MODEL_NAME", "gpt-4o-mini")

    print(f"--- Configuration ---")
    print(f"API Base: {api_base}")
    print(f"Model: {model}")
    print(f"Token Start: {api_key[:10] if api_key else 'None'}...")
    print(f"----------------------\n")

    # Detect if we should use the Ollama library
    use_ollama = "gpt-oss" in model.lower() or "ollama" in model.lower()

    if use_ollama:
        print("Using NATIVE OLLAMA library...")
        import ollama
        try:
            client = ollama.Client(host=api_base) if api_base and "localhost" not in api_base else ollama
            response = client.chat(
                model=model,
                messages=[{'role': 'user', 'content': "Say 'hello world' and nothing else."}],
                options={'temperature': 0.0}
            )
            print("SUCCESS! Output:")
            print(response['message']['content'])
        except Exception as e:
            print(f"FAILED with error type: {type(e).__name__}")
            print(f"Error Message: {str(e)}")
            print("\nDIAGNOSIS: Ollama library failed to connect. Ensure 'ollama signin' was run.")
    else:
        print("Using OPENAI client...")
        if not api_key:
            print("ERROR: No API Key found in environment.")
            return

        client = OpenAI(base_url=api_base, api_key=api_key)

        print("Attempting connection...")
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": "Say 'hello world' and nothing else."}
                ],
                max_tokens=10
            )
            print("SUCCESS! Output:")
            print(response.choices[0].message.content)
        except Exception as e:
            print(f"FAILED with error type: {type(e).__name__}")
            print(f"Error Message: {str(e)}")
            
            if "401" in str(e) or "Unauthorized" in str(e):
                 print("\nDIAGNOSIS: Unauthorized. Likely your token is not valid for this API Base URL.")
            elif "404" in str(e) or "Not Found" in str(e):
                 print("\nDIAGNOSIS: Model not found or Endpoint invalid.")
            elif "Connection" in type(e).__name__:
                 print("\nDIAGNOSIS: Network error. Cannot reach the server.")

if __name__ == "__main__":
    diagnose()
