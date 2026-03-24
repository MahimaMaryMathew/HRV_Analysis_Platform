import os
import logging
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logging.warning("python-dotenv not found. Relying on environment variables.")

# Configuration
API_URL = "https://api-inference.huggingface.co/models"
# We use the user's requested model if available, otherwise fall back to a robust default
MODEL_ID = "convaiinnovations/medgemma-4b-ecginstruct"
# Note: As of late 2025, exact model availability on the free tier varies.
# We will try the specific model, then fall back to a general instruction tuned model.
FALLBACK_MODEL_ID = "google/gemma-2-9b-it" 

def query(payload, model_id, token):
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(f"{API_URL}/{model_id}", headers=headers, json=payload)
    return response.json()

def get_feedback_from_ai(hrv_metrics):
    """
    Generates an educational HRV report using the Hugging Face Inference API.
    Uses 'requests' to avoid installing 'huggingface_hub' in restricted environments.
    
    Args:
        hrv_metrics (dict): Dictionary containing HRV parameters (rmssd, lf_hf_ratio, etc.)
        
    Returns:
        str: The generated text report.
    """
    token = os.getenv('HF_TOKEN')
    if not token:
        logging.error("HF_TOKEN not found in environment variables.")
        return "Error: Hugging Face API Token (HF_TOKEN) is missing. Please add it to your .env file."

    # Format the prompt
    prompt = (
        f"Act as an expert cardiologist and data scientist. Analyze the following Heart Rate Variability (HRV) data:\n"
        f"- RMSSD: {hrv_metrics.get('rmssd', 'N/A')} ms\n"
        f"- LF/HF Ratio: {hrv_metrics.get('lf_hf_ratio', 'N/A')}\n"
        f"- SDNN: {hrv_metrics.get('sdnn', 'N/A')} ms\n\n"
        "Provide a concise, educational summary of the user's autonomic nervous system state based on these metrics. "
        "Include 3 specific, actionable health tips to improve their HRV. "
        "Do not provide a medical diagnosis. "
        "Structure your response with the following headers:\n"
        "1. OBSERVATION\n"
        "2. ANALYSIS\n"
        "3. RECOMMENDATIONS"
    )

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 500,
            "temperature": 0.7,
            "return_full_text": False
        }
    }

    try:
        logging.info(f"Sending request to Hugging Face API (Model: {MODEL_ID})...")
        response = query(payload, MODEL_ID, token)
        
        # Check for error in response
        if isinstance(response, dict) and 'error' in response:
            logging.warning(f"Primary model error: {response['error']}")
            raise Exception(response['error'])
            
        # Parse success (response is usually a list of dicts for text-generation)
        if isinstance(response, list) and 'generated_text' in response[0]:
            return response[0]['generated_text']
        elif isinstance(response, dict) and 'generated_text' in response:
             return response['generated_text']
        else:
             logging.error(f"Unexpected response structure: {response}")
             raise Exception("Unexpected response format")

    except Exception as e:
        logging.warning(f"Primary model {MODEL_ID} failed: {e}. Trying fallback model {FALLBACK_MODEL_ID}...")
        try:
            # Fallback
            response = query(payload, FALLBACK_MODEL_ID, token)
            if isinstance(response, list) and 'generated_text' in response[0]:
                return response[0]['generated_text']
            elif isinstance(response, dict) and 'generated_text' in response:
                return response['generated_text']
            else:
                 return f"Error: Fallback model response unclear. {response}"

        except Exception as e2:
            logging.error(f"Fallback model failed: {e2}")
            return "Error: Could not generate AI report. Please check your HF_TOKEN and internet connection."
