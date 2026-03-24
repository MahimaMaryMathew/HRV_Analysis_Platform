import os
import logging
import requests
import base64
from io import BytesIO
from PIL import Image

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logging.warning("python-dotenv not found. Relying on environment variables.")

# -----------------
# HF Router Model
# -----------------
API_URL = "https://router.huggingface.co/v1/chat/completions"
ROUTER_MODEL_ID = "google/gemma-3-27b-it"

def get_feedback_from_ai(hrv_metrics):
    """
    Generates an educational HRV report using the Hugging Face Inference Router.
    """
    token = os.getenv('HF_TOKEN')
    if not token:
        logging.error("HF_TOKEN not found in environment variables.")
        return "Error: Hugging Face API Token (HF_TOKEN) is missing. Please add it to your .env file."

    # Calculate derived parameters for the Master Prompt
    sd1 = hrv_metrics.get('sd1', 0)
    sd2 = hrv_metrics.get('sd2', 1)
    sd1_sd2_ratio = sd1 / sd2 if sd2 > 0 else 0
    
    if sd1_sd2_ratio > 0.8:
        shape = "Round/Parasympathetic Dominance" # Typical of deep breathing or elite athletes
    elif sd1_sd2_ratio >= 0.3:
        shape = "Healthy Ellipse/Comet"
    else:
        shape = "Torpedo/Highly constrained"

    mean_rr = hrv_metrics.get('mean_rr', 800)
    mean_hr = 60000 / mean_rr if mean_rr else 0

    # Format the Master Prompt
    system_prompt = (
        "Role: Act as an expert in Autonomic Physiology and Clinical Data Science. Your goal is to provide a "
        "multi-domain interpretation of Heart Rate Variability (HRV) data for a user's health report.\n\n"
        "Instructions for Analysis:\n"
        "1. Synthesize, Don't List: Do not define the terms in isolation. Explain how they interact.\n"
        "2. Evaluate Autonomic Reserve: Specifically address 'Autonomic Flexibility.' Is the heart's ability to adapt "
        "consistent across different heart rates, or does it collapse under stress?\n"
        "3. Identify the 'Visual Signature': Interpret the Poincaré shape.\n"
        "   - If Comet/Ellipse: If the time-domain metrics are normal and the Poincaré plot shows a standard comet/cigar shape, explicitly state that this indicates excellent health. Do not over-pathologize or invent 'emerging restrictions' out of standard healthy baseline data.\n"
        "   - If Round/Parasympathetic Dominance: Focus on the immense vagal tone/parasympathetic activity. This suggests deep relaxation, highly athletic cardiovascular efficiency, or deep diaphragmatic breathing. Praise this as excellent recovery capacity.\n"
        "   - If Torpedo/Highly constrained: Focus on sympathetic dominance, stress, or 'Autonomic Exhaustion'.\n"
        "   - If Chaotic: Address potential rhythm irregularities (AFib).\n"
        "4. Actionable Intelligence: Provide three specific, technical recommendations. Do not give generic advice like 'sleep more.' "
        "Suggest specific interventions like 'Zone 2 Aerobic Base building,' 'Vagal Nerve stimulation via exhale-prolonged breathing,' "
        "or 'Immediate Rest for CNS recovery.' If the data is incredibly healthy/athletic, recommend 'Maintenance and progressive overload' rather than recovery.\n"
        "5. Tone: Professional, clinical yet accessible, and encouraging but direct about potential 'unhealthy' signatures.\n\n"
        "Output Format:\n"
        "- **Executive Summary**: The 'Bottom Line' of the heart's current state.\n"
        "- **The Three-Domain Deep Dive**: Connecting Time, Frequency, and Geometry.\n"
        "- **Autonomic Flexibility Score**: A qualitative assessment of cardiac reserve.\n"
        "- **Strategic Recommendations**: Data-driven next steps.\n\n"
        "IMPORTANT: Do not provide a formal medical diagnosis."
    )
    
    user_content = (
        f"Input Data:\n"
        f"* Time Domain: RMSSD {hrv_metrics.get('rmssd', 0):.2f} ms, SDNN {hrv_metrics.get('sdnn', 0):.2f} ms.\n"
        f"* Frequency Domain: LF/HF Ratio of {hrv_metrics.get('lf_hf_ratio', 0):.2f}.\n"
        f"* Non-Linear Domain (Poincaré): Shape is {shape} (SD1/SD2 ratio = {sd1_sd2_ratio:.2f}).\n"
        f"* Context: Heart rate is {mean_hr:.0f} bpm."
    )

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": ROUTER_MODEL_ID,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "max_tokens": 2000,
        "temperature": 0.7
    }

    try:
        logging.info(f"Sending request to Hugging Face Router (Model: {ROUTER_MODEL_ID})...")
        response = requests.post(API_URL, headers=headers, json=payload)
        
        if response.status_code != 200:
            logging.error(f"API Error {response.status_code}: {response.text}")
            return f"Error: API Request Failed. Status: {response.status_code}"

        result = response.json()
        
        # Parse OpenAI-compatible response
        if 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['message']['content']
        else:
            logging.error(f"Unexpected response structure: {result}")
            return "Error: Unexpected response format from AI."

    except Exception as e:
        logging.error(f"AI Request Failed: {e}")
        return "Error: Could not generate AI report. Please check your internet connection."


# -----------------
# HF Router Model (Vision / ECG)
# -----------------
VISION_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

def analyze_ecg_with_ai(image_data_url):
    """
    Analyzes an ECG image using the Hugging Face Inference Router with Qwen2.5-VL-7B-Instruct.
    Uses a highly detailed multi-layered cardiology prompt.
    """
    token = os.getenv('HF_TOKEN')
    if not token:
        logging.error("HF_TOKEN not found in environment variables.")
        return "Error: Hugging Face API Token (HF_TOKEN) is missing. Please add it to your .env file."
        
    try:
        logging.info(f"Sending ECG request to Hugging Face Router (Model: {VISION_MODEL_ID})...")
        
        system_prompt = (
            "Act as a highly sensitive ECG Anomaly Detector. Your primary goal is to flag ANY finding that deviates from a perfect, textbook normal sinus rhythm.\n\n"
            "Layer 1: Rigorous Technical Scan\n"
            "- Rate & Rhythm: Explicitly measure the R-R interval between EVERY QRS complex. If there is ANY variation in the R-R distance (even subtle changes with breathing), you MUST flag this as 'Sinus Arrhythmia' or 'Irregular Rhythm'. DO NOT gloss over small variations.\n"
            "- P-waves, PR Interval, QRS, ST-segment, T-waves: Scrutinize all waveforms for any slight deviation (e.g., borderline wide QRS, slight ST elevation/depression, flat or inverted T-waves in any lead).\n\n"
            "Layer 2: Strict Clinical Reasoning\n"
            "- You must adopt a 'better safe than sorry' approach. If a feature is borderline or ambiguous, label it as an 'Anomaly' or 'Deviant Finding'.\n"
            "- Sinus Arrhythmia is extremely common; if the R-R interval is not perfectly identical across the entire strip, report Sinus Arrhythmia.\n\n"
            "Layer 3: Simple Language Summary & Verdict\n"
            "- Provide a 2-3 sentence summary in plain language.\n"
            "- VERDICT RULE: Only output 'Normal' if the ECG is a mathematically perfect text-book example. If YOU DETECTED ANY VARIATION (including benign Sinus Arrhythmia), the verdict MUST BE 'Atypical finding detected'.\n"
            "- If Sinus Arrhythmia is the only finding, reassure the user that this is a normal physiological variation related to breathing.\n"
            "Feedback & Next Steps: Provide immediate next actions (e.g., 'No action needed', 'Consult a doctor').\n"
            "Disclaimer: This is an AI-generated analysis for research only and is not a medical diagnosis."
        )

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": VISION_MODEL_ID,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                        {"type": "text", "text": system_prompt}
                    ]
                }
            ],
            "max_tokens": 800,
            "temperature": 0.3
        }
        
        response = requests.post(API_URL, headers=headers, json=payload)
        
        if response.status_code != 200:
            logging.error(f"API Error {response.status_code}: {response.text}")
            return f"Error: API Request Failed. Status: {response.status_code}"

        result = response.json()
        
        # Parse OpenAI-compatible response
        if 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['message']['content']
        else:
            logging.error(f"Unexpected response structure: {result}")
            return "Error: Unexpected response format from AI."

    except Exception as e:
        logging.error(f"AI Vision Request Failed: {e}")
        return "Error: Could not analyze ECG image. Please check your internet connection."


# -----------------
# Custom Trained HF Model (Image Classification) — runs LOCALLY
# -----------------
# Model repo — the MixUp/CutMix augmented model lives on the 'mixup-cutmix' branch.
# To fall back to the original model, set DEFAULT_CUSTOM_ECG_REVISION = None (loads 'main').
DEFAULT_CUSTOM_ECG_MODEL    = "MMM0003/ecg-classifier"
DEFAULT_CUSTOM_ECG_REVISION = "mixup-cutmix"   # branch pushed after MixUp training

# Cache the pipeline so the model only loads once (first request ~30s, afterwards instant)
# Key format: "model_id@revision" to avoid stale cache on branch switches
_ecg_pipeline_cache = {}

def analyze_ecg_with_custom_hf_model(image_data_url, custom_model_id=None, revision=None):
    """
    Analyzes an ECG image using a custom fine-tuned Hugging Face Image Classification Model.
    Runs the model LOCALLY using the transformers pipeline (downloaded from HF Hub).

    Args:
        image_data_url  : base64 data URL of the ECG image
        custom_model_id : HF repo id (default: DEFAULT_CUSTOM_ECG_MODEL)
        revision        : branch/tag/commit to load (default: DEFAULT_CUSTOM_ECG_REVISION)
    """
    if not custom_model_id:
        custom_model_id = DEFAULT_CUSTOM_ECG_MODEL
    if revision is None:
        revision = DEFAULT_CUSTOM_ECG_REVISION

    # Cache key includes revision so branch switches don't serve stale weights
    cache_key = f"{custom_model_id}@{revision}"

    # Extract image from data URL
    try:
        if "base64," in image_data_url:
            base64_img = image_data_url.split("base64,")[1]
        else:
            base64_img = image_data_url
        img_data = base64.b64decode(base64_img)
    except Exception as e:
        logging.error(f"Error decoding image: {e}")
        return "Error: Invalid image format."

    try:
        from transformers import pipeline as hf_pipeline
        from PIL import Image as PILImage

        # Load or retrieve cached pipeline
        if cache_key not in _ecg_pipeline_cache:
            logging.info(f"Loading ECG model '{custom_model_id}' revision='{revision}' ...")
            _ecg_pipeline_cache[cache_key] = hf_pipeline(
                "image-classification",
                model=custom_model_id,
                revision=revision,
                device=-1,  # CPU (-1). Use 0 for GPU if available.
            )
            logging.info(f"Model loaded successfully!")

        classifier = _ecg_pipeline_cache[cache_key]

        # Convert bytes to PIL Image
        img = PILImage.open(BytesIO(img_data)).convert("RGB")

        # Run inference
        logging.info(f"Running ECG classification with '{custom_model_id}'...")
        results = classifier(img)

        if results and len(results) > 0:
            top_prediction = results[0]['label']
            top_score = results[0]['score'] * 100

            # Map the clinical classes from GenECG/PTB-XL to human-readable explanations
            explanations = {
                "NORM": "Normal Sinus Rhythm. The electrical pathways of the heart appear to be functioning routinely with no obvious major abnormalities detected.\nEverything looks good! Maintain your healthy lifestyle.",
                "MI": "Possible Myocardial Infarction detected. Patterns suggest a potential history of or current heart muscle damage. This is a significant finding that requires medical attention.\nWe see signs of potential heart muscle damage. Please consult a doctor for a proper checkup.",
                "STTC": "ST/T Wave Changes detected. This indicates possible ischemia (reduced blood flow to the heart muscle) or other ventricular repolarization abnormalities.\nThere are subtle changes in how your heart recovers between beats. It's recommended to have a doctor review this.",
                "CD": "Conduction Disturbance detected. There may be a delay or blockage in the electrical pathways that tell the heart muscles when to contract.\nThe electrical signals in your heart might be slightly delayed. A doctor should take a look to be sure.",
                "HYP": "Possible Hypertrophy detected. The patterns suggest an enlargement or thickening of the heart muscles, often related to high blood pressure.\nThe heart muscle appears slightly thickened, which can happen with high blood pressure. Follow up with your healthcare provider."
            }

            interpretation = explanations.get(top_prediction, "Atypical findings detected based on the trained model characteristics.")

            return interpretation
        else:
            return "Error: Model returned empty results."

    except Exception as e:
        import traceback
        logging.error(f"Custom ECG model error:\n{traceback.format_exc()}")
        return f"Error running custom model: {str(e)[:500]}"


