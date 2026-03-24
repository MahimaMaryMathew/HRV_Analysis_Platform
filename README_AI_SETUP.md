# AI Setup Instructions

You have successfully switched to the **Hugging Face Inference API** for HRV analysis.
This allows you to run advanced AI models without needing a powerful GPU on your local machine.

## Prerequisites
1.  **Python 3.8+**
2.  **Internet Connection** (to reach Hugging Face servers)

## Step 1: Get your API Token
1.  Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
2.  Log in or Sign up.
3.  Click **"Create new token"**.
4.  Name it (e.g., "HRV App") and give it **"Read"** permissions (default).
5.  Copy the token (starts with `hf_...`).

## Step 2: Configure Environment
1.  In the project folder, create a file named `.env` (if it doesn't exist).
2.  Add your token to it:
    ```bash
    HF_TOKEN=your_copied_token_here
    ```
    *(See `.env.example` for a template)*

## Step 3: Clean Install (Optional but recommended)
Since we changed some logic, verify your dependencies:
```bash
pip install -r requirements.txt
```
*Note: If you are in a restricted environment, `requests` is the only strict requirement for the AI module, which is likely already installed.*

## Step 4: Run the App
```bash
python3 app.py
```
Access the dashboard at `http://localhost:5000`.

## Features
- **Model**: `convaiinnovations/medgemma-4b-ecginstruct` (Medical specialized)
- **Fallback**: `google/gemma-2-9b-it` (General purpose)
- **Privacy**: Data is sent to Hugging Face for inference only.
