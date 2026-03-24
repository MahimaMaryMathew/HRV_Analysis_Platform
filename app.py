
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass # python-dotenv not installed, assuming env vars differ

# Replace hardcoded strings with these:
CLIENT_ID = os.getenv('FITBIT_CLIENT_ID')
CLIENT_SECRET = os.getenv('FITBIT_CLIENT_SECRET')






   # required for login sessions
import sqlite3

def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS tokens
                 (email TEXT PRIMARY KEY, access_token TEXT, refresh_token TEXT, expires_at REAL)''')
    conn.commit()
    conn.close()

init_db()

import matplotlib
matplotlib.use('Agg')

from flask import Flask, render_template, jsonify, request, redirect, url_for, session

from authlib.integrations.flask_client import OAuth


import numpy as np
from scipy import signal
import os
import matplotlib.pyplot as plt
import io
import base64
from flask_cors import CORS
import logging
import requests

from requests_oauthlib import OAuth2Session

import os
# ALLOWS OAUTH TO WORK OVER HTTP (LOCAL ONLY)
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
# Set Flask logging to a higher level to see more debug info
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)
app.secret_key = os.getenv('FLASK_SECRET_KEY', "nsecret123")

# 1. Configuration
CLIENT_ID = os.getenv('FITBIT_CLIENT_ID')
CLIENT_SECRET = os.getenv('FITBIT_CLIENT_SECRET')
REDIRECT_URI = 'http://127.0.0.1:5000/callback'

SCOPE = ['heartrate', 'profile', 'activity']

# 2. Start Authorization
@app.route("/auth/fitbit")
def auth_fitbit():
     # This creates the Fitbit login link
    if not CLIENT_ID or not CLIENT_SECRET:
        return jsonify({'error': 'Missing Fitbit credentials. Please create a .env file with FITBIT_CLIENT_ID and FITBIT_CLIENT_SECRET.'}), 500

    # This creates the Fitbit login link
    fitbit = OAuth2Session(CLIENT_ID, scope=SCOPE, redirect_uri=REDIRECT_URI)
    authorization_url, state = fitbit.authorization_url("https://www.fitbit.com/oauth2/authorize")
    session['oauth_state'] = state # Protects against CSRF attacks
    return redirect(authorization_url)

# 3. Handle the Callback
@app.route("/callback")
def callback():
    # Setup the OAuth session using the state saved earlier
    fitbit = OAuth2Session(CLIENT_ID, state=session.get('oauth_state'), redirect_uri=REDIRECT_URI)
    
    # Fetch the actual token from Fitbit
    token = fitbit.fetch_token(
        "https://api.fitbit.com/oauth2/token",
        client_secret=CLIENT_SECRET,
        authorization_response=request.url
    )
    
    # 1. Save token to browser session
    session['fitbit_token'] = token 
    session.permanent = True 
    
    # 2. Save to Database for long-term use
    email = session.get('user')
    if email:
        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute('REPLACE INTO tokens VALUES (?, ?, ?, ?)', 
                      (email, token['access_token'], token['refresh_token'], token['expires_at']))
            conn.commit()
            conn.close()
        except Exception as e:
            logging.error(f"Database error: {e}")
    
    # 3. Final Redirect
    return redirect(url_for('dashboard', tab='device'))


def get_db_token(email):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT access_token, refresh_token, expires_at FROM tokens WHERE email=?', (email,))
    row = c.fetchone()
    conn.close()
    if row:
        return {'access_token': row[0], 'refresh_token': row[1], 'expires_at': row[2]}
    return None

def get_valid_token():
    token = session.get('fitbit_token')
    if not token:
        return None

    # Check if token is expired (or about to expire in 60 seconds)
    from time import time
    if token.get('expires_at') and token['expires_at'] < time() + 60:
        extra = {'client_id': CLIENT_ID, 'client_secret': CLIENT_SECRET}
        fitbit = OAuth2Session(CLIENT_ID, token=token)
        
        # Request a new access token using the refresh token
        new_token = fitbit.refresh_token("https://api.fitbit.com/oauth2/token", **extra)
        session['fitbit_token'] = new_token
        return new_token
    
    return token

# 4. Fetch and Process Data
@app.route("/analyze_fitbit")
def analyze_fitbit():
    # 1. Get a fresh token (Checks if expired and refreshes automatically)
    token = get_valid_token() 
    if not token:
        return jsonify({'error': 'Fitbit not connected or session expired. Please login.'}), 401

    url = "https://api.fitbit.com/1/user/-/activities/heart/date/today/1d/1sec.json"
    headers = {'Authorization': f"Bearer {token['access_token']}"}
    
    try:
        response = requests.get(url, headers=headers)
        data = response.json()

        # 2. Check for API-level errors (like 'Forbidden' or 'Rate Limit')
        if 'errors' in data:
            return jsonify({'error': data['errors'][0]['message']}), 403

        # 3. Safely extract dataset using .get() to avoid KeyErrors
        intraday_data = data.get('activities-heart-intraday', {}).get('dataset', [])

        if not intraday_data:
            return jsonify({'error': 'No heart rate data found for today. Wear your tracker and sync!'}), 404

        # 4. Convert BPM to RR intervals
        rr_intervals = [60000 / entry['value'] for entry in intraday_data]

        if len(rr_intervals) < 20: # Minimum threshold for a quick check
            return jsonify({'error': 'Insufficient data points recorded for a valid HRV analysis.'}), 400

        # 5. Process with existing HRV logic
        results = calculate_hrv_parameters(np.array(rr_intervals))
        
        # Save results globally for the feedback function
        global last_analysis_results
        last_analysis_results = results
        
        return jsonify(results)

    except Exception as e:
        logging.error(f"Fitbit analysis failed: {str(e)}")
        return jsonify({'error': 'Internal server error during data processing.'}), 500







last_analysis_results = {}
@app.route("/", methods=["GET"])
def home():
    global users
    if "user" in session:
        return redirect("/dashboard")
    return redirect("/login")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        with open("users.json", "r") as f:
            users = json.load(f)


        print("Saving user:", email, password)
        print("Users before save:", users)
        # Load users
       

        if email in users and users[email] == password:
            session["user"] = email
            return redirect("/dashboard")
        else:
            return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")
@app.route('/dashboard')
def dashboard():
    if "user" not in session:
        return redirect("/login")
    return render_template('index.html')
# Registration route
import json

import os

@app.route("/register", methods=["GET", "POST"])
def register():
    # Ensure users.json exists
    if not os.path.exists("users.json"):
        with open("users.json", "w") as f:
            json.dump({"admin": "123"}, f)

    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        # Load users
        with open("users.json", "r") as f:
            users = json.load(f)

        # Check if user exists
        if email in users:
            return render_template("register.html", error="User already exists")

        # Save new user
        users[email] = password
        with open("users.json", "w") as f:
            json.dump(users, f)

        return redirect("/login")

    # GET request → show register page
    return render_template("register.html")


       
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/login")

@app.route("/terms")
def terms():
    return render_template("terms.html")

@app.route("/privacy")
def privacy():
    return render_template("privacy.html")

# HRV Analysis Functions
def generate_mock_rr_intervals(num_points=500):
    """
    Generates a mock time series of RR intervals for demonstration.
    """
    base_rr = np.random.uniform(700, 950)
    rr_intervals = np.random.normal(base_rr, 50, num_points)
    sin_wave = np.random.uniform(10, 30) * np.sin(np.linspace(0, np.random.uniform(1, 3) * np.pi, num_points))
    rr_intervals += sin_wave
    shift_start = np.random.randint(100, num_points - 100)
    shift_end = shift_start + np.random.randint(5, 15)
    rr_intervals[shift_start:shift_end] += np.random.uniform(30, 80)
    rr_intervals = np.clip(rr_intervals, 400, 1500)
    return rr_intervals

def generate_plot_base64(rr_intervals, sd1, sd2):
    """
    Generates HRV plots and returns them as a base64 encoded string.
    """
    fs = 1000 / np.mean(rr_intervals)
    freqs, pxx = signal.welch(x=rr_intervals, fs=fs, nperseg=256)
    
    img = io.BytesIO()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    ax1.plot(rr_intervals, color='b')
    ax1.set_title('Time-Domain Tachogram')
    ax1.set_xlabel('Beat Number')
    ax1.set_ylabel('RR Interval (ms)')
    ax1.grid(True)
    
    ax2.plot(freqs, pxx, color='g')
    ax2.set_title('Frequency-Domain Power Spectrum')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Power ($ms^2/Hz$)')
    ax2.set_xlim(0, 0.5)
    ax2.grid(True)
    
    rr_x = rr_intervals[:-1]
    rr_y = rr_intervals[1:]
    ax3.scatter(rr_x, rr_y, alpha=0.5, color='r')
    ax3.set_title('Poincaré Plot')
    ax3.set_xlabel('$RR_n (ms)$')
    ax3.set_ylabel('$RR_{n+1} (ms)$')
    ax3.text(1.05, 0.5, f'SD1: {sd1:.2f} ms\nSD2: {sd2:.2f} ms',
             transform=ax3.transAxes, ha='left', va='center')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig(img, format='png')
    plt.close(fig)
    img.seek(0)
    
    plot_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    return plot_base64

def calculate_hrv_parameters(rr_intervals):
    """
    Calculates various HRV parameters from RR-interval data.
    """
    mean_rr = np.mean(rr_intervals)
    sdnn = np.std(rr_intervals, ddof=1)
    rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
    
    fs = 1000.0 / mean_rr
    freqs, pxx = signal.welch(x=rr_intervals, fs=fs, nperseg=256)
    
    vlf_band = (0.003, 0.04)
    lf_band = (0.04, 0.15)
    hf_band = (0.15, 0.4)
    
    # Using np.trapezoid instead of np.trapz to address the DeprecationWarning
    vlf_power = np.trapezoid(pxx[(freqs >= vlf_band[0]) & (freqs < vlf_band[1])], freqs[(freqs >= vlf_band[0]) & (freqs < vlf_band[1])])
    lf_power = np.trapezoid(pxx[(freqs >= lf_band[0]) & (freqs < lf_band[1])], freqs[(freqs >= lf_band[0]) & (freqs < lf_band[1])])
    hf_power = np.trapezoid(pxx[(freqs >= hf_band[0]) & (freqs < hf_band[1])], freqs[(freqs >= hf_band[0]) & (freqs < hf_band[1])])
    
    lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0
    
    sd1 = np.std(np.diff(rr_intervals) / np.sqrt(2))
    sd2 = np.std((rr_intervals[:-1] + rr_intervals[1:]) / np.sqrt(2))
    
    return {
        'mean_rr': mean_rr,
        'sdnn': sdnn,
        'rmssd': rmssd,
        'lf_hf_ratio': lf_hf_ratio,
        'sd1': sd1,
        'sd2': sd2
    }


@app.route('/analyze_hrv', methods=['POST'])
def analyze_hrv_with_results_storage():
    global last_analysis_results
    logging.info("Starting HRV data analysis...")

    request_data = request.get_json(silent=True)
    if not request_data or 'method' not in request_data:
        return jsonify({'error': 'Invalid request body. Method not specified.'}), 400

    data = request_data.get('data')
    method = request_data.get('method')
    
    rr_intervals = None
    if method == 'csv' or method == 'manual':
        if not data:
            return jsonify({'error': 'No data provided for the selected method.'}), 400
        try:
            rr_intervals_str = data.split(',')
            rr_intervals_list = [float(rr.strip()) for rr in rr_intervals_str if rr.strip()]
        except ValueError:
            return jsonify({'error': 'Invalid data format. Please provide comma-separated numbers only.'}), 400
        
        if len(rr_intervals_list) < 200:
            return jsonify({'error': 'Insufficient data. Please provide at least 200 RR-intervals for proper analysis.'}), 400
        
        rr_intervals = np.array(rr_intervals_list)
    elif method == 'device':
        token = session.get('fitbit_token')
        if not token:
            return jsonify({'error': 'Fitbit not connected. Please sync first.'}), 401
        
        # API Request for real heart rate data
        url = "https://api.fitbit.com/1/user/-/activities/heart/date/today/1d/1sec.json"
        headers = {'Authorization': f"Bearer {token['access_token']}"}
        
        try:
            response = requests.get(url, headers=headers)
            data = response.json()
            dataset = data['activities-heart-intraday']['dataset']
            
            if not dataset:
                return jsonify({'error': 'No heart rate data found for today. Wear your tracker!'}), 404
            
            # Convert BPM to RR Intervals (ms) -> 60000 / BPM
            rr_intervals = [60000 / entry['value'] for entry in dataset]
            
        except Exception as e:
            return jsonify({'error': f'Failed to fetch Fitbit data: {str(e)}'}), 500

    elif method == 'mock':
        rr_intervals = generate_mock_rr_intervals()
    else:
        return jsonify({'error': 'Invalid analysis method specified.'}), 400

    try:
        rr_intervals = np.array(rr_intervals)
        
        analysis_results = calculate_hrv_parameters(rr_intervals)
        last_analysis_results = analysis_results
        
        sd1 = analysis_results['sd1']
        sd2 = analysis_results['sd2']
        
        plot_base64 = generate_plot_base64(rr_intervals, sd1, sd2)

        logging.info("Analysis complete. Sending results to frontend.")
        return jsonify({
            'results': analysis_results,
            'plot': plot_base64
        })
    except Exception as e:
        # We will now print the actual error to the console for better debugging.
        logging.error(f"An error occurred during analysis: {e}", exc_info=True)
        return jsonify({'error': 'An error occurred during analysis. Please try again.'}), 500

from remote_ai import get_feedback_from_ai

@app.route('/get_feedback', methods=['GET'])
def get_feedback():
    global last_analysis_results
    
    # Check if we have results to analyze
    if not last_analysis_results:
         return jsonify({'error': 'No analysis results found. Please run an HRV analysis first.'}), 400

    logging.info("Requesting AI feedback from Hugging Face API...")
    try:
        feedback_text = get_feedback_from_ai(last_analysis_results)
        return jsonify({'feedback': feedback_text})
    except Exception as e:
        logging.error(f"Error in feedback route: {e}")
        return jsonify({'error': 'Failed to generate feedback.'}), 500

if __name__ == '__main__':
      app.run(host='0.0.0.0', port=5000, debug=True)
