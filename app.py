from flask import Flask, request, render_template, redirect, url_for, session, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
import json
import os
import sys

# Add the parent directory of main.py to the Python path
# This allows Flask to import the module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from main import eDNABiodiversityPipeline

app = Flask(__name__)
app.secret_key = 'super-secret-key'  # Change this for production!
CORS(app)

# Demo user data (username: password hash)
users = {
    "admin": generate_password_hash("password"),
    "alice": generate_password_hash("wonderland123"),
    "bob": generate_password_hash("builder456"),
}

# Ensure the plots directory exists
PLOTS_DIR = os.path.join(app.root_path, 'plots')
if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()
    if username in users and check_password_hash(users[username], password):
        session['username'] = username
        return jsonify({'success': True, 'username': username})
    return jsonify({'success': False, 'error': 'Invalid username or password.'}), 401

@app.route('/api/logout', methods=['POST'])
def logout():
    session.pop('username', None)
    return jsonify({'success': True})

@app.route('/api/check_auth', methods=['GET'])
def check_auth():
    return jsonify({'authenticated': 'username' in session, 'username': session.get('username')})

@app.route('/api/analyze', methods=['POST'])
def analyze_edna():
    try:
        # Clear old plots before running the analysis
        for f in os.listdir(PLOTS_DIR):
            os.remove(os.path.join(PLOTS_DIR, f))

        # Initialize and run the pipeline
        pipeline = eDNABiodiversityPipeline(n_sequences=2000, sequence_length=200, k=4)
        results = pipeline.run_full_pipeline()
        
        # Return the results as JSON
        return jsonify(results)
    except Exception as e:
        app.logger.error(f"Analysis failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/plots/<path:filename>')
def serve_plot(filename):
    return send_from_directory(PLOTS_DIR, filename)

@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('home'))
    return render_template('login.html')

@app.route('/home.html')
def home():
    if 'username' not in session:
        return redirect(url_for('index'))
    return render_template('home.html', username=session['username'])

if __name__ == '__main__':
    app.run(debug=True)
