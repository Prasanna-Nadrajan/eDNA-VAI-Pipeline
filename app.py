from flask import Flask, request, render_template, redirect, url_for, session, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import json
import os
import sys

# Add the parent directory of main.py to the Python path
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

# Upload settings
UPLOAD_FOLDER = os.path.join(app.root_path, 'uploads')
ALLOWED_EXTENSIONS = {'fasta', 'fa', 'fna', 'fastq', 'fq'}
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
            try:
                os.remove(os.path.join(PLOTS_DIR, f))
            except:
                pass

        # Check if using sample file
        use_sample = request.form.get('use_sample', 'false').lower() == 'true'
        
        # Check for file upload
        uploaded_file = request.files.get('file')
        fasta_file = None
        fasta_content = None
        
        if use_sample:
            # Use the sample.fasta file
            sample_path = os.path.join(app.root_path, 'sample.fasta')
            if os.path.exists(sample_path) and os.path.getsize(sample_path) > 0:
                fasta_file = sample_path
                app.logger.info(f"Using sample FASTA file: {sample_path}")
            else:
                return jsonify({'error': 'Sample FASTA file not found or empty'}), 400
                
        elif uploaded_file and uploaded_file.filename:
            if allowed_file(uploaded_file.filename):
                # Save the uploaded file
                filename = secure_filename(uploaded_file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                uploaded_file.save(filepath)
                fasta_file = filepath
                app.logger.info(f"Using uploaded file: {filepath}")
            else:
                return jsonify({'error': 'Invalid file type. Please upload a .fasta, .fa, .fna, .fastq, or .fq file'}), 400
        
        # Initialize pipeline with plots directory
        pipeline = eDNABiodiversityPipeline(
            n_sequences=2000,  # Fallback for mock data
            sequence_length=200, 
            k=4,
            plots_dir=PLOTS_DIR
        )
        
        # Run the pipeline
        if fasta_file:
            results = pipeline.run_full_pipeline(fasta_file=fasta_file)
        elif fasta_content:
            results = pipeline.run_full_pipeline(fasta_content=fasta_content)
        else:
            # No file provided - use mock data generation
            app.logger.info("No file provided, using mock data generation")
            results = pipeline.run_full_pipeline()
        
        # Return the results as JSON
        return jsonify(results)
        
    except Exception as e:
        app.logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
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
    # use_reloader=False prevents TensorFlow from being loaded twice
    app.run(debug=True, use_reloader=False)
