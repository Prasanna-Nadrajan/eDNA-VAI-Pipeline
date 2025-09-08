from flask import Flask, request, render_template, redirect, url_for, session, jsonify
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'super-secret-key'  # Change this for production!
CORS(app)

# Demo user data (username: password hash)
users = {
    "admin": generate_password_hash("password"),
    "alice": generate_password_hash("wonderland123"),
    "bob": generate_password_hash("builder456"),
}

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