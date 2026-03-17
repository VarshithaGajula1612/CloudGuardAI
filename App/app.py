import sqlite3
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "unmyeong")

DATABASE = 'database.db'

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            email TEXT NOT NULL UNIQUE,
            phone_number TEXT NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()

        if user and check_password_hash(user['password'], password):
            session['username'] = user['username']
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password.', 'error')

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        phone_number = request.form['phone_number']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template('register.html')

        conn = get_db_connection()
        user_check = conn.execute(
            'SELECT * FROM users WHERE username = ? OR email = ?',
            (username, email)
        ).fetchone()

        if user_check:
            flash('Username or email already exists. Please choose a different one.', 'error')
            conn.close()
            return render_template('register.html')
        hashed_password = generate_password_hash(password)

        try:
            conn.execute(
                'INSERT INTO users (username, email, phone_number, password) VALUES (?, ?, ?, ?)',
                (username, email, phone_number, hashed_password)
            )
            conn.commit()
            flash('Registration successful! You can now log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('An error occurred during registration. Please try again.', 'error')
        finally:
            conn.close()

    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

@app.route('/home')
def home():
    if 'username' not in session:
        flash('Please log in to access the home page.', 'error')
        return redirect(url_for('login'))

    conn = get_db_connection()
    user = conn.execute(
        'SELECT username, email, phone_number FROM users WHERE username = ?',
        (session['username'],)
    ).fetchone()
    conn.close()

    return render_template('home.html', user=user)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'username' not in session:
        flash('You must be logged in to make predictions.', 'warning')
        return redirect(url_for('login'))

    if request.method == 'POST':
        try:
            user_input = request.form['user_input']  

            ARTIFACT_DIR = "model/artifacts"
            model_path = os.path.join(ARTIFACT_DIR, "xgb_cybersecurity_model_gpu.pkl")
            le_target_path = os.path.join(ARTIFACT_DIR, "label_encoder_target.pkl")
            le_attack_path = os.path.join(ARTIFACT_DIR, "label_encoder_attack_type.pkl")
            le_high_path = os.path.join(ARTIFACT_DIR, "label_encoders_high_cardinality.pkl")

            model = joblib.load(model_path)
            le_target = joblib.load(le_target_path)
            le_attack = joblib.load(le_attack_path)
            le_high = joblib.load(le_high_path)
            columns = [
                "attack_type", "target_system", "attacker_ip", "target_ip", "data_compromised_GB",
                "attack_duration_min", "security_tools_used", "user_role", "location",
                "attack_severity", "industry", "response_time_min", "mitigation_method"
            ]

            values = [x.strip() for x in user_input.split(",")]
            if len(values) != len(columns):
                flash(f"Expected {len(columns)} values but got {len(values)}.", "error")
                return redirect(url_for('predict'))

            input_df = pd.DataFrame([values], columns=columns)
            num_cols = ["data_compromised_GB", "attack_duration_min", "attack_severity", "response_time_min"]
            input_df[num_cols] = input_df[num_cols].astype(float)

            for c in ["attacker_ip", "target_ip"]:
                if c in input_df.columns:
                    input_df = input_df.drop(columns=[c])

            input_df["data_per_min"] = input_df["data_compromised_GB"] / (input_df["attack_duration_min"] + 1)
            input_df["response_efficiency"] = input_df["response_time_min"] / (input_df["attack_duration_min"] + 1)
            input_df["severity_ratio"] = input_df["attack_severity"] / (input_df["response_time_min"] + 1)
            input_df["is_long_attack"] = (input_df["attack_duration_min"] > 300).astype(int)
            input_df = input_df.replace([np.inf, -np.inf], np.nan).fillna(0)

            input_df["attack_type"] = le_attack.transform(input_df["attack_type"])

            for col, enc in le_high.items():
                if col in input_df.columns:
                    input_df[col] = input_df[col].map(lambda x: x if x in enc.classes_ else enc.classes_[0])
                    input_df[col] = enc.transform(input_df[col])

            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0]
            pred_label = le_target.inverse_transform([int(pred)])[0]
            confidence = float(np.max(prob)) * 100

            conn = get_db_connection()
            c = conn.cursor()
            c.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    user_input TEXT NOT NULL,
                    predicted_label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()

            c.execute('''
                INSERT INTO predictions (username, user_input, predicted_label, confidence)
                VALUES (?, ?, ?, ?)
            ''', (session['username'], user_input, pred_label, confidence))
            conn.commit()
            conn.close()

            flash(f"Prediction successful! Result: {pred_label} (Confidence: {confidence:.2f}%)", 'success')
            return render_template('predict.html', prediction=pred_label, confidence=confidence)

        except Exception as e:
            flash(f"Prediction failed: {str(e)}", 'error')
            return redirect(url_for('predict'))

    return render_template('predict.html')



@app.route('/history')
def history():
    # Ensure user is logged in
    if 'username' not in session:
        flash('Please log in to view your prediction history.', 'warning')
        return redirect(url_for('login'))

    try:
        conn = get_db_connection()
        c = conn.cursor()
        # Fetch all predictions for the current user
        c.execute('''
            SELECT user_input, predicted_label, confidence, created_at
            FROM predictions
            WHERE username = ?
            ORDER BY created_at DESC
        ''', (session['username'],))
        predictions = c.fetchall()
        conn.close()

        # Pass to template
        return render_template('history.html', predictions=predictions)

    except Exception as e:
        flash(f"Error loading prediction history: {str(e)}", 'error')
        return redirect(url_for('predict'))

@app.route('/analytics')
def analytics():
    if 'username' not in session:
        flash('Please log in to view analytics.', 'warning')
        return redirect(url_for('login'))

    try:
        conn = get_db_connection()
        c = conn.cursor()

        c.execute('''
            SELECT predicted_label, COUNT(*) AS count
            FROM predictions
            WHERE username = ?
            GROUP BY predicted_label
        ''', (session['username'],))
        data = c.fetchall()
        conn.close()

        labels = [row['predicted_label'] for row in data]
        counts = [row['count'] for row in data]

        return render_template('analytics.html', labels=labels, counts=counts)

    except Exception as e:
        flash(f"Error loading analytics: {str(e)}", 'error')
        return redirect(url_for('predict'))


@app.route('/datascience')
def datascience():
    return render_template('datascience.html')

@app.route('/exsisting')
def exsisting():
    return render_template('exsisting.html')    

@app.route('/proposed')
def proposed():
    return render_template('proposed.html')

if __name__ == '__main__':
    debug = os.environ.get("FLASK_DEBUG", "").strip() in {"1", "true", "True", "yes", "YES"}
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=debug)
