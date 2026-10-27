import numpy as np
import pandas as pd
import nltk
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from nltk.sentiment import SentimentIntensityAnalyzer
import ollama
import logging
import joblib
from tensorflow.keras.models import load_model
import csv
from datetime import datetime
import os
from flask import Response
from camera import VideoCamera

# Download VADER Lexicon
nltk.download('vader_lexicon')

# Initialize Flask App
app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = 'your_secret_key'
logging.basicConfig(filename='user_logs.log', level=logging.INFO)

# Load LSTM Model, Scaler, Encoder
lstm_model = load_model("lstm_stress_model.h5")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# ====================================
# USERS STORAGE (FIXED)
# ====================================
users = {}

# Ensure CSV exists (NOW INCLUDING PASSWORD)
def ensure_csv(file, headers):
    if not os.path.exists(file):
        with open(file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

ensure_csv("users.csv", ["username", "age", "password"])
ensure_csv("stress_history.csv", ["username", "stress_level", "message", "timestamp"])
ensure_csv("phq9_history.csv", ["username", "score", "severity", "timestamp"])
ensure_csv("chat_sentiment.csv", ["username", "user_message", "sentiment", "timestamp"])

# ====================================
# LOAD USERS FROM CSV AT STARTUP
# ====================================
if os.path.exists("users.csv"):
    with open("users.csv", "r") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) == 3:
                username, age, password = row
                users[username] = {"password": password, "age": age}

# ====================================
# ALWAYS ENSURE ADMIN EXISTS
# ====================================
users["admin"] = {"password": "admin", "age": "0"}

chat_histories = {}



# ============================================================
# CSV INITIALIZATION
# ============================================================

def ensure_csv(file, headers):
    if not os.path.exists(file):
        with open(file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

ensure_csv("users.csv", ["username", "age"])
ensure_csv("stress_history.csv", ["username", "stress_level", "message", "timestamp"])
ensure_csv("phq9_history.csv", ["username", "score", "severity", "timestamp"])
ensure_csv("chat_sentiment.csv", ["username", "user_message", "sentiment", "timestamp"])


# ============================================================
# ROUTES
# ============================================================

@app.route('/')
def index():
    return render_template("index.html")
@app.route('/emotion')
def emotion_page():
    return render_template("emotion.html")
@app.route('/emotion_api')
def emotion_api():
    return jsonify({"emotion": VideoCamera.last_emotion})


def gen(camera):
    while True:
        frame, emotion= camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        age = request.form['age']
        password = request.form['password']

        # Check if exists already
        if username in users:
            return render_template("signup.html", error="Username already exists")

        # Save in memory
        users[username] = {"password": password, "age": age}

        # Save to CSV (NOW WITH PASSWORD)
        with open("users.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([username, age, password])

        return redirect(url_for('login'))

    return render_template("signup.html")



# ---------------------------- LOGIN ---------------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = users.get(username)

        if user and user['password'] == password:
            session['user'] = username
            return redirect(url_for('profile'))

        return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")


# ---------------------------- LOGOUT --------------------------
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('index'))


# ============================================================
# SENTIMENT ANALYSIS PAGE
# ============================================================

@app.route('/vader')
def vader_page():
    return render_template("home.html")


@app.route('/vader_stress', methods=['POST'])
def vader_stress():
    user_text = request.form.get("user_text", "")
    score = sia.polarity_scores(user_text)
    compound = score["compound"]

    sentiment = (
        "positive" if compound >= 0.05 else
        "negative" if compound <= -0.05 else
        "neutral"
    )

    return render_template("home.html", sentiment=sentiment, user_text=user_text)


# ============================================================
# CHATBOT (Ollama LLM)
# ============================================================

@app.route('/chat')
def chat():
    return render_template("chat.html")


@app.route('/chat_api', methods=['POST'])
def chat_api():
    try:
        data = request.get_json()
        user_msg = data.get("message", "")
        username = session.get('user', 'anonymous')

        # Save sentiment + chat message
        sentiment_score = sia.polarity_scores(user_msg)
        sentiment = (
            "positive" if sentiment_score['compound'] >= 0.05 else
            "negative" if sentiment_score['compound'] <= -0.05 else
            "neutral"
        )

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        with open("chat_sentiment.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([username, user_msg, sentiment, timestamp])

        # Maintain conversation history
        if username not in chat_histories:
            chat_histories[username] = []
        chat_histories[username].append({"role": "user", "content": user_msg})

        # Prompt for LLM
        prompt = (
            "You are a warm, empathetic mental health assistant.\n"
            "Provide short, supportive, calming responses.\n"
        )

        for msg in chat_histories[username][-5:]:
            prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"

        prompt += "Assistant:"

        response = ollama.chat(
            model="mistral",
            messages=[{"role": "user", "content": prompt}]
        )

        ai_reply = response['message']['content'].strip()
        short_reply = " ".join(ai_reply.split()[:60])

        chat_histories[username].append({"role": "assistant", "content": short_reply})

        return jsonify({"response": short_reply, "sentiment": sentiment})

    except Exception:
        return jsonify({"response": "⚠️ Chatbot not available.", "sentiment": "neutral"})


# ============================================================
# STRESS QUESTIONNAIRE (LSTM MODEL)
# ============================================================

@app.route('/questionnaire', methods=['GET', 'POST'])
def questionnaire():
    if request.method == 'POST':
        try:
            fields = [
                'anxiety_level', 'mental_health_history', 'depression', 'headache',
                'sleep_quality', 'breathing_problem', 'living_conditions',
                'daily_responsibility', 'day_stress', 'future_career_concerns',
                'extracurricular_activities'
            ]

            values = [float(request.form.get(f)) for f in fields]
            array = np.array(values).reshape(1, -1)

            scaled = scaler.transform(array)
            reshaped = scaled.reshape(1, 1, scaled.shape[1])

            prediction = lstm_model.predict(reshaped)
            pred_class = np.argmax(prediction)
            label = label_encoder.inverse_transform([pred_class])[0]

            # Stress Response Dictionary
            stress_responses = {
                0: {
                    "label": "No Significant Stress",
                    "message": (
                        "You appear emotionally steady and managing your daily challenges well. "
                        "Your responses indicate a balanced mental state with no significant signs of stress."
                    ),
                    "suggestions": [
                        "• Maintain a consistent sleep routine to support emotional stability.",
                        "• Engage in enjoyable hobbies or creative activities that bring you joy.",
                        "• Stay physically active—regular walks, yoga, or light exercises keep your mind clear.",
                        "• Practice gratitude journaling or short mindfulness moments to reinforce positivity.",
                        "• Maintain social connections with family and friends to keep your mood healthy."
                    ]
                },

                1: {
                    "label": "Moderate Stress",
                    "message": (
                        "You may be experiencing mild to moderate stress. This is quite common and often "
                        "related to daily responsibilities, lifestyle pressures, or emotional strain. "
                        "Your responses suggest that some areas of your well-being could use attention."
                    ),
                    "suggestions": [
                        "• Take short breaks throughout the day to prevent mental fatigue.",
                        "• Practice deep breathing or 4-7-8 breathing exercises to calm the nervous system.",
                        "• Stay physically active—light exercise boosts mood and lowers stress hormones.",
                        "• Set small, achievable goals instead of taking on too much at once.",
                        "• Consider reducing unnecessary tasks or obligations when possible.",
                        "• Talk to someone you trust if you feel emotionally overwhelmed.",
                        "• Limit caffeine and screen time before bed to improve sleep quality."
                    ]
                },

                2: {
                    "label": "High Stress",
                    "message": (
                        "Your results indicate a high level of stress. You may be feeling emotionally "
                        "overwhelmed, mentally drained, or physically exhausted. Prolonged stress can "
                        "impact your health, mood, and daily functioning, so it’s important to take steps "
                        "toward managing and reducing what you're experiencing."
                    ),
                    "suggestions": [
                        "• Prioritize rest—your body and mind need recovery time; try to get 7–9 hours of sleep.",
                        "• Practice grounding techniques (like the 5-4-3-2-1 sensory method) during moments of overwhelm.",
                        "• Reduce or reorganize your workload if possible—delegate tasks or postpone non-urgent responsibilities.",
                        "• Seek emotional support from someone you trust or consider journaling to process your feelings.",
                        "• Engage in calming activities such as meditation, nature walks, or relaxing music.",
                        "• Avoid skipping meals—balanced nutrition helps regulate mood and energy levels.",
                        "• If stress feels constant or unmanageable, consider speaking with a mental health professional."
                    ]
                }
            }

            result = stress_responses.get(label)

            # Save CSV
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            with open("stress_history.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([session.get('user'), result["label"], result["message"], timestamp])

            return render_template(
                "result.html",
                stress_level=result["label"],
                message=result["message"],
                suggestions=result["suggestions"]
            )

        except Exception as e:
            return render_template("error.html", error_message=str(e))

    return render_template("questionnaire.html")


# ============================================================
# PHQ-9 SCREENING
# ============================================================

@app.route('/phq9', methods=['GET', 'POST'])
def phq9():
    questions = [
        "Little interest or pleasure in doing things?",
        "Feeling down, depressed, or hopeless?",
        "Trouble falling/staying asleep, or sleeping too much?",
        "Feeling tired or having little energy?",
        "Poor appetite or overeating?",
        "Feeling bad about yourself?",
        "Trouble concentrating on things?",
        "Moving/speaking slowly or being fidgety?",
        "Thoughts of self-harm?"
    ]

    # MUST BE DEFINED HERE
    choices = [
        "0 - Not at all",
        "1 - Several days",
        "2 - More than half the days",
        "3 - Nearly every day"
    ]

    if request.method == 'POST':
        try:
            scores = [int(request.form.get(f'q{i}', None)) for i in range(1, 10)]

            # CHECK if any are missing
            if None in scores:
                return render_template(
                    "error.html",
                    error_message="Fill in all questions."
                )

            total_score = sum(scores)

            # Severity Levels
            if total_score <= 4:
                level = "Minimal"
            elif total_score <= 9:
                level = "Mild"
            elif total_score <= 14:
                level = "Moderate"
            elif total_score <= 19:
                level = "Moderately Severe"
            else:
                level = "Severe"

            # DESCRIPTION + SUGGESTIONS
            recommendations = {
                "Minimal": {
                    "message": (
                        "Your symptoms appear minimal, and there are currently no strong signs of depression. "
                        "This is a good time to continue taking care of your emotional well-being and maintaining "
                        "healthy habits that support your mood. Even with minimal symptoms, regular self-care helps "
                        "prevent future stress or emotional difficulty."
                    ),
                    "suggestions": [
                        "• Maintain a regular sleep schedule and aim for 7–9 hours of rest.",
                        "• Engage in enjoyable hobbies or activities that make you feel fulfilled.",
                        "• Stay socially connected—spend time with friends, family, or people who uplift you.",
                        "• Practice gratitude journaling or short daily mindfulness routines.",
                        "• Maintain a balanced diet and hydrate regularly.",
                        "• Stay physically active with walks, stretching, or light exercise.",
                        "• Continue monitoring your mood weekly to ensure you remain emotionally balanced."
                    ]
                },

                "Mild": {
                    "message": (
                        "Your responses suggest mild depressive symptoms. These feelings may come and go and are often related "
                        "to stress, life changes, or emotional strain. Paying attention to early signs can help you manage your well-being "
                        "before symptoms intensify. Small adjustments to your routine may significantly improve mood and energy."
                    ),
                    "suggestions": [
                        "• Try light physical activity such as walking, yoga, or stretching to elevate your mood.",
                        "• Practice relaxation techniques such as deep breathing or guided meditation.",
                        "• Talk to someone you trust about what you’re experiencing—connection can be very grounding.",
                        "• Set aside time for enjoyable and calming activities like reading, music, or hobbies.",
                        "• Reduce screen time before bed and aim for consistent sleep habits.",
                        "• Avoid isolating yourself; try engaging in small social interactions.",
                        "• If symptoms persist for more than two weeks, consider a mental health consultation."
                    ]
                },

                "Moderate": {
                    "message": (
                        "Your results indicate moderate depressive symptoms. You may be experiencing emotional fatigue, reduced motivation, "
                        "changes in sleep or appetite, or difficulty managing daily responsibilities. These symptoms are manageable, but they "
                        "should not be ignored. With support and healthy coping strategies, you can begin to feel more stable and energized."
                    ),
                    "suggestions": [
                        "• Break larger tasks into smaller, manageable steps to reduce overwhelm.",
                        "• Increase exposure to natural sunlight or spend time outdoors regularly.",
                        "• Set gentle daily goals—celebrate even small accomplishments.",
                        "• Consider speaking with a counselor, therapist, or mental health professional.",
                        "• Limit alcohol, nicotine, or caffeine, which can worsen mood symptoms.",
                        "• Stay connected with supportive people and avoid emotional isolation.",
                        "• Journaling your thoughts may help you process emotions more clearly.",
                        "• Practice grounding techniques to reduce emotional distress (e.g., the 5-4-3-2-1 method)."
                    ]
                },

                "Moderately Severe": {
                    "message": (
                        "Your symptoms suggest moderately severe depression. These experiences may be affecting your daily functioning, "
                        "motivation, or emotional stability. You may be feeling consistently sad, withdrawn, or overwhelmed. It’s important to "
                        "seek structured support to help you manage and reduce these symptoms. You do not have to face this alone."
                    ),
                    "suggestions": [
                        "• Seek support from a therapist, psychologist, or mental health professional as soon as possible.",
                        "• Create a structured daily routine that includes sleep, meals, and gentle activities.",
                        "• Reach out to close friends or family—connection can provide emotional relief.",
                        "• Avoid making major life decisions during this period.",
                        "• Practice grounding or breathing exercises during moments of intense emotional discomfort.",
                        "• Ensure you are eating regular, balanced meals—nutrition affects mood.",
                        "• Reduce stressful or unnecessary responsibilities when possible.",
                        "• If symptoms worsen, do not hesitate to seek immediate professional support."
                    ]
                },

                "Severe": {
                    "message": (
                        "Your results indicate severe depressive symptoms, which may include persistent sadness, hopelessness, fatigue, "
                        "loss of interest, or thoughts of self-harm. This level of depression is serious and requires immediate support. "
                        "Please remember that help is available, and reaching out is a strong and courageous step. You do not have to manage "
                        "this alone—professional and emotional support can make a meaningful difference."
                    ),
                    "suggestions": [
                        "• Seek immediate help from a mental health professional, therapist, or psychiatrist.",
                        "• Talk to someone you trust right now—sharing how you feel is important.",
                        "• Avoid being alone during times of intense emotional distress.",
                        "• Reduce or pause high-stress tasks and give yourself permission to rest.",
                        "• Consider contacting your local mental health helpline for immediate support.",
                        "• Maintain a simple structured routine focused on sleep, hydration, and nourishment.",
                        "• Engage in calming activities like slow breathing, gentle movements, or soothing music.",
                        "• If you have thoughts of harming yourself, please contact emergency services or a crisis hotline immediately."
                    ]
                }
            }

            r = recommendations[level]

            # SAVE TO CSV
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

            with open("phq9_history.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([session.get("user"), total_score, level, timestamp])

            return render_template(
                "phq9_result.html",
                score=total_score,
                level=level,
                message=r["message"],
                suggestions=r["suggestions"]
            )

        except Exception as e:
            return render_template("error.html", error_message=str(e))

    # FIX: PASS choices ALWAYS
    return render_template("phq9.html", questions=questions, choices=choices)

# ============================================================
# ADMIN DASHBOARD
# ============================================================

@app.route('/admin')
def admin_dashboard():
    # Only admin can access
    if session.get("user") != "admin":
        return "❌ Unauthorized Access: Admins only", 403

    users_df = pd.read_csv("users.csv")
    stress_df = pd.read_csv("stress_history.csv", encoding="latin1")

    phq9_df = pd.read_csv("phq9_history.csv", encoding="ISO-8859-1", encoding_errors="ignore")


    total_users = len(users_df)
    total_stress = len(stress_df)
    total_phq9 = len(phq9_df)

    avg_stress = stress_df['stress_level'].value_counts().to_dict()

    phq9_counts = phq9_df['severity'].value_counts().to_dict()
    phq9_labels = list(phq9_counts.keys())
    phq9_values = list(phq9_counts.values())

    high_risk_users = (
        phq9_df[phq9_df["severity"].isin(["Moderately Severe", "Severe"])]["username"]
        .unique()
        .tolist()  # <-- THIS FIXES THE ERROR
    )

    return render_template(
        "admin_dashboard.html",
        total_users=total_users,
        total_stress=total_stress,
        total_phq9=total_phq9,
        avg_stress=avg_stress,
        high_risk_users=high_risk_users,
        phq9_labels=phq9_labels,
        phq9_counts=phq9_values
    )

# ============================================================
# USER PROFILE SUMMARY
# ============================================================

@app.route('/profile')
def profile():
    username = session.get("user")

    stress_df = pd.read_csv("stress_history.csv", encoding="ISO-8859-1", encoding_errors="ignore")
    phq9_df = pd.read_csv("phq9_history.csv", encoding="ISO-8859-1", encoding_errors="ignore")
    chat_df = pd.read_csv("chat_sentiment.csv", encoding="ISO-8859-1", encoding_errors="ignore")

    user_stress = stress_df[stress_df["username"] == username]
    user_phq9 = phq9_df[phq9_df["username"] == username]
    user_chat = chat_df[chat_df["username"] == username]

    return render_template(
        "profile.html",
        username=username,
        stress_history=user_stress.to_dict("records"),
        phq9_history=user_phq9.to_dict("records"),
        chat_summary=user_chat.to_dict("records")
    )

@app.route('/user_dashboard')
def user_dashboard():
    username = session.get("user")

    if not username:
        return redirect(url_for('login'))

    # Load CSVs safely
    stress_df = pd.read_csv("stress_history.csv", encoding="ISO-8859-1", encoding_errors="ignore")
    phq9_df = pd.read_csv("phq9_history.csv", encoding="ISO-8859-1", encoding_errors="ignore")
    chat_df = pd.read_csv("chat_sentiment.csv", encoding="ISO-8859-1", encoding_errors="ignore")

    # Filter with COPY() to avoid SettingWithCopyWarning
    user_stress = stress_df[stress_df["username"] == username].copy()
    user_phq9 = phq9_df[phq9_df["username"] == username].copy()
    user_chat = chat_df[chat_df["username"] == username].copy()

    # Convert timestamps safely
    user_stress["timestamp"] = pd.to_datetime(user_stress["timestamp"], errors='coerce')
    user_phq9["timestamp"] = pd.to_datetime(user_phq9["timestamp"], errors='coerce')
    user_chat["timestamp"] = pd.to_datetime(user_chat["timestamp"], errors='coerce')

    # Send to template
    return render_template(
        "user_dashboard.html",
        username=username,
        stress_data=user_stress.to_dict("records"),
        phq9_data=user_phq9.to_dict("records"),
        chat_data=user_chat.to_dict("records"),
        stress_json=user_stress.to_json(orient="records"),
        phq9_json=user_phq9.to_json(orient="records"),
        chat_json=user_chat.to_json(orient="records")
    )


# ============================================================
# RUN APP
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)
