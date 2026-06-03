import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime
from transformers import pipeline

# Set page config for a cleaner layout
st.set_page_config(page_title="Student Wellness Tracker", layout="wide", page_icon="🧠")

# --- STEP 1: Core AI Engine (Cached to prevent reloading on every click) ---
@st.cache_resource
def load_nlp_pipeline():
    # j-hartmann/emotion-english-distilroberta-base maps to 7 emotion vectors
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True
    )

class MoodAnalystAgent:
    def __init__(self):
        self.classifier = load_nlp_pipeline()

    def analyze_journal(self, text: str):
        raw_predictions = self.classifier(text)
        
        if isinstance(raw_predictions, list) and len(raw_predictions) > 0:
            predictions = raw_predictions[0] if isinstance(raw_predictions[0], list) else raw_predictions
        else:
            predictions = raw_predictions

        # Sort emotions by confidence score descending
        sorted_emotions = sorted(predictions, key=lambda x: x.get('score', 0), reverse=True)
        top_emotions = sorted_emotions[:3]
        
        # Format metrics (converting decimal to percentage 0-100)
        primary_emotions = {e['label']: round(e['score'] * 100, 2) for e in top_emotions if 'label' in e and 'score' in e}
        
        # Heuristic Burnout Verification
        burnout_keywords = ["exhausted", "burnt out", "cannot cope", "overwhelmed", "give up", "tired of exams"]
        has_keyword = any(keyword in text.lower() for keyword in burnout_keywords)
        high_stress = any(e['label'] in ['sadness', 'fear'] and e['score'] > 0.40 for e in sorted_emotions)
        
        burnout_flag = True if (has_keyword or high_stress) else False
        return {"primary_emotions": primary_emotions, "burnout_flag": burnout_flag, "full_predictions": sorted_emotions}

# --- STEP 2: Database Layer ---
class WellnessDBManager:
    def __init__(self, db_name="colab_wellness.db"):
        self.db_name = db_name
        with sqlite3.connect(self.db_name) as conn:
            conn.cursor().execute("""
                CREATE TABLE IF NOT EXISTS mood_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    journal_text TEXT,
                    primary_emotion TEXT,
                    confidence REAL,
                    burnout_flag INTEGER
                )
            """)
            conn.commit()

    def save_log(self, text: str, analysis_result: dict):
        emotions = analysis_result["primary_emotions"]
        primary_emotion = list(emotions.keys())[0] if emotions else "neutral"
        confidence = emotions[primary_emotion] if emotions else 0.0
        burnout_val = 1 if analysis_result["burnout_flag"] else 0
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with sqlite3.connect(self.db_name) as conn:
            conn.cursor().execute("""
                INSERT INTO mood_logs (timestamp, journal_text, primary_emotion, confidence, burnout_flag)
                VALUES (?, ?, ?, ?, ?)
            """, (timestamp, text, primary_emotion, confidence, burnout_val))
            conn.commit()

    def fetch_dashboard_data(self):
        with sqlite3.connect(self.db_name) as conn:
            df = pd.read_sql_query("SELECT id, timestamp, journal_text, primary_emotion, confidence, burnout_flag FROM mood_logs ORDER BY timestamp DESC", conn)
        return df

# --- STEP 3: Streamlit UI Implementation ---
st.title("🧠 Intelligent Student Mental Health & Mood Tracker")
st.markdown("A localized, privacy-first AI agent designed to track student well-being and flag burnout thresholds.")

# Initialize backend instances
agent = MoodAnalystAgent()
db = WellnessDBManager()

# Layout: Split screen into Input section and Analytics section
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Daily Journal Log")
    user_input = st.text_area("How are you feeling today? (Reflections, exam stress, thoughts...)", height=150, placeholder="Type here...")
    
    if st.button("Analyze & Log Entry", type="primary"):
        if user_input.strip() == "":
            st.warning("Journal entry cannot be empty!")
        else:
            with st.spinner("AI Agent is analyzing your emotional state..."):
                # Run NLP & Heuristic analysis
                analysis = agent.analyze_journal(user_input)
                # Save data locally
                db.save_log(user_input, analysis)
                
                st.success("Entry securely saved to local database!")
                
                # Render instant feedback cards
                primary_emo = list(analysis["primary_emotions"].keys())[0]
                confidence_score = list(analysis["primary_emotions"].values())[0]
                
                st.metric(label="Detected Primary Emotion", value=f"{primary_emo.title()}", delta=f"{confidence_score}% Confidence")
                
                if analysis["burnout_flag"]:
                    st.error("⚠️ HIGH RISK DETECTED: Burnout Safety Flag Activated. Please consider taking a break or reaching out to campus counseling centers.")
                else:
                    st.success("✅ Burnout Safety Flag: Normal / Healthy Trajectory")

with col2:
    st.subheader("📊 Emotional Distribution Insights")
    # Quick live check if input exists to dynamically display current text breakdown
    if user_input.strip() != "":
        current_analysis = agent.analyze_journal(user_input)
        # Parse all scores for a quick horizontal bar chart
        chart_data = pd.DataFrame(current_analysis["full_predictions"])
        chart_data['score'] = chart_data['score'] * 100
        chart_data = chart_data.sort_values(by="score", ascending=True)
        st.bar_chart(data=chart_data, x="label", y="score", horizontal=True, use_container_width=True)
    else:
        st.info("Write a journal entry on the left to see your real-time emotion vector visualization.")


st.subheader("📜 Historical Metrics & Analytics Dashboard")
# Always fetch and refresh data log view from SQLite
df_logs = db.fetch_dashboard_data()

if not df_logs.empty:
    # Stylizing the pandas dataframe output for Streamlit
    def highlight_burnout(val):
        color = 'background-color: #ffcccc; color: black;' if val == 1 else ''
        return color

    styled_df = df_logs.style.map(highlight_burnout, subset=['burnout_flag'])
    st.dataframe(styled_df, use_container_width=True)
else:
    st.write("No historical records found. Start journaling above to populate your timeline.")
