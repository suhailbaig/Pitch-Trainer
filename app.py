
import streamlit as st
import whisper
import librosa
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import tempfile

st.title("🗣️ AI Pitch Trainer")
st.write("Upload your call audio and compare it with an ideal pitch.")

def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"]

def get_text_similarity(text1, text2):
    tfidf = TfidfVectorizer().fit_transform([text1, text2])
    return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)

def analyze_audio_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    rms = librosa.feature.rms(y=y).mean()
    return tempo, rms

agent_file = st.file_uploader("Upload Agent Call (.mp3 or .wav)", type=["mp3", "wav"])
ideal_file = st.file_uploader("Upload Ideal Call (.mp3 or .wav)", type=["mp3", "wav"])

if agent_file and ideal_file:
    with tempfile.NamedTemporaryFile(delete=False) as agent_temp, tempfile.NamedTemporaryFile(delete=False) as ideal_temp:
        agent_temp.write(agent_file.read())
        ideal_temp.write(ideal_file.read())

        agent_text = transcribe_audio(agent_temp.name)
        ideal_text = transcribe_audio(ideal_temp.name)
        similarity = get_text_similarity(ideal_text, agent_text)
        sentiment = analyze_sentiment(agent_text)
        tempo, rms = analyze_audio_features(agent_temp.name)

        score = (similarity * 50) + (sentiment['pos'] * 30) + (min(tempo, 150)/150 * 20)
        st.success("✅ Analysis Complete!")
        st.metric("Overall Pitch Score", f"{score:.2f}/100")
        st.write("📝 Feedback:")
        if similarity < 0.7:
            st.write("- Improve your adherence to the core script.")
        if sentiment['pos'] < 0.3:
            st.write("- Try sounding more positive and confident.")
        if tempo < 90:
            st.write("- Speak a little faster to maintain engagement.")
