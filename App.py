import streamlit as st
import speech_recognition as sr
from gtts import gTTS
import os
import tempfile

from parser import parse_question

st.title("🎓 DeKUT Computer Science and Information Technology Smart FAQ Assistant")

input_type = st.radio("Choose input method:", ["Text", "Voice"])

# Handle voice input
def get_voice_input():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙️ Listening... Please ask your question.")
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            st.success(f"🗣️ You said: {text}")
            return text
        except sr.UnknownValueError:
            st.error("Sorry, I could not understand your voice.")
        except sr.RequestError:
            st.error("Could not connect to the voice service.")
    return ""

if input_type == "Text":
    user_input = st.text_input("Type your question here:")
else:
    if st.button("Record"):
        user_input = get_voice_input()
    else:
        user_input = ""

# Run parser
if user_input:
    print("DEBUG: Received input:", user_input)
    result = parse_question(user_input)
    if result["parsed"]:
        st.success(f"🗂️ Category: {result['category']}")
        st.write(f"💬 Response: {result['response']}")

        # Convert response to speech
        tts = gTTS(result["response"])
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            st.audio(fp.name, format='audio/mp3')
    else:
        st.error("❌ Sorry, I couldn't understand your question.")
