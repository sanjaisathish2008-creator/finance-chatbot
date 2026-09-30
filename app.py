import os
from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv
from google import genai
from google.genai import types
from chatbot_config import SYSTEM_PROMPT

load_dotenv()
app = Flask(__name__)
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
PORT = int(os.getenv("PORT", "5000"))
client = genai.Client(api_key=API_KEY) if API_KEY else None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    if not API_KEY or client is None:
        return jsonify(error="Gemini API key is not configured. Please add it to the .env file."), 500
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()
    if not message:
        return jsonify(error="Please enter a message."), 400
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=message,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.4,
            ),
        )
        answer = (response.text or "").strip()
        if not answer:
            return jsonify(error="The chatbot returned an empty response."), 502
        return jsonify(response=answer)
    except Exception:
        return jsonify(error="Sorry, I couldn't process your request right now. Please try again."), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
