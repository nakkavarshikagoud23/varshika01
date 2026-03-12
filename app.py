# stylesense_app.py
import os
import json
import base64
import requests
from flask import Flask, render_template_string, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from datetime import datetime
import re
import time
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs('uploads', exist_ok=True)

# Configuration
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
HF_API_KEY = os.environ.get('HF_API_KEY', '')

# Load HTML template from file
_TEMPLATE_DIR = os.path.dirname(os.path.abspath(__file__)) 
with open(os.path.join(_TEMPLATE_DIR, 'templates', 'index.html'), 'r', encoding='utf-8') as f:
    HTML_TEMPLATE = f.read()


class FashionAIAnalyzer:
    MODELS = ['gemini-2.0-flash', 'gemini-2.0-flash-lite', 'gemini-1.5-flash']
    MAX_RETRIES = 3
    RETRY_DELAY = 30

    def __init__(self, api_key):
        self.api_key = api_key
        genai.configure(api_key=api_key)

    def _call_with_retry(self, prompt, is_multimodal=False):
        last_error = None
        for model_name in self.MODELS:
            model = genai.GenerativeModel(model_name)
            for attempt in range(self.MAX_RETRIES):
                try:
                    print(f"Trying model: {model_name} (attempt {attempt + 1}/{self.MAX_RETRIES})")
                    response = model.generate_content(prompt)
                    if response.text:
                        return response.text
                except Exception as e:
                    last_error = e
                    error_str = str(e)
                    if '429' in error_str or 'quota' in error_str.lower() or 'rate' in error_str.lower():
                        delay = self.RETRY_DELAY
                        retry_match = re.search(r'retry in (\d+)', error_str.lower())
                        if retry_match:
                            delay = int(retry_match.group(1)) + 2
                        print(f"Rate limited on {model_name}. Waiting {delay}s...")
                        time.sleep(delay)
                        continue
                    else:
                        print(f"Error with {model_name}: {e}")
                        break
        if last_error:
            raise last_error
        raise Exception("All models failed")

    def analyze_image(self, image_data):
        try:
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image_part = {"mime_type": "image/jpeg", "data": image_bytes}
            return self._call_with_retry([
                "Describe this fashion image in detail, focusing on clothing items, colors, style, and overall look.",
                image_part
            ], is_multimodal=True)
        except Exception as e:
            print(f"Image analysis error: {e}")
            return ""

    def generate_fashion_recommendations(self, context):
        image_desc = context.get('image_description', '')
        image_note = f"\nInspiration image description: {image_desc}" if image_desc else ""
        mood = context.get('mood', '')
        mood_note = f"\n- Mood/Feeling: {mood} (Analyze this mood using NLP principles — identify the emotional tone, energy level, and personality traits expressed, then map these to matching fashion aesthetics)" if mood else ""
        mood_json_key = '"mood_analysis": "A detailed NLP-based analysis of the client\'s mood and how it maps to fashion choices — explain the emotional tone detected and why specific styles match this mood",' if mood else ''

        prompt = f"""You are a professional fashion stylist, trend expert, and NLP specialist. A client has provided the following details:

- Gender: {context.get('gender', 'prefer-not-to-say')}
- Body type: {context.get('body_type', 'any')}
- Style preference: {context['style']}
- Occasion: {context['occasion'] or 'General / Everyday'}
- Color preferences: {context['colors'] or 'Open to suggestions'}
- Budget level: {context['budget']} (low=budget-friendly, medium=moderate, high=premium, luxury=luxury){mood_note}{image_note}
- Additional notes: {context['notes'] or 'None'}

Respond ONLY with a valid JSON object — no markdown, no extra text, no code fences — with exactly these keys:
{{
  {mood_json_key}
  "outfit_recommendations": "...",
  "styling_tips": "...",
  "color_analysis": "...",
  "trends": "...",
  "trend_tags": ["tag1", "tag2", "tag3", "tag4"],
  "shopping_suggestions": "...",
  "visual_prompt": "3-5 descriptive keywords for the outfit and setting. No sentences."
}}

Make each value a detailed 2-3 sentence paragraph. Tailor everything specifically to the client's gender and body type. {('For mood_analysis, use NLP principles to interpret the emotional sentiment, then explain how this mood translates to specific fashion choices.' if mood else '')}"""

        try:
            generated = self._call_with_retry(prompt)
            parsed = self._extract_json(generated)
            if parsed:
                return parsed
            print(f"Could not parse AI response: {generated[:300]}")
            return {"error": "AI returned an unexpected format. Please try again."}
        except Exception as e:
            error_str = str(e)
            print(f"Recommendation generation error: {e}")
            if '429' in error_str or 'quota' in error_str.lower():
                return {"error": "API quota exceeded. Please wait a minute and try again."}
            return {"error": f"Failed to generate recommendations: {error_str}"}

    def _extract_json(self, text):
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        try:
            return json.loads(text.strip())
        except Exception:
            pass
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
        return None


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/check-key', methods=['GET'])
def check_key():
    return jsonify({'has_key': bool(GEMINI_API_KEY)})


@app.route('/api/set-key', methods=['POST'])
def set_key():
    global GEMINI_API_KEY
    try:
        data = request.get_json()
        key = data.get('key', '').strip()
        if not key:
            return jsonify({'success': False, 'error': 'No key provided'})
        genai.configure(api_key=key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        model.generate_content("Say hi")
        GEMINI_API_KEY = key
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/check-hf-key', methods=['GET'])
def check_hf_key():
    return jsonify({'has_key': bool(HF_API_KEY)})


@app.route('/api/set-hf-key', methods=['POST'])
def set_hf_key():
    global HF_API_KEY
    try:
        data = request.get_json()
        key = data.get('hf_key', '').strip()
        if not key:
            return jsonify({'success': False, 'error': 'No key provided'})
        
        # Simple validation ping to a small model
        headers = {"Authorization": f"Bearer {key}"}
        response = requests.post(
            "https://router.huggingface.co/hf-inference/models/openai/clip-vit-base-patch32",
            headers=headers,
            json={"inputs": "test connection"}
        )
        # Even if it errors with model loading, a 401 means invalid key
        if response.status_code == 401:
             return jsonify({'success': False, 'error': 'Invalid Hugging Face API key'})
        
        HF_API_KEY = key
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/generate-image', methods=['POST'])
def generate_image():
    if not HF_API_KEY:
        return jsonify({'error': 'Please set your Hugging Face API key first.'}), 400

    try:
        data = request.get_json()
        prompt = data.get('prompt', '').strip()
        if not prompt:
            return jsonify({'error': 'No visual prompt provided'}), 400

        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        # Using a fast Stable Diffusion base model on HF Inference API
        payload = {"inputs": prompt}
        
        response = requests.post(
            "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-base-1.0",
            headers=headers,
            json=payload
        )

        if response.status_code != 200:
            return jsonify({'error': f"Image generation failed: {response.text}"}), 500

        # Encode bytes to base64
        image_base64 = base64.b64encode(response.content).decode('utf-8')
        return jsonify({'image': image_base64})

    except Exception as e:
        print(f"Error in generate_image: {e}")
        return jsonify({'error': 'Failed to generate visual representation'}), 500



@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    try:
        if not GEMINI_API_KEY:
            return jsonify({'error': 'Please set your Gemini API key first.'})

        analyzer = FashionAIAnalyzer(GEMINI_API_KEY)
        context = {
            'style': request.form.get('style', 'casual'),
            'occasion': request.form.get('occasion', ''),
            'colors': request.form.get('colors', ''),
            'budget': request.form.get('budget', 'medium'),
            'notes': request.form.get('notes', ''),
            'gender': request.form.get('gender', 'prefer-not-to-say'),
            'body_type': request.form.get('body_type', 'any'),
            'mood': request.form.get('mood', ''),
        }

        image_data = request.form.get('image', '')
        if image_data:
            image_description = analyzer.analyze_image(image_data)
            if image_description:
                context['image_description'] = image_description

        recommendations = analyzer.generate_fashion_recommendations(context)
        return jsonify(recommendations)
    except Exception as e:
        print(f"Error in get_recommendations: {e}")
        return jsonify({'error': 'Failed to generate recommendations'}), 500


@app.route('/api/trends', methods=['GET'])
def get_trends():
    if not GEMINI_API_KEY:
        return jsonify({'error': 'API key not configured'}), 400

    analyzer = FashionAIAnalyzer(GEMINI_API_KEY)
    prompt = """You are a fashion trend analyst. Provide current fashion trends for 2025-2026.
Respond ONLY with a valid JSON object — no markdown, no code fences — with these keys:
{
  "2025_trends": ["trend1", "trend2", "trend3", "trend4", "trend5", "trend6"],
  "seasonal": {"spring": "...", "summer": "...", "fall": "...", "winter": "..."}
}
Make each seasonal value a descriptive sentence. The 2025_trends list should have 6 trending movements."""

    try:
        generated = analyzer._call_with_retry(prompt)
        parsed = analyzer._extract_json(generated)
        if parsed:
            return jsonify(parsed)
        return jsonify({'error': 'AI returned unexpected format'}), 500
    except Exception as e:
        print(f"Trends error: {e}")
        return jsonify({'error': 'Failed to generate trends'}), 500


if __name__ == '__main__':
    print("=" * 50)
    print("StyleSense Fashion AI Server Starting...")
    print("=" * 50)
    print("Visit: http://localhost:5000")
    if GEMINI_API_KEY:
        print("✅ Gemini API key is configured")
    else:
        print("⚠️  No Gemini API key set.")
        
    if HF_API_KEY:
        print("✅ Hugging Face API key is configured")
    else:
        print("⚠️  No Hugging HF key set.")
    print("=" * 50)
    app.run(debug=True, port=5000)