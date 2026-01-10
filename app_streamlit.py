import streamlit as st
import requests
import json
import tempfile
import subprocess
import os
from datetime import datetime, timedelta
from pymongo import MongoClient
import gridfs
from bson import ObjectId
from dotenv import load_dotenv
import PyPDF2
from mailjet_rest import Client
from fpdf import FPDF

load_dotenv()


WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "base")
FFMPEG_PATH = os.getenv("FFMPEG_PATH")
if FFMPEG_PATH:
    os.environ["PATH"] += os.pathsep + os.path.dirname(FFMPEG_PATH)

API_URL = os.getenv("API_URL", "https://dreaddit-api.yellowmeadow-d89419e0.francecentral.azurecontainerapps.io/predict")

# MongoDB
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "stress_db")

# Mailjet
MAILJET_API_KEY = os.getenv("MAILJET_API_KEY")
MAILJET_SECRET_KEY = os.getenv("MAILJET_SECRET_KEY")
SENDER_EMAIL = os.getenv("ALERT_SENDER_EMAIL")
MANAGER_EMAIL = os.getenv("ALERT_MANAGER_EMAIL", "zaineb.benfadhl@polytechnicien.tn")

# MONGODB + GRIDFS

@st.cache_resource
def get_mongo():
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        db = client[MONGO_DB]
        fs = gridfs.GridFS(db)
        client.admin.command('ping') 
        return db, fs, True
    except Exception as e:
        st.warning(f"MongoDB non disponible : {e}")
        return None, None, False

db, fs, mongo_available = get_mongo()
samples_col = db["voice_samples"] if mongo_available else None


# WHISPER MODEL

@st.cache_resource
def load_whisper_model():
    try:
        from faster_whisper import WhisperModel
        return WhisperModel(WHISPER_MODEL_NAME, device="cpu", compute_type="int8"), True
    except Exception as e:
        st.warning(f"Whisper non disponible : {e}")
        return None, False

whisper_model, whisper_available = load_whisper_model()


# PDF EXTRACTION

def extract_text_from_pdf(uploaded_pdf) -> str:
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Erreur extraction PDF : {e}")
        return ""


# AUDIO TRANSCRIPTION

def convert_to_wav_16k_mono(input_path: str, output_path: str):
    cmd = ["ffmpeg", "-y", "-i", input_path, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", output_path]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stderr

def transcribe_audio_file(uploaded_audio) -> str:
    if uploaded_audio is None or not whisper_available:
        return ""
    original_name = getattr(uploaded_audio, "name", "") or "recording.webm"
    _, ext = os.path.splitext(original_name)
    ext = (ext or ".webm").lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_in:
        audio_bytes = uploaded_audio.getvalue() if hasattr(uploaded_audio, "getvalue") else uploaded_audio.read()
        tmp_in.write(audio_bytes)
        input_path = tmp_in.name
    output_path = input_path + ".wav"
    try:
        code, err = convert_to_wav_16k_mono(input_path, output_path)
        if code != 0:
            st.error("FFmpeg: conversion échouée")
            st.code(err or "No FFmpeg stderr")
            return ""
        segments, info = whisper_model.transcribe(output_path, language="en")
        text = " ".join(seg.text for seg in segments).strip()
        return text
    except Exception as e:
        st.error("Transcription échouée")
        st.code(str(e))
        return ""
    finally:
        for p in (input_path, output_path):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass


# SAVE SAMPLE MONGODB

def save_voice_sample(uploaded_audio, transcript: str, prediction: int, probability: float, text_source: str, pdf_filename: str = None, risk_level: str = None) -> str | None:
    if not mongo_available:
        return None
    audio_file_id = None
    filename = None
    mime_type = None
    if uploaded_audio is not None:
        filename = getattr(uploaded_audio, "name", "") or "recording.webm"
        mime_type = getattr(uploaded_audio, "type", "") or "audio/webm"
        audio_bytes = uploaded_audio.getvalue() if hasattr(uploaded_audio, "getvalue") else uploaded_audio.read()
        audio_file_id = fs.put(audio_bytes, filename=filename, contentType=mime_type, createdAt=datetime.utcnow())
    doc = {
        "createdAt": datetime.utcnow(),
        "source": "streamlit",
        "text_source": text_source,
        "audio_file_id": audio_file_id,
        "filename": filename,
        "mime_type": mime_type,
        "pdf_filename": pdf_filename,
        "transcript": transcript,
        "pred_label": prediction,
        "pred_prob_stress": probability,
        "risk_level": risk_level,
        "human_label": None,
        "used_in_training": False,
    }
    res = samples_col.insert_one(doc)
    return str(res.inserted_id)


# AGENT ALERT

def send_alert_email(user_text: str, predicted_label: str, probability: float) -> bool:
    if not MAILJET_API_KEY or not MAILJET_SECRET_KEY or not SENDER_EMAIL or not MANAGER_EMAIL:
        st.warning(" Mailjet ou emails non configurés")
        return False
    try:
        mailjet = Client(auth=(MAILJET_API_KEY, MAILJET_SECRET_KEY), version="v3.1")
        data = {
            "Messages": [ {
                "From": {"Email": SENDER_EMAIL, "Name": "Stress Monitor IA"},
                "To": [{"Email": MANAGER_EMAIL, "Name": "Manager"}],
                "Subject": " Alerte stress détecté par IA",
                "TextPart": f"Classe prédite : {predicted_label}\nProbabilité stress : {probability:.2%}\nTexte : {user_text}",
                "HTMLPart": f"<p>Classe prédite : {predicted_label}</p><p>Probabilité stress : {probability:.2%}</p><blockquote>{user_text}</blockquote>"
            }]
        }
        result = mailjet.send.create(data=data)
        return result.status_code == 200
    except Exception as e:
        st.error(f"Erreur envoi mail : {e}")
        return False

# TEXT SANITIZATION FOR PDF

def sanitize_text_for_pdf(text: str) -> str:
    """Remove or replace characters that can't be encoded in latin-1"""
    if not text:
        return ""
    #  Unicode characters with ASCII equivalents
    replacements = {
        '\u2019': "'", 
        '\u2018': "'", 
        '\u201c': '"', 
        '\u201d': '"',  
        '\u2013': '-',  
        '\u2014': '-', 
        '\u2026': '...', 
        '\u00a0': ' ',  
        '\u2022': '*', 
        '\u2122': 'TM', 
        '\u00ae': '(R)',
        '\u00a9': '(C)',
    }
    for unicode_char, ascii_char in replacements.items():
        text = text.replace(unicode_char, ascii_char)
    
    try:
        text.encode('latin-1')
        return text
    except UnicodeEncodeError:
        return text.encode('latin-1', errors='replace').decode('latin-1')

# GENERATION RAPPORT PDF

def generate_daily_report():
    if not mongo_available: 
        return None

    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    tomorrow = today + timedelta(days=1)

    samples = list(samples_col.find({"createdAt": {"$gte": today, "$lt": tomorrow}}))
    if not samples: 
        return None

    # Calcul des statistiques
    total_analyses = len(samples)
    cas_stress = sum(1 for s in samples if s.get('pred_label') == 1)
    taux_stress = (cas_stress / total_analyses * 100) if total_analyses > 0 else 0
    
    # Probabilité moyenne pour les cas de stress
    stress_samples = [s for s in samples if s.get('pred_label') == 1]
    proba_moyenne = (sum(s.get('pred_prob_stress', 0) for s in stress_samples) / len(stress_samples) * 100) if stress_samples else 0
    
    # Répartition des risques basée sur la probabilité
    risk_counts = {'Low': 0, 'Medium': 0, 'High': 0}
    for s in samples:
        risk = s.get('risk_level')
        if not risk or risk == 'N/A':
            prob = s.get('pred_prob_stress', 0)
            if prob >= 0.8:
                risk = 'High'
            elif prob >= 0.5:
                risk = 'Medium'
            else:
                risk = 'Low'
        if risk in risk_counts:
            risk_counts[risk] += 1

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 12, "Rapport Journalier - Stress au Travail", ln=True, align="C")
    pdf.ln(5)
    pdf.set_font("Arial", "", 12)
    date_str = today.strftime("%d/%m/%Y")
    pdf.cell(0, 8, f"Date : {date_str}", ln=True)
    pdf.ln(3)
    
    # Statistiques principales
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, f"Total d'analyses : {total_analyses}", ln=True)
    pdf.cell(0, 8, f"Cas stress detectes : {cas_stress}", ln=True)
    pdf.cell(0, 8, f"Taux de stress : {taux_stress:.2f}%", ln=True)
    pdf.cell(0, 8, f"Probabilite moyenne : {proba_moyenne:.2f}%", ln=True)
    pdf.ln(5)
    
    # Répartition des risques
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Repartition des risques :", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 7, f"- Low : {risk_counts['Low']}", ln=True)
    pdf.cell(0, 7, f"- Medium : {risk_counts['Medium']}", ln=True)
    pdf.cell(0, 7, f"- High : {risk_counts['High']}", ln=True)
    pdf.ln(8)
    
    # Détails des échantillons (si stress détecté)
    if stress_samples:
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Details des cas de stress :", ln=True)
        pdf.ln(3)
        
        for i, s in enumerate(stress_samples, 1):
            pdf.set_font("Arial", "B", 11)
            text_source = sanitize_text_for_pdf(s.get('text_source', ''))
            pdf_filename = sanitize_text_for_pdf(s.get('pdf_filename', 'N/A'))
            prob_stress = s.get('pred_prob_stress', 0)
            risk = s.get('risk_level')
            if not risk or risk == 'N/A':
                if prob_stress >= 0.8:
                    risk = 'High'
                elif prob_stress >= 0.5:
                    risk = 'Medium'
                else:
                    risk = 'Low'
            
            header_text = f"Cas #{i} - Source: {text_source} | Risque: {risk} | Prob: {prob_stress:.2%}"
            pdf.multi_cell(0, 7, sanitize_text_for_pdf(header_text))
            
            if pdf_filename != 'N/A':
                pdf.set_font("Arial", "I", 10)
                pdf.cell(0, 6, f"PDF: {pdf_filename}", ln=True)
            
            pdf.set_font("Arial", "", 10)
            text = s.get("transcript", "")
            text = sanitize_text_for_pdf(text)
            display_text = text[:300] + ("..." if len(text) > 300 else "")
            pdf.multi_cell(0, 5, display_text)
            pdf.ln(4)

    report_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(report_file.name)
    return report_file.name
# STREAMLIT UI

st.set_page_config(page_title="Stress Detector", page_icon="🧠", layout="centered")
st.title("Détection de stress (Texte + Voix + PDF)")

st.sidebar.title("🤖 Agent IA")
enable_agent = st.sidebar.checkbox("Activer alerte manager", value=True)
threshold = st.sidebar.slider("Seuil prob stress pour alerte", 0.5,0.99,0.85,0.01)

mode = st.radio("Mode :", ["Texte","Voix (enregistrement)","Rapport PDF","Texte + Voix"], horizontal=True)

user_text, audio_file, pdf_file, pdf_text = "", None, None, ""

if mode in ["Texte","Texte + Voix"]:
    user_text = st.text_area("Votre texte (anglais)", height=180)
if mode in ["Voix (enregistrement)","Texte + Voix"]:
    st.write("Enregistrez un vocal :")
    if whisper_available:
        audio_file = st.audio_input("Enregistrement vocal")
        if audio_file: st.audio(audio_file)
    else: st.warning("Transcription vocale non disponible")
if mode=="Rapport PDF":
    pdf_file = st.file_uploader("Sélectionnez un PDF", type=["pdf"])
    if pdf_file: pdf_text = extract_text_from_pdf(pdf_file)


# BOUTONS

analyze_btn = st.button("Analyser", type="primary")
generate_report_btn = st.button("Générer rapport manager PDF")


# PIPELINE ANALYSE
def run_classification_pipeline(text_source:str, final_text:str, audio_uploaded, pdf_filename:str=None):
    if not final_text.strip(): st.warning("Texte vide"); return
    with st.spinner("Analyse en cours..."):
        try:
            response = requests.post(API_URL, headers={"Content-Type":"application/json"}, data=json.dumps({"text":final_text}), timeout=15)
            if response.status_code==200:
                result = response.json()
                st.subheader("Résultat du modèle")
                st.write(f"Source : {text_source}")
                if pdf_filename: st.write(f"PDF : {pdf_filename}")
                prediction = result.get("prediction")
                label_text = result.get("label")
                probability = result.get("probability")
                risk_level = result.get("risk_level")
                if prediction==1: st.error("😰 Stress détecté")
                else: st.success("😌 Pas de stress")
                if probability is not None: st.write(f"Probabilité de stress : {probability:.2%}")
                if risk_level: st.write(f"Niveau de risque : {risk_level}")

                # Save MongoDB
                if mongo_available:
                    saved_id = save_voice_sample(audio_uploaded, final_text, prediction, probability, text_source, pdf_filename, risk_level)
                    if saved_id: st.success(f"Sauvegardé dans MongoDB (id={saved_id})")

                # Agent IA
                if enable_agent and prediction==1 and probability>=threshold:
                    sent = send_alert_email(final_text, label_text, probability)
                    if sent: st.success(f"Email envoyé au manager : {MANAGER_EMAIL}")
                    else: st.error("Échec envoi email")
            else:
                st.error(f"Erreur API {response.status_code}")
        except Exception as e:
            st.error(f"Erreur API : {e}")

# ANALYSE
if analyze_btn:
    transcript = ""
    if mode in ["Voix (enregistrement)","Texte + Voix"] and audio_file:
        transcript = transcribe_audio_file(audio_file)
        if transcript: st.subheader("Transcription :"); st.write(transcript)
        else: st.error("Échec transcription audio")

    final_text = " ".join([user_text.strip(), transcript.strip()]).strip() if mode=="Texte + Voix" else user_text.strip() or transcript or pdf_text
    if final_text: run_classification_pipeline(mode, final_text, audio_file, pdf_file.name if pdf_file else None)


# GENERATION RAPPORT PDF

if generate_report_btn:
    report_path = generate_daily_report()
    if report_path:
        with open(report_path, "rb") as f:
            st.download_button("Télécharger rapport PDF", f, file_name="rapport_manager.pdf", mime="application/pdf")
    else:
        st.warning("Aucun échantillon disponible pour générer le rapport aujourd'hui.")


# VALIDATION HUMAINE

if mongo_available:
    st.write("---")
    st.subheader("Validation humaine")
    st.caption("Validez les prédictions pour entraîner le modèle")
    unlabeled = list(samples_col.find({"human_label": None}).sort("createdAt", -1).limit(10))
    if not unlabeled: st.info("Aucun sample en attente")
    else:
        for doc in unlabeled:
            sample_id = str(doc["_id"])
            pred_label = doc.get('pred_label', '?')
            stress_prob = doc.get('pred_prob_stress', 0)
            text_source = doc.get('text_source', 'Unknown')
            pdf_name = doc.get('pdf_filename', '')

            st.write(f"Sample `{sample_id}` — Source: `{text_source}` — Prédiction: `{pred_label}` | Probabilité stress: `{stress_prob:.2%}`")
            if pdf_name: st.caption(f"PDF: {pdf_name}")

            transcript_text = doc.get("transcript", "")
            display_text = transcript_text[:500] + ("..." if len(transcript_text) > 500 else "")
            st.text(display_text)

            col1, col2, col3 = st.columns([1,1,2])
            with col1:
                if st.button("NON stress (0)", key=f"ok0_{sample_id}"):
                    samples_col.update_one({"_id": ObjectId(sample_id)}, {"$set": {"human_label": 0}})
                    st.success("Label humain = 0 enregistré.")
                    st.rerun()

            with col2:
                if st.button("STRESS (1)", key=f"ok1_{sample_id}"):
                    samples_col.update_one({"_id": ObjectId(sample_id)}, {"$set": {"human_label": 1}})
                    st.success("Label humain = 1 enregistré.")
                    st.rerun()

            with col3:
                st.caption("Après validation, lancez `trainer_worker.py` pour ré-entraîner le modèle")
            st.write("---")