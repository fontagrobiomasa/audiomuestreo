# streamlit_app.py
import streamlit as st
import subprocess
import numpy as np
import whisper
import re
import tempfile
import os

st.title("Procesador de alturas desde audio")

uploaded_file = st.file_uploader("Subí un archivo de audio (.opus)", type=["opus"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".opus") as tmp_opus:
        tmp_opus.write(uploaded_file.read())
        opus_path = tmp_opus.name

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        wav_path = tmp_wav.name

    # Convertir opus a wav
    subprocess.run([
        "ffmpeg", "-y", "-i", opus_path,
        "-f", "wav", "-acodec", "pcm_s16le",
        "-ac", "1", "-ar", "16000", wav_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Transcribir
    st.write("Transcribiendo el audio con Whisper...")
    model = whisper.load_model("base")
    result = model.transcribe(wav_path)
    texto = result["text"]

    # Extraer alturas
    alturas = re.findall(r'\d+(?:[.,]\d+)?', texto)
    alturas = [float(a.replace(",", ".")) for a in alturas]
    alturas_array = np.array(alturas)

    promedio = np.mean(alturas_array)
    desvio = np.std(alturas_array)
    n = len(alturas_array)

    # Mostrar resultados
    st.subheader("Texto transcripto")
    st.write(texto)

    st.subheader("Resultados")
    st.write(f"Cantidad de alturas detectadas: {n}")
    st.write(f"Promedio: {promedio:.2f}")
    st.write(f"Desvío estándar: {desvio:.2f}")

    # Limpiar archivos temporales
    os.remove(opus_path)
    os.remove(wav_path)
