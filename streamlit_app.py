import streamlit as st
import os
import tempfile
import numpy as np
import re
from faster_whisper import WhisperModel
from pydub import AudioSegment

# --- Función para convertir a .wav ---
def convertir_a_wav(input_path):
    ext = os.path.splitext(input_path)[-1].lower()
    audio = AudioSegment.from_file(input_path, format=ext.replace(".", ""))
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
        audio.export(tmp_wav.name, format="wav")
        return tmp_wav.name

# --- Función para extraer alturas ---
def extraer_alturas(texto):
    alturas = re.findall(r'\d+(?:[.,]\d+)?', texto)
    return [float(a.replace(",", ".")) for a in alturas]

# --- App ---
st.title("AudioMuestreo - Transcripción y análisis de alturas")

st.markdown("""
Subí un archivo de audio con alturas (números) habladas. Se transcribirá automáticamente
y se calculará el promedio, desvío estándar y cantidad de valores detectados.
""")

# Archivos permitidos
audio_file = st.file_uploader(
    "Subí un archivo de audio (wav, mp3, m4a, ogg, opus, aac)",
    type=["wav", "mp3", "m4a", "ogg", "opus", "aac"]
)

# Selección de idioma
lang = st.selectbox("Idioma del audio:", ["es", "en", "pt", "fr", "de"])

# Procesar
if audio_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[-1]) as tmp_file:
        tmp_file.write(audio_file.read())
        tmp_audio_path = tmp_file.name

    st.audio(audio_file, format="audio/" + os.path.splitext(audio_file.name)[-1].replace('.', ''))

    if st.button("Transcribir y analizar"):
        with st.spinner("Procesando..."):

            try:
                # Convertir a .wav
                wav_path = convertir_a_wav(tmp_audio_path)

                # Cargar modelo
                model = WhisperModel("base", compute_type="int8")

                # Transcripción
                segments, info = model.transcribe(wav_path, language=lang)
                texto = " ".join([seg.text for seg in segments])

                # Extraer alturas
                alturas = extraer_alturas(texto)
                alturas_array = np.array(alturas)
                promedio = np.mean(alturas_array) if len(alturas) > 0 else 0
                desvio = np.std(alturas_array) if len(alturas) > 0 else 0
                n = len(alturas)

                # Mostrar resultados
                st.success("Transcripción completa.")
                st.markdown("### Texto transcripto:")
                st.text_area("Texto", value=texto.strip(), height=300)

                st.markdown("### Estadísticas de alturas detectadas:")
                st.write(f"- Cantidad de muestras (N): {n}")
                st.write(f"- Promedio: {promedio:.2f}")
                st.write(f"- Desvío estándar: {desvio:.2f}")

            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                os.remove(tmp_audio_path)
                if 'wav_path' in locals():
                    os.remove(wav_path)
