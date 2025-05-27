import streamlit as st
import os
import tempfile
import re
import numpy as np
from faster_whisper import WhisperModel

# Título de la aplicación
st.title("AudioMuestreo - Transcripción de audio a texto")

# Descripción
st.markdown(
    """
    Esta herramienta permite transcribir archivos de audio utilizando el modelo **faster-whisper**.
    
    Subí un archivo de audio y esperá unos segundos para ver la transcripción.
    """
)

# Cargar archivo de audio (agregado .opus y .aac)
audio_file = st.file_uploader(
    "Subí un archivo de audio (wav, mp3, m4a, ogg, opus, aac)",
    type=["wav", "mp3", "m4a", "ogg", "opus", "aac"]
)

# Selección de idioma
lang = st.selectbox("Seleccioná el idioma del audio:", ["es", "en", "pt", "fr", "de"])

def extraer_alturas(texto):
    """Extrae todos los números decimales del texto, usando punto o coma como separador."""
    alturas = re.findall(r'\d+(?:[.,]\d+)?', texto)
    return [float(a.replace(",", ".")) for a in alturas]

# Botón para iniciar transcripción
if audio_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[-1]) as tmp_file:
        tmp_file.write(audio_file.read())
        tmp_audio_path = tmp_file.name

    st.audio(audio_file, format="audio/" + os.path.splitext(audio_file.name)[-1].replace('.', ''))

    if st.button("Transcribir"):
        with st.spinner("Transcribiendo..."):
            try:
                # Cargar el modelo faster-whisper
                model = WhisperModel("base", compute_type="int8")
                segments, info = model.transcribe(tmp_audio_path, language=lang)

                st.success("Transcripción completa.")
                st.markdown("### Texto transcripto:")
                full_text = " ".join([seg.text for seg in segments])
                st.text_area("Texto", value=full_text.strip(), height=300)

                # Extraer alturas
                alturas = extraer_alturas(full_text)
                if alturas:
                    alturas_array = np.array(alturas)
                    promedio = np.mean(alturas_array)
                    desvio = np.std(alturas_array)
                    n = len(alturas_array)

                    st.markdown("### Estadísticas de alturas detectadas:")
                    st.write(f"- Cantidad de muestras (N): {n}")
                    st.write(f"- Promedio: {promedio:.2f}")
                    st.write(f"- Desvío estándar: {desvio:.2f}")
                else:
                    st.info("No se detectaron números en el texto transcripto.")

            except Exception as e:
                st.error(f"Ocurrió un error al transcribir el audio: {e}")
            finally:
                os.remove(tmp_audio_path)
