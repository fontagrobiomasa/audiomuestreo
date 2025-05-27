import streamlit as st
import os
import tempfile
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

# Cargar archivo de audio
audio_file = st.file_uploader("Subí un archivo de audio (wav, mp3, m4a, etc.)", type=["wav", "mp3", "m4a", "ogg"])

# Selección de idioma
lang = st.selectbox("Seleccioná el idioma del audio:", ["es", "en", "pt", "fr", "de"])

# Botón para iniciar transcripción
if audio_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[-1]) as tmp_file:
        tmp_file.write(audio_file.read())
        tmp_audio_path = tmp_file.name

    st.audio(audio_file, format="audio/" + os.path.splitext(audio_file.name)[-1].replace('.', ''))

    if st.button("Transcribir"):
        with st.spinner("Transcribiendo..."):
            try:
                # Cargar el modelo faster-whisper (usar tiny o base para mayor velocidad)
                model = WhisperModel("base", compute_type="int8")

                segments, info = model.transcribe(tmp_audio_path, language=lang)

                st.success("Transcripción completa.")
                st.markdown("### Texto transcripto:")
                full_text = ""
                for segment in segments:
                    full_text += segment.text + " "
                st.text_area("Texto", value=full_text.strip(), height=300)

            except Exception as e:
                st.error(f"Ocurrió un error al transcribir el audio: {e}")
            finally:
                os.remove(tmp_audio_path)
