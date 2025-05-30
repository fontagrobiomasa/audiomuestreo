import streamlit as st
import os
import tempfile
import re
import numpy as np
import pandas as pd
from faster_whisper import WhisperModel

# Título de la aplicación
st.title("AudioMuestreo - Transcripción y análisis de múltiples audios")

st.markdown(
    """
    Esta herramienta permite transcribir archivos de audio y analizar números detectados en el texto (por ejemplo, alturas).
    
    Podés subir múltiples archivos de audio y ver un resumen con las estadísticas calculadas.
    """
)

# Subir múltiples archivos
audio_files = st.file_uploader(
    "Subí uno o más archivos de audio (wav, mp3, m4a, ogg, opus, aac)",
    type=["wav", "mp3", "m4a", "ogg", "opus", "aac"],
    accept_multiple_files=True
)

# Selección de idioma
lang = st.selectbox("Seleccioná el idioma del audio:", ["es", "en", "pt", "fr", "de"])

# Función para extraer números (alturas) del texto
def extraer_alturas(texto):
    alturas = re.findall(r'\d+(?:[.,]\d+)?', texto)
    return [float(a.replace(",", ".")) for a in alturas]

if audio_files and st.button("Transcribir todos"):
    resultados = []
    with st.spinner("Procesando archivos..."):
        try:
            # Cargar modelo una sola vez
            model = WhisperModel("base", compute_type="int8")

            for audio_file in audio_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[-1]) as tmp_file:
                    tmp_file.write(audio_file.read())
                    tmp_audio_path = tmp_file.name

                try:
                    segments, _ = model.transcribe(tmp_audio_path, language=lang)
                    texto = " ".join([seg.text for seg in segments])
                    alturas = extraer_alturas(texto)

                    if alturas:
                        alturas_array = np.array(alturas)
                        promedio = np.mean(alturas_array)
                        desvio = np.std(alturas_array)
                        minimo = np.min(alturas_array)
                        maximo = np.max(alturas_array)
                        n = len(alturas_array)
                    else:
                        promedio = desvio = minimo = maximo = n = 0

                    resultados.append({
                        "Archivo": audio_file.name,
                        "N": n,
                        "Promedio": round(promedio, 2),
                        "Desvío estándar": round(desvio, 2),
                        "Mínimo": round(minimo, 2),
                        "Máximo": round(maximo, 2)
                    })

                except Exception as e:
                    resultados.append({
                        "Archivo": audio_file.name,
                        "N": "Error",
                        "Promedio": "-",
                        "Desvío estándar": f"{e}",
                        "Mínimo": "-",
                        "Máximo": "-"
                    })

                finally:
                    os.remove(tmp_audio_path)

        except Exception as e:
            st.error(f"Error general durante el procesamiento: {e}")
    
    # Mostrar tabla de resultados
    if resultados:
        df_resultados = pd.DataFrame(resultados)
        st.markdown("### Resultados por archivo")
        st.dataframe(df_resultados, use_container_width=True)
