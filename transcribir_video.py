import os
import whisper
from moviepy.editor import VideoFileClip

def transcribir_video_local(ruta_video: str) -> str:
    """
    Extrae el audio de un video, lo transcribe a texto usando Whisper
    y devuelve la transcripción.
    """
    print(f"Cargando video desde: {ruta_video}")
    if not os.path.exists(ruta_video):
        return "Error: El archivo de video no fue encontrado."

    try:
        # 1. Extraer audio del video
        video = VideoFileClip(ruta_video)
        ruta_audio_temporal = "temp_audio.mp3"
        print("Extrayendo audio del video...")
        video.audio.write_audiofile(ruta_audio_temporal, codec='mp3')
        video.close()
        print("Audio extraído con éxito.")

        # 2. Transcribir el audio con Whisper
        print("Cargando modelo de Whisper... (Esto puede tardar la primera vez)")
        # Puedes cambiar el modelo según tus necesidades de velocidad vs. precisión
        # Modelos disponibles: tiny, base, small, medium, large
        model = whisper.load_model("base") 
        
        print("Iniciando transcripción... Este proceso puede tardar varios minutos.")
        # Especificar el idioma ayuda a mejorar la precisión
        resultado = model.transcribe(ruta_audio_temporal, language='en')
        transcripcion = resultado['text']
        print("Transcripción completada.")

    except Exception as e:
        return f"Ocurrió un error: {e}"
        
    finally:
        # 3. Limpiar el archivo de audio temporal
        if os.path.exists(ruta_audio_temporal):
            os.remove(ruta_audio_temporal)
            print("Archivo de audio temporal eliminado.")

    return transcripcion

# --- INICIO DEL SCRIPT ---
if __name__ == "__main__":
    # IMPORTANTE: Cambia esta línea por la ruta de tu video
    ruta_del_video = "mi_video.mp4" 

    texto_transcrito = transcribir_video_local(ruta_del_video)

    if texto_transcrito and not texto_transcrito.startswith("Error:"):
        # Imprimir en consola
        print("\n--- TRANSCRIPCIÓN ---")
        print(texto_transcrito)
        
        # Guardar en un archivo de texto
        nombre_archivo_salida = "transcripcion.txt"
        with open(nombre_archivo_salida, "w", encoding="utf-8") as f:
            f.write(texto_transcrito)
        print(f"\nTranscripción guardada en el archivo: {nombre_archivo_salida}")
    else:
        # Imprimir errores
        print(texto_transcrito)