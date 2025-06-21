import os
import time  # <--- 1. Importamos la librería time
import whisper
from moviepy.editor import VideoFileClip

def transcribir_video_local(ruta_video: str): # <--- 2. Modificamos lo que devuelve la función
    """
    Extrae el audio de un video, lo transcribe a texto usando Whisper
    y devuelve la transcripción junto con los tiempos de cada fase.
    """
    print(f"Cargando video desde: {ruta_video}")
    if not os.path.exists(ruta_video):
        return "Error: El archivo de video no fue encontrado.", {}

    tiempos = {}
    ruta_audio_temporal = "temp_audio.mp3"

    try:
        # --- Fase 1: Extracción de Audio ---
        t_inicio_extraccion = time.time()
        video = VideoFileClip(ruta_video)
        print("Extrayendo audio del video...")
        video.audio.write_audiofile(ruta_audio_temporal, codec='mp3', logger=None) # logger=None para una salida más limpia
        video.close()
        t_fin_extraccion = time.time()
        tiempos['extraccion'] = t_fin_extraccion - t_inicio_extraccion
        print("Audio extraído con éxito.")

        # --- Fase 2: Carga del Modelo ---
        t_inicio_carga = time.time()
        print("Cargando modelo de Whisper...")
        model = whisper.load_model("base") 
        t_fin_carga = time.time()
        tiempos['carga_modelo'] = t_fin_carga - t_inicio_carga
        
        # --- Fase 3: Transcripción ---
        t_inicio_transcripcion = time.time()
        print("Iniciando transcripción... Este proceso puede tardar varios minutos.")
        resultado = model.transcribe(ruta_audio_temporal, language='en')
        transcripcion = resultado['text']
        t_fin_transcripcion = time.time()
        tiempos['transcripcion'] = t_fin_transcripcion - t_inicio_transcripcion
        print("Transcripción completada.")

    except Exception as e:
        return f"Ocurrió un error: {e}", {}
        
    finally:
        if os.path.exists(ruta_audio_temporal):
            os.remove(ruta_audio_temporal)
            print("Archivo de audio temporal eliminado.")

    return transcripcion, tiempos # <--- 3. Devolvemos el texto y el diccionario de tiempos

# --- INICIO DEL SCRIPT ---
if __name__ == "__main__":
    ruta_del_video = "mi_video.mp4" 

    # 4. Capturamos los dos valores que devuelve la función
    texto_transcrito, tiempos_proceso = transcribir_video_local(ruta_del_video)

    if texto_transcrito and not texto_transcrito.startswith("Error:"):
        print("\n--- TRANSCRIPCIÓN ---")
        print(texto_transcrito)
        
        nombre_archivo_salida = "transcripcion.txt"
        with open(nombre_archivo_salida, "w", encoding="utf-8") as f:
            f.write(texto_transcrito)
        print(f"\nTranscripción guardada en el archivo: {nombre_archivo_salida}")

        # 5. Mostramos el resumen de tiempos
        if tiempos_proceso:
            tiempo_total = sum(tiempos_proceso.values())
            print("\n--- TIEMPOS DE PROCESAMIENTO ---")
            print(f"Extracción de audio: {tiempos_proceso.get('extraccion', 0):.2f} segundos.")
            print(f"Carga del modelo IA: {tiempos_proceso.get('carga_modelo', 0):.2f} segundos.")
            print(f"Transcripción del audio: {tiempos_proceso.get('transcripcion', 0):.2f} segundos.")
            print("---------------------------------")
            print(f"Tiempo Total: {tiempo_total:.2f} segundos.")

    else:
        print(texto_transcrito)