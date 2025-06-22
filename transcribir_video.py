import os
import time
import whisper
from moviepy.editor import VideoFileClip

def transcribir_archivo_local(ruta_archivo: str):
    """
    Identifica si el archivo es video o audio, lo procesa, lo transcribe
    y devuelve el resultado y los tiempos.
    """
    print(f"▶️ Procesando archivo: {ruta_archivo}")
    if not os.path.exists(ruta_archivo):
        return "Error: El archivo no fue encontrado.", {}

    # Definir extensiones conocidas
    EXTENSIONES_VIDEO = ['.mp4', '.mov', '.avi', '.mkv', '.webm']
    EXTENSIONES_AUDIO = ['.mp3', '.wav', '.m4a', '.flac'] # Incluimos MP3 y otros formatos comunes
    
    tiempos = {}
    nombre_base, extension = os.path.splitext(ruta_archivo)
    extension = extension.lower()
    
    ruta_audio_para_whisper = None
    es_video = False

    # --- Fase 1: Preparación del Audio ---
    if extension in EXTENSIONES_VIDEO:
        es_video = True
        print("📹 Archivo de video detectado.")
        ruta_audio_para_whisper = "temp_audio.mp3"
        
        t_inicio_extraccion = time.time()
        try:
            video = VideoFileClip(ruta_archivo)
            print("🎧 Extrayendo audio del video...")
            video.audio.write_audiofile(ruta_audio_para_whisper, codec='mp3', logger=None)
            video.close()
            t_fin_extraccion = time.time()
            tiempos['extraccion'] = t_fin_extraccion - t_inicio_extraccion
            print("✅ Audio extraído con éxito.")
        except Exception as e:
            return f"❌ Error al extraer el audio: {e}", {}

    elif extension in EXTENSIONES_AUDIO:
        print("🎵 Archivo de audio detectado. Se usará directamente.")
        ruta_audio_para_whisper = ruta_archivo
        tiempos['extraccion'] = 0 # No hay extracción, tiempo es 0
    else:
        return f"❌ Error: Tipo de archivo '{extension}' no soportado.", {}

    try:
        # --- Fase 2: Carga del Modelo ---
        t_inicio_carga = time.time()
        print("🧠 Cargando modelo de Whisper...")
        # Puedes cambiar 'base' por 'small' o 'medium' si tu máquina es potente
        model = whisper.load_model("base")
        t_fin_carga = time.time()
        tiempos['carga_modelo'] = t_fin_carga - t_inicio_carga
        
        # --- Fase 3: Transcripción ---
        t_inicio_transcripcion = time.time()
        print("🎤 Iniciando transcripción...")
        resultado = model.transcribe(ruta_audio_para_whisper, language='en', verbose=None) 
        transcripcion = resultado['text']
        t_fin_transcripcion = time.time()
        tiempos['transcripcion'] = t_fin_transcripcion - t_inicio_transcripcion
        print("\n✅ Transcripción completada.")

    except Exception as e:
        return f"❌ Ocurrió un error durante la transcripción: {e}", {}
        
    finally:
        # Solo eliminamos el archivo de audio si fue un video el que lo generó
        if es_video and os.path.exists("temp_audio.mp3"):
            os.remove("temp_audio.mp3")
            print("🗑️ Archivo de audio temporal eliminado.")

    return transcripcion, tiempos

# --- INICIO DE LA EJECUCIÓN DEL SCRIPT ---
if __name__ == "__main__":
    
    # --- IMPORTANTE: CAMBIA ESTA LÍNEA POR LA RUTA DE TU ARCHIVO ---
    # Puede ser un video o un archivo de audio MP3
    ruta_del_archivo_a_transcribir = "mi_video_o_audio.mp4" 
    
    # Llamamos a la función con el archivo
    texto_transcrito, tiempos_proceso = transcribir_archivo_local(ruta_del_archivo_a_transcribir)

    if texto_transcrito and not texto_transcrito.startswith("Error:"):
        print("\n--- 📝 TRANSCRIPCIÓN ---")
        print(texto_transcrito)
        
        # --- Lógica para el nombre de archivo dinámico ---
        nombre_base_input = os.path.splitext(os.path.basename(ruta_del_archivo_a_transcribir))[0]
        nombre_archivo_salida = f"transcripción_{nombre_base_input}.txt"
        
        with open(nombre_archivo_salida, "w", encoding="utf-8") as f:
            f.write(texto_transcrito)
        print(f"\n💾 Transcripción guardada en el archivo: {nombre_archivo_salida}")

        # Mostramos el resumen de tiempos
        if tiempos_proceso:
            tiempo_total = sum(tiempos_proceso.values())
            print("\n--- ⏱️ TIEMPOS DE PROCESAMIENTO ---")
            if tiempos_proceso.get('extraccion', 0) > 0:
                 print(f"Extracción de audio: {tiempos_proceso.get('extraccion', 0):.2f} segundos.")
            print(f"Carga del modelo IA: {tiempos_proceso.get('carga_modelo', 0):.2f} segundos.")
            print(f"Transcripción del audio: {tiempos_proceso.get('transcripcion', 0):.2f} segundos.")
            print("---------------------------------")
            print(f"Tiempo Total: {tiempo_total:.2f} segundos.")

    else:
        # Imprime el mensaje de error si algo falló
        print(f"\n{texto_transcrito}")