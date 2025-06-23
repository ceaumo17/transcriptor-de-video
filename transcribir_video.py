import os
import time
import whisper
from moviepy.editor import VideoFileClip

def format_timestamp(seconds: float) -> str:
    """Convierte segundos a un formato de tiempo HH:MM:SS."""
    assert seconds >= 0, "non-negative timestamp expected"
    total_seconds = round(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

def transcribir_archivo_local(ruta_archivo: str):
    print(f"▶️ Procesando archivo: {ruta_archivo}")
    if not os.path.exists(ruta_archivo):
        return None, {}

    EXTENSIONES_VIDEO = ['.mp4', '.mov', '.avi', '.mkv', '.webm']
    EXTENSIONES_AUDIO = ['.mp3', '.wav', '.m4a', '.flac']
    
    tiempos = {}
    nombre_base, extension = os.path.splitext(ruta_archivo)
    extension = extension.lower()
    
    ruta_audio_para_whisper = None
    es_video = False

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
            return None, {}
    elif extension in EXTENSIONES_AUDIO:
        print("🎵 Archivo de audio detectado. Se usará directamente.")
        ruta_audio_para_whisper = ruta_archivo
        tiempos['extraccion'] = 0
    else:
        return None, {}

    resultado_transcripcion = None
    try:
        t_inicio_carga = time.time()
        print("🧠 Cargando modelo de Whisper...")
        model = whisper.load_model("base")
        t_fin_carga = time.time()
        tiempos['carga_modelo'] = t_fin_carga - t_inicio_carga
        
        t_inicio_transcripcion = time.time()
        print("🎤 Iniciando transcripción...")
        resultado_transcripcion = model.transcribe(ruta_audio_para_whisper, language='en', verbose=True) 
        
        t_fin_transcripcion = time.time()
        tiempos['transcripcion'] = t_fin_transcripcion - t_inicio_transcripcion
        print("\n✅ Transcripción completada.")

    except Exception as e:
        print(f"❌ Ocurrió un error durante la transcripción: {e}")
        return None, {}
        
    finally:
        if es_video and os.path.exists("temp_audio.mp3"):
            os.remove("temp_audio.mp3")
            print("🗑️ Archivo de audio temporal eliminado.")

    return resultado_transcripcion, tiempos

# --- INICIO DE LA EJECUCIÓN DEL SCRIPT ---
if __name__ == "__main__":
    
    # --- IMPORTANTE: CAMBIA ESTA LÍNEA POR LA RUTA DE TU ARCHIVO ---
    ruta_del_archivo_a_transcribir = "02-HTC TV - Adam Nassor.mp4" 
    
    resultado_completo, tiempos_proceso = transcribir_archivo_local(ruta_del_archivo_a_transcribir)

    if resultado_completo and 'segments' in resultado_completo:
        print("\n--- 📝 TRANSCRIPCIÓN FINAL CON TIEMPOS ---")
        
        nombre_base_input = os.path.splitext(os.path.basename(ruta_del_archivo_a_transcribir))[0]
        nombre_archivo_salida = f"transcripción_{nombre_base_input}.txt"
        
        with open(nombre_archivo_salida, "w", encoding="utf-8") as f:
            for segmento in resultado_completo['segments']:
                tiempo_inicio = format_timestamp(segmento['start'])
                tiempo_fin = format_timestamp(segmento['end'])
                texto = segmento['text']
                linea = f"[{tiempo_inicio} --> {tiempo_fin}] {texto.strip()}"
                print(linea)
                f.write(linea + "\n")

        print(f"\n💾 Transcripción guardada en el archivo: {nombre_archivo_salida}")

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
        print("\nNo se pudo generar una transcripción válida.")