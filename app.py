from fastapi import FastAPI, UploadFile, File, HTTPException
from model.model import AudioProcessor
import torchaudio

app = FastAPI()
audio_processor = AudioProcessor()

@app.post("/predict/")
async def predict_audio(file: UploadFile = File(...)):
    try:
        # Leer archivo de audio
        waveform, sample_rate = torchaudio.load(file.file)
        # Preprocesar audio
        signal = audio_processor.preprocess_audio(waveform, sample_rate)
        # Predecir
        results = audio_processor.predict(signal, 16000)
        return {"age": results["age"], "gender_probs": results["gender_probs"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")
