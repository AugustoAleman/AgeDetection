import requests

# Define la URL de la API
url = "http://127.0.0.1:8000/predict/"

# Ruta del archivo de audio que quieres probar
audio_file_path = "E:/Users/1158209/Desktop/Envs/tio_ricky/pruebas/data/octavio2.wav"

# Abre el archivo de audio y envíalo como una solicitud POST
with open(audio_file_path, "rb") as audio_file:
    # Envía la solicitud
    response = requests.post(url, files={"file": audio_file})

# Verifica el estado de la respuesta
if response.status_code == 200:
    # Muestra la respuesta en formato JSON
    print("Predicciones:")
    print(response.json())
else:
    # Muestra el mensaje de error
    print(f"Error: {response.status_code}")
    print(response.json())
