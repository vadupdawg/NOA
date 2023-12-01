import os
import base64
from io import BytesIO
from flask_cors import cross_origin
from flask import Flask, request, jsonify
from pydub import AudioSegment
from deepgram import Deepgram
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain.text_splitter import CharacterTextSplitter
import logging
import json
from google.cloud import storage
from google.cloud import tasks_v2
from google.cloud import firestore
import tempfile

CLOUD_BUCKET_NAME = "stt-noa"
PROJECT = 'stt2langchain1'

client = tasks_v2.CloudTasksClient()
project = PROJECT
queue = 'NOA'
location = 'europe-west1'
parent = client.queue_path(project, location, queue)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
global_log_fields = {}
def log_message(message, severity="NOTICE", **additional_fields):
    entry = {
        "message": message,
        "severity": severity,
        **global_log_fields,
        **additional_fields,
    }
    print(json.dumps(entry))

app = Flask(__name__)

def process_message(message):
#process the message from the pubsub topic
    db = firestore.Client()

    message_data = json.loads(message)
    email = message_data.get('email')
    audio_file_name = message_data.get('audio_file_name')
    order_id = message_data.get('order_id')
    order_ref = db.collection('orders').document(order_id)

    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )

    client.create_collection(
    collection_name=order_id,
    vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
    )

    transcribe_with_speakers = transcribe_audio_gcs(CLOUD_BUCKET_NAME, audio_file_name)

    # Upload naar Qdrant en controleer of het succesvol is
    upload_to_qdrant(transcribe_with_speakers, order_id)

    # Initialiseer of update het Firestore-document voor de order
    order_data = {
        'transcriptie_assemblyai': 'klaar',
        'full_transcript_with_speakers': transcribe_with_speakers
        }
    order_ref.update(order_data)

    return "qdrant succesful", 200

def upload_to_qdrant(transcription_from_assemblyai, collection_name):
#upload to qdrant
    client = QdrantClient(
        os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )

    embeddings = OpenAIEmbeddings()

    vector_store = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embeddings,
    )

    def get_chunks(text):
        text_splitter = CharacterTextSplitter(
            separator= "\n",
            chunk_size=50000, 
            chunk_overlap=10000,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        return chunks

    texts = get_chunks(transcription_from_assemblyai)
    vector_store.add_texts(texts)

    return "upload function done", 200

def format_transcript(response):
    formatted_transcript = ""
    if 'results' in response and 'channels' in response['results']:
        for channel in response['results']['channels']:
            for alternative in channel['alternatives']:
                for paragraph_info in alternative['paragraphs']['paragraphs']:
                    speaker_label = f"Speaker {paragraph_info['speaker']}"
                    for sentence in paragraph_info['sentences']:
                        formatted_transcript += f"{speaker_label}: {sentence['text']}\n"
    return formatted_transcript

def transcribe_audio_gcs(bucket_name, blob_name):
    # Initialiseer de Google Cloud Storage client
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # Download de audio data en converteer naar mp3
    audio_data = blob.download_as_bytes()
    audio_stream = BytesIO(audio_data)
    audio = AudioSegment.from_file(audio_stream)

    # Tijdelijk bestand voor het mp3 bestand
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_mp3:
        audio.export(temp_mp3.name, format="mp3")
        mp3_file_path = temp_mp3.name

    # Initialiseer Deepgram
    dg_client = Deepgram(os.getenv("DEEPGRAM_API_KEY"))

    with open(mp3_file_path, 'rb') as audio:
        source = {'buffer': audio, 'mimetype': 'audio/mp3'}
        options = {
            "model": "enhanced",
            "language": "nl",
            "smart_format": True,
            "punctuate": True,
            "paragraphs": True,
            "diarize": True,
            "filler_words": True,
            "utterances": True,
        }

        # Vraag transcriptie aan bij Deepgram
        response = dg_client.transcription.sync_prerecorded(source, options)
        transcript = format_transcript(response)

    # Verwijder het tijdelijke bestand
    os.remove(mp3_file_path)

    return transcript

@app.route('/process_message', methods=['POST'])
@cross_origin()
def process_message_endpoint():
#endpoint for the pubsub topic to send messages to the worker service to process the audio file
    message_data = request.json
    app.logger.info(f"Received message data: {message_data}")

    if not message_data:
        return jsonify({"error": "No message_data provided"}), 400

    encoded_data = message_data.get('message', {}).get('data')
    if encoded_data:
        decoded_data = base64.b64decode(encoded_data).decode('utf-8')
        message_data_json = json.loads(decoded_data)

        email = message_data_json.get('email')
        audio_file_name = message_data_json.get('audio_file_name')
    else:
        email = None 
        audio_file_name = None

    app.logger.info(f"Email: {email}, Audio File Name: {audio_file_name}")

    if not email or not audio_file_name:
        return jsonify({"error": "Missing email or audio_file_name"}), 400

    url = 'https://worker-diarization-wtajjbsheq-ez.a.run.app/task_handler'
    payload = json.dumps(message_data_json).encode()

    task = {
        'http_request': {
            'http_method': 'POST',
            'url': url,
            'headers': {
                'Content-type': 'application/json',
            },
            'body': payload
        }
    }

    client.create_task(request={"parent": parent, "task": task})

    return "Task queued", 200

@app.route('/task_handler', methods=['POST'])
def task_handler():
#handler for the cloud task that is created in the process_message_endpoint
    payload = json.loads(request.data.decode())
    process_message(json.dumps(payload))
    return "Task completed", 200

if __name__ == '__main__':
    app.run(port=int(os.environ.get("PORT", 8080)), host='0.0.0.0', debug=True)