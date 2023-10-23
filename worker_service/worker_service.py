from flask_cors import cross_origin
from pydub import AudioSegment
import openai
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import RefineDocumentsChain, LLMChain, SimpleSequentialChain
import logging
import json
from google import pubsub_v1
from google.cloud import storage
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from flask import Flask, request, jsonify
import base64
from google.cloud import tasks_v2
from google.cloud import storage

CLOUD_BUCKET_NAME = "stt-noa"
PROJECT = 'stt2langchain1'
access_token=os.getenv("HUGGINGFACE_API_KEY")

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
    message_data = json.loads(message)
    email = message_data.get('email')
    audio_file_name = message_data.get('audio_file_name')
    file_extension = os.path.splitext(audio_file_name)[1]
    local_file_name = f"{audio_file_name}_notule{file_extension}"
    download_audio_from_gcs(CLOUD_BUCKET_NAME, audio_file_name, local_file_name)
    result = select_audio_file(local_file_name)

    if result and email:
        transcript = result.get("transcript", "Geen transcript beschikbaar")
        summary = result.get("summary", "Geen samenvatting beschikbaar")
        send_email(email, transcript, summary)

    print(f"Verwerking voltooid. Resultaat: {result}")
    return "Message processed", 200

def download_audio_from_gcs(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")

def select_audio_file(file_path):

    if file_path:
        file_size = os.path.getsize(file_path)
        full_transcript = ""
        if file_size > 25 * 1024 * 1024:
            split_audio_files = split_audio_file(file_path)
            for split_file in split_audio_files:
                text = transcribe_audio(split_file)
                full_transcript += text + "\n"
        else:
            full_transcript = transcribe_audio(file_path)

        summary = summarize_transcript(full_transcript)
        return {"transcript": full_transcript, "summary": summary}
    
def transcribe_audio(file_path):

    with open(file_path, "rb") as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript['text']

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)


def split_audio_file(bucket_name, file_path):

    audio = AudioSegment.from_file(file_path)
    file_size = len(audio)
    chunk_length = 1000000
    chunks = [audio[i:i + chunk_length] for i in range(0, file_size, chunk_length)]
    
    split_files = []
    for i, chunk in enumerate(chunks):
        split_file_path = f"{file_path}_part_{i}.mp3"
        chunk.export(split_file_path, format="mp3")
        
        # Upload het gesplitste bestand naar de GCS bucket
        destination_blob_name = f"audio_chunks/{split_file_path}"
        upload_blob(bucket_name, split_file_path, destination_blob_name)
        
        split_files.append(destination_blob_name)

    return split_files

class CustomDocument:
    def __init__(self, page_content, metadata={}):
        self.page_content = page_content
        self.metadata = metadata

def summarize_transcript(transcript):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=15000, chunk_overlap=500
    )
    llm2 = ChatOpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"), model_name="gpt-4")
    llm = ChatOpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"), model_name="gpt-3.5-turbo-16k")

    docs = text_splitter.create_documents([transcript])
    
    document_prompt = PromptTemplate(
        input_variables=["page_content"],
        template="{page_content}"
    )

    document_variable_name = "context"
    initial_response_name = "prev_response"
    
    initial_prompt = PromptTemplate.from_template(
        """
        Het volgende is een deel van de vergadering:
        "{context}"
        Zoek hieruit de belangrijkste data voor een notule, actiepunten, gebeurtenissen, besprekingen, onderwerpen etc. 
        Hou de volgorde in stand en geef aan welke acties tot personen behoren.
        Gebruik zoveel mogelijk tokens.
        DATA:
        """
    )

    initial_llm_chain = LLMChain(
        llm=llm,
        prompt=initial_prompt,
        verbose=False
        )
    
    refine_prompt = PromptTemplate.from_template(
        """
        Je krijgt data van een deel van een vergadering.
        Het is jouw taak om deze data vorm te geven als een notule, het is niet erg dat de notule nog niet compleet is of eruit ziet als een notule, dat komt in een latere stap
        Hier is de data die je moet gebruiken: "{prev_response}"
        We hebben de mogelijkheid om de bestaande notule opnieuw te controleren.
        Pas aan waar meer context nodig is wanneer het in de bestaande notule niet duidelijk is.
        Blijf dus de hele tekst evalueren.
        ------------
        {context}
        ------------
        Gezien de nieuwe context, verfijn en verrijk de oorspronkelijke notule.
        Gebruik zoveel mogelijk tokens.
        """
    )
    
    refine_llm_chain = LLMChain(
        llm=llm,
        prompt=refine_prompt,
        verbose=False
        )
    
    refine_chain = RefineDocumentsChain(
        initial_llm_chain=initial_llm_chain,
        refine_llm_chain=refine_llm_chain,
        document_prompt=document_prompt,
        document_variable_name=document_variable_name,
        initial_response_name=initial_response_name,
        return_intermediate_steps=False,
        verbose=False
    )

    sort_prompt = """
        Gegeven de volgende notulen van een vergadering:
        {summary}
        Sorteer, categorizeer en verfijn deze notulen zodat wat opgeleverd wordt een overzichtelijk, informatief en duidelijke notule is.
        Probeer door middel van kopjes de notule te structureren.
        gebruik zoveel mogelijk tokens, maar zorg ervoor dat er geen data dubbel in de notule staat.
        Blijf dus de gehele tekst evalueren.
        """
    sort_prompt_template = PromptTemplate(template=sort_prompt, input_variables=["summary"])

    sort_llm_chain = LLMChain(
        llm=llm2,
        prompt=sort_prompt_template,
        verbose=False
        )
    
    overall_chain = SimpleSequentialChain(
        chains=[refine_chain, sort_llm_chain], 
        verbose=False
    )
    
    results = overall_chain.run(docs)
    
    return results

def send_email(email, transcript, summary):
    from_email = os.environ.get("FROM_EMAIL")
    email_password = os.environ.get("EMAIL_PASSWORD")
    to_email = email

    subject = "Uw vergadertranscript en samenvatting"

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject

    body = f"Transcript:\n{transcript}\n\nSamenvatting:\n{summary}"
    msg.attach(MIMEText(body, 'plain'))

    smtp_server = "mail.privateemail.com"
    smtp_port = 587 

    server = smtplib.SMTP(smtp_server, smtp_port)
    server.starttls()
    server.login(from_email, email_password)
    text = msg.as_string()
    server.sendmail(from_email, to_email, text)
    server.quit()

client = tasks_v2.CloudTasksClient()

project = PROJECT
queue = 'NOA'
location = 'europe-west1'
parent = client.queue_path(project, location, queue)


@app.route('/process_message', methods=['POST'])
@cross_origin()
def process_message_endpoint():
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

    url = 'https://worker-service-wtajjbsheq-ez.a.run.app/task_handler'
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
    payload = json.loads(request.data.decode())
    process_message(json.dumps(payload))
    return "Task completed", 200


if __name__ == '__main__':
    app.run(port=int(os.environ.get("PORT", 8080)), host='0.0.0.0', debug=True)

    subscriber = pubsub_v1.SubscriberClient()
    subscription_path = subscriber.subscription_path(PROJECT, 'long-running-tasks-sub')
    
    def callback(message):
        print(f"Received message: {message}")
        process_message(message.data.decode('utf-8'))
    subscriber.subscribe(subscription_path, callback=callback)
    