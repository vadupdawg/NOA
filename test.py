from collections import defaultdict
import os
import base64
import time
import io
from io import BytesIO
from flask_cors import cross_origin
from flask import Flask, request, jsonify
from pydub import AudioSegment

import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
import logging
import json
from google.cloud import storage
from google.cloud import tasks_v2
from google.cloud import firestore
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

CLOUD_BUCKET_NAME = "stt-noa"
PROJECT = 'stt2langchain1'

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
    db = firestore.Client()

    message_data = json.loads(message)
    email = message_data.get('email')
    audio_file_name = message_data.get('audio_file_name')
    dynamic_fields = message_data.get('dynamic_fields', {})
    categories = list(dynamic_fields.values())
    model_type = message_data.get('model_type')
    order_id = message_data.get('order_id')
    amount = message_data.get('amount')
    natural_language_initiator = message_data.get('natural_language_initiator')
    order_ref = db.collection('orders').document(order_id)

    summaries = []
    transcripts = []

    if not check_file_upload(CLOUD_BUCKET_NAME, audio_file_name):
        return "File not uploaded", 400
    file_size = get_file_size_from_gcs(CLOUD_BUCKET_NAME, audio_file_name)

    if file_size > 25 * 1024 * 1024:
        split_audio_files = split_audio_file_gcs(CLOUD_BUCKET_NAME, audio_file_name)
        for split_file in split_audio_files:
            transcript = transcribe_audio_gcs(CLOUD_BUCKET_NAME, split_file)
            transcripts.append(transcript)

            if natural_language_initiator:
                classification_result = classify_text_content(transcript)
                categories = [category.name for category in classification_result.categories]
                        
            summary = summarize_transcript(transcript, categories, model_type)
            summaries.append(summary)
    else:
        transcript = transcribe_audio_gcs(CLOUD_BUCKET_NAME, audio_file_name)
        transcripts.append(transcript)

        if natural_language_initiator:
            classification_result = classify_text_content(transcript)
            categories = [category.name for category in classification_result.categories]
        
        summary = summarize_transcript(transcript, categories, model_type)
        summaries.append(summary)

    final_summary = "\n".join(summaries)
    final_transcript = "\n".join(transcripts)

    organized_summaries = defaultdict(list)
    current_key = None
    for summary in summaries:
        for line in summary.strip().split('\n'):
            if any(line.startswith(cat + ":") for cat in categories):
                current_key = line.rstrip(":")
            elif current_key is not None:
                organized_summaries[current_key].append(line.strip())

    final_summary = ""
    for key in categories:
        sub_summary = ' '.join(organized_summaries[key])
        llm = ChatOpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"), model_name="gpt-3.5-turbo-16k")
        template ="""
        Herzie de tekst met de volgende aandachtspunten:
        Verwijder alle dubbele regels.
        Verwijder regels die geen informatieve waarde hebben, zoals regels die alleen zeggen "niet vermeld", "geen informatie beschikbaar" of die enkel een opsommingsteken '-' bevatten zonder verdere tekst.
        Zorg ervoor dat elke regel past binnen het formaat van een notule.
        Hou het oorspronkelijke formaat aan, compleet met kopjes en opsommingstekens.

        Dit is het huidige verslag dat je moet herzien:
        "{docs}"

        Elke herziene regel moet op een nieuwe regel worden weergegeven en beginnen met een opsommingsteken '-'.

        Begin nu met herzien:
        """
        prompt = PromptTemplate(
            template=template, input_variables=["docs"]
        )
        llmchain = LLMChain(
            llm=llm,
            prompt=prompt,
            verbose=True
        )
        revised_sub_summary = llmchain.predict(docs=sub_summary)
        final_summary += f"\n\n{key}:\n{revised_sub_summary}"
    
        user_friendly_data = {
        'Bestandsnaam': audio_file_name,
        'E-mail': email,
        'Model Type': 'GPT-3' if model_type == 'gpt3' else 'GPT-4',
        'Dynamische Velden': ", ".join([f"{key}: {value}" for key, value in dynamic_fields.items()]),
        'Order ID': order_id,
        'Prijs (in centen)': amount
        }

    if email:
        formatted_final_summary = format_summary_text(final_summary)
        send_email(email, final_transcript, formatted_final_summary, user_friendly_data)

    extra_data = {
    'final_transcript': final_transcript,
    'organized_summaries': organized_summaries,
    'summaries': summaries,
    'final_summary': final_summary,
    'formatted_final_summary': formatted_final_summary,
    }
    order_ref.update(extra_data)

    print(f"Verwerking voltooid. Samenvatting: {formatted_final_summary}")
    return "Message processed", 200

def classify_text_content(text_content):
    client = LanguageServiceClient()

    document = Document(
        content=text_content,
        type_=Document.Type.PLAIN_TEXT
    )

    response = client.classify_text(document=document)

    return response

def format_summary_text(text):
    formatted_lines = []
    for line in text.split('\n'):
        if line.endswith(':'):
            formatted_lines.append("\n" + line)
        else:
            formatted_lines.extend([f"- {x.strip()}" for x in line.split('-') if x.strip()])
    formatted_text = '\n'.join(formatted_lines)
    return formatted_text

def check_file_upload(bucket_name, blob_name, retries=3, delay=10):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    for i in range(retries):
        blob = bucket.get_blob(blob_name)
        if blob and blob.size:
            print(f"Bestand {blob_name} is succesvol geüpload met grootte {blob.size} bytes.")
            return True
        print(f"Bestand {blob_name} is nog niet geüpload, wachten {delay} seconden...")
        time.sleep(delay)
        delay *= 2
    
    print(f"Bestand {blob_name} is niet geüpload na {retries} pogingen.")
    return False

def get_file_size_from_gcs(bucket_name, blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.get_blob(blob_name)
    return blob.size if blob else 0

def transcribe_audio_gcs(bucket_name, blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    audio_data = blob.download_as_bytes()
    audio_file = BytesIO(audio_data)
    audio_file.name = blob_name
    
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript['text']

def split_audio_file_gcs(bucket_name, blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    audio_data = blob.download_as_bytes()
    audio = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
    
    chunk_length = 300000
    chunks = [audio[i:i + chunk_length] for i in range(0, len(audio), chunk_length)]
    
    split_files = []
    for i, chunk in enumerate(chunks):
        buffer = io.BytesIO()
        chunk.export(buffer, format="mp3")
        
        buffer_size = len(buffer.getbuffer())
        if buffer_size > 20 * 1024 * 1024:
            continue
        
        split_file_path = f"{blob_name}_part_{i}.mp3"
        split_blob = bucket.blob(split_file_path)
        split_blob.upload_from_string(buffer.getvalue(), content_type='audio/mp3')
        split_files.append(split_file_path)

    return split_files

class CustomDocument:
    def __init__(self, page_content, metadata={}):
        self.page_content = page_content
        self.metadata = metadata

def summarize_transcript(transcript, categories, model_type):
    dynamic_prompt_parts = ["Extraheer uit het deel van de vergadering de volgende informatie\n'"]
    for cat in categories:
        dynamic_prompt_parts.append(f"{cat}:")
    dynamic_prompt_parts.append("'")
    dynamic_prompt = "\n".join(dynamic_prompt_parts)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000, chunk_overlap=2500
    )
    if model_type == 'gpt3':
        llm = ChatOpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"), model_name="gpt-3.5-turbo-16k")
    elif model_type == 'gpt4':
        llm = ChatOpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"), model_name="gpt-4")

    docs = text_splitter.create_documents([transcript])
    
    template ="""
        '
        {dynamic_prompt}
        '
        Schrijf zo uitgebreid en gedetailleerd mogelijk.
        Voeg onder deze kopjes de desbetreffende informatie met opsommingstekens, een opsommingsteken is "-".
        Je hoeft dus alleen de kopjes hierboven te gebruiken en kan de rest weglaten, ook hoef je niet te vertellen als waarnaar gevraagd is niet in de gegeven tekst te vinden is.
        Zorg ervoor dat elke regel opzichzelf genoeg verduidelijking geeft.
        Dit is een deel van de vergadering:
        "{docs}"
        Als je niks kan vinden voor een kopje dan hoef je dat niet te vermelden.
        Begin nu:
        """
    prompt = PromptTemplate(
        template=template, input_variables=["dynamic_prompt", "docs"]
    )

    llmchain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True
        )
    
    results = llmchain.predict(dynamic_prompt=dynamic_prompt, docs=docs)
    
    return results

def send_email(email, transcript, summary, user_friendly_data):
    from_email = os.environ.get("FROM_EMAIL")
    email_password = os.environ.get("EMAIL_PASSWORD")
    to_email = email

    subject = "Uw vergadertranscript en samenvatting"

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject

    body = f'''
        <html>
            <head></head>
            <body>
                <p>Details van uw order:</p>
                <br>
                <ul>
                    <li><strong>Order ID:</strong> {user_friendly_data['Order ID']}</li>
                    <li><strong>Bestandsnaam:</strong> {user_friendly_data['Bestandsnaam']}</li>
                    <li><strong>E-mail:</strong> {user_friendly_data['E-mail']}</li>
                    <li><strong>Model Type:</strong> {user_friendly_data['Model Type']}</li>
                    <li><strong>Dynamische Velden:</strong> {user_friendly_data['Dynamische Velden']}</li>
                    <li><strong>Prijs (in centen):</strong> {user_friendly_data['Prijs (in centen)']}</li> 
                </ul>
                <br>
                    <h2>Transcript:</h2>
                    <pre>{transcript}</pre>
                    <br>
                    <h2>Notule:</h2>
                    <pre>{summary}</pre>
                    <br>
                    <p>Met vriendelijke groet,</p>
                    <p>Het team van GroeimetAi.io</p>
            </body>
        </html>
        '''
    msg.attach(MIMEText(body, 'html'))


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