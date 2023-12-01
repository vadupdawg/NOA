import json
from concurrent import futures
import os
import logging
from datetime import timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from google.cloud import pubsub_v1, storage
from google.cloud import storage
from google.cloud import firestore
from google.oauth2.service_account import Credentials
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import stripe

stripe.api_key = os.getenv("STRIPE_API_KEY")

CLOUD_BUCKET_NAME = "stt-noa"
PUBSUB_TOPIC = "long-running-tasks"
PROJECT = 'stt2langchain1'

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

publish_futures = []

service_account_info = json.loads(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'))
credentials = Credentials.from_service_account_info(service_account_info)

def get_callback(data):
    def callback(publish_future):
        try:
            print(publish_future.result(timeout=120))
        except futures.TimeoutError:
            print(f"Publishing {data} timed out.")
    return callback

def generate_signed_url(bucket_name, blob_name):
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    url = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(minutes=60),
        method="PUT",
        content_type="application/octet-stream"
    )
    return url

def publish_message(project_id, topic_id, message):
    publisher = pubsub_v1.PublisherClient(credentials=credentials)
    topic_path = publisher.topic_path(project_id, topic_id)
    
    publish_future = publisher.publish(topic_path, message.encode('utf-8'))
    publish_future.add_done_callback(get_callback(message))
    
    publish_futures.append(publish_future)

def send_email(email, order_id, user_friendly_data):
    from_email = os.environ.get("FROM_EMAIL")
    email_password = os.environ.get("EMAIL_PASSWORD")
    to_email = email

    subject = "Uw order is aangemaakt!"

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject

    body = f'''
        <html>
            <head></head>
            <body>
                <p>Transactie voor ordernummer <strong>{order_id}</strong> is voltooid en verwerking is gestart. Uw transcript, tekst en termen om naar te zoeken volgen in de volgende mail!</p>
                <br>
                <ul>
                    <li><strong>Bestandsnaam:</strong> {user_friendly_data['Bestandsnaam']}</li>
                    <li><strong>Prijs (in centen):</strong> {user_friendly_data['Prijs (in centen)']}</li>
                    <li><strong>E-mail:</strong> {user_friendly_data['E-mail']}</li>
                    <li><strong>Model Type:</strong> {user_friendly_data['Model Type']}</li>
                    <li><strong>Dynamische Velden:</strong> {user_friendly_data['Dynamische Velden']}</li>
                    <li><strong>Status:</strong> {user_friendly_data['Status']}</li>
                </ul>
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

@app.route('/create-checkout-session', methods=['POST'])
@cross_origin()
def create_checkout_session():
        data = request.json
        file_name = data.get('fileName')
        amount = data.get('amount')
        email = data.get('email')
        modelType = data.get('modelType')
        dynamicFields = data.get('dynamicFields')
        natural_language_initiator = data.get('natural_language_initiator')

        signed_url = generate_signed_url(CLOUD_BUCKET_NAME, file_name)

        db = firestore.Client(credentials=credentials)

        order_ref = db.collection('orders').document()
        order_data = {
            'file_name': file_name,
            'amount': amount,
            'email': email,
            'model_type': modelType,
            'dynamic_fields': dynamicFields,
            'status': 'pending',
            'natural_language_initiator': natural_language_initiator,
        }
        order_ref.set(order_data)
        user_friendly_data = {
            'Bestandsnaam': order_data['file_name'],
            'Prijs (in centen)': order_data['amount'],
            'E-mail': order_data['email'],
            'Model Type': 'GPT-3' if order_data['model_type'] == 'gpt3' else 'GPT-4',
            'Dynamische Velden': ", ".join([f"{key}: {value}" for key, value in order_data['dynamic_fields'].items()]),
            'Status': 'In behandeling' if order_data['status'] == 'pending' else 'Voltooid',
            'Slimme termen detectie' : order_data['natural_language_initiator']
        }

        checkout_session = stripe.checkout.Session.create(
            client_reference_id=order_ref.id,
            payment_method_types=['card','ideal'],
            line_items=[{
                'price_data': {
                    'currency': 'eur',
                    'unit_amount': int(amount),
                    'product_data': {
                        'name': 'Audio File Processing',
                    },
                },
                'quantity': 1,
            }],
            mode='payment',
            success_url='https://frontend-wtajjbsheq-ez.a.run.app/success.html',
            cancel_url='https://frontend-wtajjbsheq-ez.a.run.app/',
        )

        send_email(email, order_ref.id, user_friendly_data)

        return jsonify(id=checkout_session.id, signed_url=signed_url)

@app.route('/webhook', methods=['POST'])
def stripe_webhook():
    payload = request.data.decode('utf-8')
    sig_header = request.headers.get('Stripe-Signature')
    endpoint_secret = os.getenv("STRIPE_SIGN_SECRET")

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, endpoint_secret
        )
    except stripe.error.SignatureVerificationError as e:
        return jsonify({'error': 'Invalid signature'}), 400
    except stripe.error.InvalidRequestError as e:
        return jsonify({'error': 'Invalid request'}), 400

    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        order_id = session['client_reference_id']

        db = firestore.Client(credentials=credentials)
        order_ref = db.collection('orders').document(order_id)
        order_data = order_ref.get().to_dict()

        message = json.dumps({
            'audio_file_name': order_data['file_name'],
            'email': order_data['email'],
            'dynamic_fields': order_data['dynamic_fields'],
            'model_type': order_data['model_type'],
            'order_id': order_id,
            'amount': order_data['amount'],
            'natural_language_initiator': order_data['natural_language_initiator'],
        })
        publish_message(PROJECT, PUBSUB_TOPIC, message)

        futures.wait(publish_futures, return_when=futures.ALL_COMPLETED)
        publish_futures.clear()

    return jsonify({'status': 'success'}), 200

if __name__ == '__main__':
    app.run(port=int(os.environ.get("PORT", 8080)), host='0.0.0.0', debug=True)