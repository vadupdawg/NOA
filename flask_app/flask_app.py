import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS, cross_origin
import os
import logging
from google.cloud import pubsub_v1, storage
from concurrent import futures
from google.cloud import storage
from google.cloud import firestore
from google.oauth2.service_account import Credentials
import stripe
from datetime import timedelta

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

@app.route('/create-checkout-session', methods=['POST'])
@cross_origin()
def create_checkout_session():
        data = request.json
        file_name = data.get('fileName')
        amount = data.get('amount')
        email = data.get('email')
        modelType = data.get('modelType')
        dynamicFields = data.get('dynamicFields')

        signed_url = generate_signed_url(CLOUD_BUCKET_NAME, file_name)
        print(file_name)
        print(amount)
        print(dynamicFields)
        print(signed_url)

        db = firestore.Client(credentials=credentials)

        order_ref = db.collection('orders').document()
        order_data = {
            'file_name': file_name,
            'amount': amount,
            'email': email,
            'model_type': modelType,
            'dynamic_fields': dynamicFields,
            'status': 'pending'
        }
        order_ref.set(order_data)

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
            success_url='https://frontend-wtajjbsheq-ez.a.run.app/succes.html',
            cancel_url='https://frontend-wtajjbsheq-ez.a.run.app/',
        )
        
        # Stuur de signed URL terug naar de client, samen met de checkout session ID
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
            'model_type': order_data['model_type']
        })
        publish_message(PROJECT, PUBSUB_TOPIC, message)

        futures.wait(publish_futures, return_when=futures.ALL_COMPLETED)
        publish_futures.clear()

    return jsonify({'status': 'success'}), 200

if __name__ == '__main__':
    app.run(port=int(os.environ.get("PORT", 8080)), host='0.0.0.0', debug=True)