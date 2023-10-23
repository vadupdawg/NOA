import json
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import os
import logging
from google.cloud import pubsub_v1, storage
from concurrent import futures
import datetime
from google.cloud import storage
from google.oauth2.service_account import Credentials


# Configuratie
CLOUD_BUCKET_NAME = "stt-noa"
PUBSUB_TOPIC = "long-running-tasks"
PROJECT = 'stt2langchain1'

# Logging setup
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Flask app initialisatie
app = Flask(__name__)
CORS(app)

publish_futures = []

service_account_info = json.loads(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'))
credentials = Credentials.from_service_account_info(service_account_info)

def get_callback(data):
    def callback(publish_future):
        try:
            print(publish_future.result(timeout=60))
        except futures.TimeoutError:
            print(f"Publishing {data} timed out.")
    return callback

def generate_upload_signed_url_v4(bucket_name, blob_name):


    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    url = blob.generate_signed_url(
        version="v4",
        # This URL is valid for 15 minutes
        expiration=datetime.timedelta(minutes=15),
        # Allow PUT requests using this URL.
        method="PUT",
        content_type="application/octet-stream",
    )

    print("Generated PUT signed URL:")
    print(url)
    print("You can use this URL with any user agent, for example:")
    print(
        "curl -X PUT -H 'Content-Type: application/octet-stream' "
        "--upload-file my-file '{}'".format(url)
    )
    return url


def publish_message(project_id, topic_id, message):
    publisher = pubsub_v1.PublisherClient(credentials=credentials)
    topic_path = publisher.topic_path(project_id, topic_id)
    
    publish_future = publisher.publish(topic_path, message.encode('utf-8'))
    publish_future.add_done_callback(get_callback(message))
    
    publish_futures.append(publish_future)


@app.route('/get_signed_url', methods=['POST'])
@cross_origin()
def get_signed_url_endpoint():
    request_data = request.json
    blob_name = request_data.get('blob_name')
    email = request_data.get('email')

    if not blob_name or not email:
        return jsonify({"error": "Blob name and email are required"}), 400

    # Het genereren van de signed URL
    signed_url = generate_upload_signed_url_v4(
        bucket_name=CLOUD_BUCKET_NAME,
        blob_name=blob_name
    )

    message_data = {
        'audio_file_name': blob_name,
        'email': email
    }
    message = json.dumps(message_data)
    publish_message(PROJECT, PUBSUB_TOPIC, message)

    futures.wait(publish_futures, return_when=futures.ALL_COMPLETED)
    publish_futures.clear()  # Clear the list

    return jsonify({"signed_url": signed_url}), 200



if __name__ == '__main__':
    app.run(port=int(os.environ.get("PORT", 8080)), host='0.0.0.0', debug=True)