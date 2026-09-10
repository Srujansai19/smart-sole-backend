import os
import logging
from flask import Flask, request, jsonify
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient

# Disable default Flask logging to keep terminal clean
log = logging.getLogger('werkzeug')
log.disabled = True

load_dotenv()
mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri)
db = client['smartsole_db']
collection = db['sensor_logs']

app = Flask(__name__)

@app.route('/api/sensor', methods=['POST'])
def receive_data():
    data = request.json
    if data:
        db_document = data.copy()
        db_document["timestamp"] = datetime.now()
        collection.insert_one(db_document)
        print(f"\rReceived Data - Heel: {data.get('heel', 0)}g | Arch: {data.get('arch', 0)}g | Toe: {data.get('toe', 0)}g   ", end="", flush=True)
        return jsonify({"status": "Success"}), 200
    return jsonify({"status": "Invalid Data"}), 400

if __name__ == '__main__':
    print("Flask Receiver Started. Listening for ESP32 data...")
    app.run(host='0.0.0.0', port=5000)