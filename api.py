import os
from datetime import datetime

from flask import Flask, request, jsonify
from pymongo import MongoClient
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise RuntimeError("MONGO_URI is not set")


# MongoDB connection
client = MongoClient(MONGO_URI)

db = client["smartsole_db"]

collection = db["sensor_logs"]


# Flask application
app = Flask(__name__)


# Home / health check
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "Smart Sole API is running",
        "service": "Smart Sole Backend",
        "endpoint": "/api/sensor"
    })


# ESP32 sensor endpoint
@app.route("/api/sensor", methods=["POST"])
def receive_data():

    data = request.get_json(silent=True)

    if not data:
        return jsonify({
            "status": "Invalid Data"
        }), 400

    document = data.copy()

    document["timestamp"] = datetime.utcnow()

    collection.insert_one(document)

    print(
        f"Received Data | "
        f"Heel: {data.get('heel', 0)}g | "
        f"Arch: {data.get('arch', 0)}g | "
        f"Meta1: {data.get('meta1', 0)}g | "
        f"Meta3: {data.get('meta3', 0)}g | "
        f"Meta5: {data.get('meta5', 0)}g | "
        f"Toe: {data.get('toe', 0)}g"
    )

    return jsonify({
        "status": "Success"
    }), 200


# Local testing
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
