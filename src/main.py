from appwrite.client import Client
from appwrite.services.databases import Databases
import face_recognition
import numpy as np
import base64
import io
from PIL import Image
import json
import os

"""
  'req' variable has:
    'headers' - object with request headers
    'payload' - request body data as a string
    'variables' - object with function variables

  'res' variable has:
    'send(text, status)' - function to return text response. Status code defaults to 200
    'json(obj, status)' - function to return JSON response. Status code defaults to 200
"""

def main(req, res):
    # Initialize Appwrite client
    client = Client()
    client.set_endpoint(os.environ['APPWRITE_FUNCTION_ENDPOINT'])
    client.set_project(os.environ['APPWRITE_FUNCTION_PROJECT_ID'])
    client.set_key(os.environ['APPWRITE_API_KEY'])

    # Initialize database service
    database = Databases(client)
    
    try:
        # Get input data
        data = json.loads(req.payload)
        incoming_face_base64 = data.get('faceImage')
        
        if not incoming_face_base64:
            return res.json({
                'success': False,
                'message': 'No face image provided'
            }, 400)

        # Convert base64 to image
        incoming_face_bytes = base64.b64decode(incoming_face_base64)
        incoming_face_image = Image.open(io.BytesIO(incoming_face_bytes))
        
        # Convert to RGB if necessary
        if incoming_face_image.mode != 'RGB':
            incoming_face_image = incoming_face_image.convert('RGB')
        
        # Convert to numpy array
        incoming_face_array = np.array(incoming_face_image)
        
        # Get face encodings
        incoming_face_encodings = face_recognition.face_encodings(incoming_face_array)
        
        if not incoming_face_encodings:
            return res.json({
                'success': False,
                'message': 'No face detected in the image'
            }, 400)

        incoming_face_encoding = incoming_face_encodings[0]

        # Get all visitors from database
        visitors = database.list_documents(
            os.environ['DATABASE_ID'],
            os.environ['VISITOR_COLLECTION_ID']
        )

        # Check each visitor's face
        for visitor in visitors['documents']:
            if 'faceEncoding' not in visitor:
                continue

            stored_face_encoding = np.array(json.loads(visitor['faceEncoding']))
            
            # Compare faces
            face_distance = face_recognition.face_distance(
                [stored_face_encoding],
                incoming_face_encoding
            )[0]

            # If faces match (distance less than 0.6)
            if face_distance < 0.6:
                return res.json({
                    'success': True,
                    'matched': True,
                    'visitorId': visitor['$id'],
                    'confidence': float(1 - face_distance)
                })

        # If no match found
        return res.json({
            'success': True,
            'matched': False,
            'message': 'No matching visitor found'
        })

    except Exception as e:
        return res.json({
            'success': False,
            'message': str(e)
        }, 500)
