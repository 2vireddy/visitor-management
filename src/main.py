import cv2
import numpy as np
import base64
import io
from PIL import Image
import json
import os
from appwrite.client import Client
from appwrite.services.databases import Databases

"""
  'req' variable has:
    'headers' - object with request headers
    'payload' - request body data as a string
    'variables' - object with function variables

  'res' variable has:
    'send(text, status)' - function to return text response. Status code defaults to 200
    'json(obj, status)' - function to return JSON response. Status code defaults to 200
"""

def extract_face_features(image):
    # Convert PIL Image to OpenCV format
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Load the pre-trained face detection cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Convert to grayscale
    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        return None
        
    # Get the largest face
    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
    x, y, w, h = largest_face
    
    # Extract face ROI
    face_roi = gray[y:y+h, x:x+w]
    
    # Resize to a standard size
    face_roi = cv2.resize(face_roi, (128, 128))
    
    # Flatten and normalize the features
    features = face_roi.flatten() / 255.0
    
    return features.tolist()

def compare_faces(face1_features, face2_features, threshold=0.8):
    if face1_features is None or face2_features is None:
        return False, 0.0
        
    # Convert to numpy arrays
    face1_array = np.array(face1_features)
    face2_array = np.array(face2_features)
    
    # Calculate correlation coefficient
    correlation = np.corrcoef(face1_array, face2_array)[0, 1]
    
    # Calculate similarity score (0 to 1)
    similarity = max(0, correlation)
    
    return similarity >= threshold, similarity

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
        get_encoding = data.get('getEncoding', False)
        
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
        
        # Extract face features
        incoming_face_features = extract_face_features(incoming_face_image)
        
        if incoming_face_features is None:
            return res.json({
                'success': False,
                'message': 'No face detected in the image'
            }, 400)

        # If only encoding is requested, return it
        if get_encoding:
            return res.json({
                'success': True,
                'encoding': incoming_face_features
            })

        # Get all visitors from database
        visitors = database.list_documents(
            os.environ['DATABASE_ID'],
            os.environ['VISITOR_COLLECTION_ID']
        )

        # Check each visitor's face
        best_match = None
        best_confidence = 0

        for visitor in visitors['documents']:
            if 'faceEncoding' not in visitor:
                continue

            stored_face_features = json.loads(visitor['faceEncoding'])
            is_match, confidence = compare_faces(stored_face_features, incoming_face_features)
            
            if is_match and confidence > best_confidence:
                best_match = visitor
                best_confidence = confidence

        if best_match:
            return res.json({
                'success': True,
                'matched': True,
                'visitorId': best_match['$id'],
                'confidence': float(best_confidence)
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
