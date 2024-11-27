from appwrite.client import Client
from appwrite.services.databases import Databases
import base64
import io
from PIL import Image
import json
import os
import math

"""
  'req' variable has:
    'headers' - object with request headers
    'payload' - request body data as a string
    'variables' - object with function variables

  'res' variable has:
    'send(text, status)' - function to return text response. Status code defaults to 200
    'json(obj, status)' - function to return JSON response. Status code defaults to 200
"""

def get_image_features(image, size=(64, 64)):
    """Convert image to grayscale and resize for comparison"""
    # Convert to grayscale
    gray_image = image.convert('L')
    # Resize for consistent comparison
    resized = gray_image.resize(size)
    # Get pixel data
    pixels = list(resized.getdata())
    # Normalize pixels
    avg = sum(pixels) / len(pixels)
    return [1 if p > avg else 0 for p in pixels]

def compare_features(features1, features2):
    """Compare two feature sets using Hamming distance"""
    if len(features1) != len(features2):
        return 0.0
    
    # Calculate Hamming distance
    matches = sum(1 for i in range(len(features1)) if features1[i] == features2[i])
    total = len(features1)
    
    # Convert to similarity score (0 to 1)
    return matches / total

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
        
        # Extract features
        incoming_features = get_image_features(incoming_face_image)

        # If only encoding is requested, return it
        if get_encoding:
            return res.json({
                'success': True,
                'encoding': incoming_features
            })

        # Get all visitors from database
        visitors = database.list_documents(
            os.environ['DATABASE_ID'],
            os.environ['VISITOR_COLLECTION_ID']
        )

        # Check each visitor's face
        best_match = None
        best_similarity = 0.0
        threshold = 0.80  # Minimum similarity threshold

        for visitor in visitors['documents']:
            if 'faceEncoding' not in visitor:
                continue

            stored_features = json.loads(visitor['faceEncoding'])
            similarity = compare_features(stored_features, incoming_features)
            
            if similarity > threshold and similarity > best_similarity:
                best_match = visitor
                best_similarity = similarity

        if best_match:
            return res.json({
                'success': True,
                'matched': True,
                'visitorId': best_match['$id'],
                'confidence': float(best_similarity)
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
