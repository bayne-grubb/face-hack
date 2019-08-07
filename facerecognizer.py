import asyncio, io, glob, os, sys, time, uuid
from urllib.parse import urlparse
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person, SnapshotObjectType, OperationStatusType
import cv2
# Set the FACE_SUBSCRIPTION_KEY environment variable with your key as the value.
# This key will serve all examples in this document.
KEY = '71ae76026ffc4bdfa8836cd950d2d66d'
# Set the API endpoint for your Face subscription.
# You may need to change the first part ("westus") to match your subscription
ENDPOINT_STRING = "westcentralus"
ENDPOINT = 'https://{}.api.cognitive.microsoft.com/'.format(ENDPOINT_STRING)
face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))
video_capture = cv2.VideoCapture(0)
# Check success
if not video_capture.isOpened():
   raise Exception("Could not open video device")
# Read picture. ret === True on success
ret, frame = video_capture.read()
# Close device
video_capture.release()
detected_faces = face_client.face.detect_with_stream(frame)
print(detected_faces)
#cv2.imwrite('yeet.jpg', frame)