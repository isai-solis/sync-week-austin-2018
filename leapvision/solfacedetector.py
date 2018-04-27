import cv2
import os

class FaceDetector(object):
    '''
    Face detector implementation that returns bounding boxes for faces in an image.
    '''

    def __init__(self, cascPath = None):
        if not cascPath:
            self.cascPath = path_to_cascade = os.path.join(
                cv2.__path__[0], 
                'data', 
                'haarcascade_frontalface_default.xml'
            )
        else:
            self.cascPath = cascPath

        
        self.faceCascade = cv2.CascadeClassifier(self.cascPath)
    
    def facedetect(self, image):
        faces = self.faceCascade.detectMultiScale(
            image, 
            scaleFactor= 1.1,
            minNeighbors= 5,
            minSize= (30, 30),
            flags= cv2.CASCADE_SCALE_IMAGE
        )
        result = list()
        for facerect in faces:
            result.append({
            'label': 'person',
            'box': tuple(facerect)
            })
        return result
            