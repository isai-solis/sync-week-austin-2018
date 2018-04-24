import cv2

class HumanScanner(object):
    '''
    Scanner implementation that returns a list of bounding boxes for any
    humans detected in a given image.
    '''
    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def scan(self, image):
        (rects, weights) = self.hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)
        result = dict()
        for entry in zip(rects, weights):
            result['box'] = entry[0]
            result['score'] = entry[1]
        return result
