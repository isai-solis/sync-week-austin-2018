import cv2

class PersonDetector(object):

    def __init__(self, scoreThreshold=0.6, winStride=(8, 8),
		padding=(8, 8), scale=1.05):
        self.scoreThreshold = scoreThreshold
        self.winStride = winStride
        self.padding = padding
        self.scale = scale
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # detect people in the image
    def scan(self, image):
        (rects, weights) = self.hog.detectMultiScale(image, winStride=self.winStride,
		padding=self.padding, scale=self.scale)
        result = list()
        for (rect, score) in zip(rects, weights):
            if score[0] < self.scoreThreshold or score[0] > 1.0:
                    continue
            result.append({
                'label': 'person',
                'score': score[0],
                'box': tuple(rect)
            })
        return result    
                

