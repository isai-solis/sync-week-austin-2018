import cv2

class Pixelator(object):
    '''
    Represents a tracker for a single object. The algorithm used depends on the
    tracker_type parameter passed to the constructor.
    '''

    def __init__(self, shrinkRatio = None, growRatio = None):
        if not shrinkRatio:
            self.shrinkRatio = 0.1

        if not growRatio:
            self.growRatio = 10.0

    def pixelate(self, src):
        resized = cv2.resize(src, (0, 0), fx = 0.05, fy = 0.05, interpolation = INTER_NEAREST)
        pixelated = cv2.resize(resized, (0,0), fx = 20.0, fy = 20.0, interpolation = INTER_NEAREST)
        

        