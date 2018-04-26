import cv2

class PersonTracker(object):

    def __init__(self, image, box):
        self.tracker = cv2.TrackerMedianFlow_create()
        self.tracker.init(image=image, boundingBox=box)

    # track people in the frames
    def track(self, image):
            (ok, box) = self.tracker.update(image)
            if not ok:
                return (False, ok)
            return (ok, (int(box[0]), int(box[1]), int(box[2]), int(box[3])))