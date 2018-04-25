import cv2


class ObjectTracker(object):
    def __init__(self, image, box, tracker_type='KCF'):
        self.tracker = self._create_tracker(tracker_type)
        self.tracker.init(image=image, boundingBox=box)
    
    def clear(self):
        self.tracker.clear()

    def _create_tracker(self, tracker_type):
        supported_types = {
            'BOOSTING': cv2.TrackerBoosting_create,
            'GOTURN': cv2.TrackerGOTURN_create,
            'KCF': cv2.TrackerKCF_create,
            'MIL': cv2.TrackerMIL_create,
            'MEDIANFLOW': cv2.TrackerMedianFlow_create,
            'TLD': cv2.TrackerTLD_create,
        }
        if not tracker_type in supported_types:
            raise "Unsupported tracker type {}".format(tracker_type)
        return supported_types[tracker_type]()
    
    def track(self, image):
        (ok, box) = self.tracker.update(image)
        if not ok:
            return (False, box)
        return (ok, (int(box[0]), int(box[1]), int(box[2]), int(box[3])))