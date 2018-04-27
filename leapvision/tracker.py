import cv2
import rtree
import foundation


class SingleObjectTracker(object):
    '''
    Represents a tracker for a single object. The algorithm used depends on the
    tracker_type parameter passed to the constructor.
    '''

    def __init__(self, image, box, trackerType='KCF'):
        self.tracker = self._createTracker(trackerType)
        self.tracker.init(image=image, boundingBox=box)
        self.box = box

    def clear(self):
        self.tracker.clear()

    def _createTracker(self, trackerType):
        supportedTypes = {
            'BOOSTING': cv2.TrackerBoosting_create,
            'GOTURN': cv2.TrackerGOTURN_create,
            'KCF': cv2.TrackerKCF_create,
            'MIL': cv2.TrackerMIL_create,
            'MEDIANFLOW': cv2.TrackerMedianFlow_create,
            'TLD': cv2.TrackerTLD_create,
        }
        if not trackerType in supportedTypes:
            raise "Unsupported tracker type {}".format(trackerType)
        return supportedTypes[trackerType]()

    def track(self, image):
        (ok, box) = self.tracker.update(image)
        if not ok:
            return (False, box)
        self.box = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        return (ok, self.box)


class MultiObjectTrackerEntry(object):
    def __init__(self, trackedObject, tracker):
        self.trackedObject = trackedObject
        self.tracker = tracker

    def __getitem__(self, key):
        if key == 'box':
            return self.tracker.box
        return self.trackedObject[key]


class MultiObjectTracker(object):
    '''
    Represents a tracker for a one or more objects. The algorithm used for each
    object depends on the tracker_type parameter passed to the constructor.
    '''

    def __init__(self, trackerType='KCF'):
        self.trackerType = trackerType
        self.entries = list()
        self.index = rtree.index.Index()

    def add(self, image, objects):
        '''
        Starts tracking the objects provided in the form:
        {
            'score': float,
            'box': (x1, y1, x2, y2),
            ...trackerType
        }
        '''
        for value in objects:
            box = value['box']
            tracker = SingleObjectTracker(
                image,
                box,
                trackerType=self.trackerType
            )
            self.entries.append(MultiObjectTrackerEntry(value, tracker))

        self.entries = foundation.non_max_suppression(
            self.entries
        )
        self.index = rtree.index.Index()
        for (i, entry) in enumerate(self.entries):
            box = entry['box']
            self.index.insert(i, box)

    def track(self, image):
        # TODO: Consider updating only the trackers that intersect with motion.
        entriesToRemove = list()
        boxes = list()
        for entry in self.entries:
            (ok, box) = entry.tracker.track(image)
            if ok:
                boxes.append(box)
            else:
                entriesToRemove.append(entry)
        for entry in entriesToRemove:
            self.entries.remove(entry)
        return (True, boxes) if boxes else (False, list())
