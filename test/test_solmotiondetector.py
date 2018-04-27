import cv2
import os
import sys
import unittest

if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import leapvision.solmotiondetector

class PersonTrackerTest(unittest.TestCase):
    def setUp(self):
        self.video_path = os.path.join(
            os.path.dirname(__file__), 'data', 'clownswalking.mp4')
        self.camera = cv2.VideoCapture(self.video_path)
        ok, frame = self.camera.read()
        box = (1400, 320, 380, 730)
        #cv2.rectangle(frame, box[:2], (box[0]+box[2], box[1]+box[3]), (0, 255, 0))
        #cv2.imwrite("c:\\Users\\v-issoli\\isai.jpg", frame)
        self.tracker = leapvision.solmotiondetector.PersonTracker(frame, box)

    def test_initial_track(self):
        self.camera = cv2.VideoCapture(self.video_path)
        ok, frame = self.camera.read()
        ok, box = self.tracker.track(frame)
        self.assertTrue(ok)
        self.assertAlmostEqual(box[0], 1400, delta=2)
        self.assertAlmostEqual(box[1], 320, delta=2)
        self.assertAlmostEqual(box[2], 380, delta=2)
        self.assertAlmostEqual(box[3], 730, delta=2) 

    def test_update_track(self):
        for x in range (20):
            ok, frame = self.camera.read()
            ok, box = self.tracker.track(frame)
        self.assertTrue(ok)
        self.assertAlmostEqual(box[0], 1375, delta=2)
        self.assertAlmostEqual(box[1], 307, delta=2)
        self.assertAlmostEqual(box[2], 395, delta=2)
        self.assertAlmostEqual(box[3], 759, delta=2)
    
if __name__ == '__main__':
    unittest.main()