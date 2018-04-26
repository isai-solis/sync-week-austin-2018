import cv2
import os
import sys
import unittest

if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import leapvision.tracker


class SingleObjectTrackerTest(unittest.TestCase):
    def setUp(self):
        self.video_path = os.path.join(
            os.path.dirname(__file__),
            'data',
            'people_walking.mp4'
        )
        self.camera = cv2.VideoCapture(self.video_path)
        ok, frame = self.camera.read()
        box = (421, 417, 100, 160)
        self.tracker = leapvision.tracker.SingleObjectTracker(
            frame,
            box
        )

    def test_same_image(self):
        self.camera = cv2.VideoCapture(self.video_path)
        ok, frame = self.camera.read()
        ok, box = self.tracker.track(frame)
        self.assertTrue(ok)
        self.assertAlmostEqual(box[0], 421, delta=2)
        self.assertAlmostEqual(box[1], 417, delta=2)
        self.assertAlmostEqual(box[2], 100, delta=2)
        self.assertAlmostEqual(box[3], 160, delta=2)

    def test_future_frame(self):
        ok, frame = self.camera.read()
        for x in range(30):
            ok, frame = self.camera.read()
            ok, box = self.tracker.track(frame)
        self.assertTrue(ok)
        self.assertAlmostEqual(box[0], 423, delta=2)
        self.assertAlmostEqual(box[1], 325, delta=2)
        self.assertAlmostEqual(box[2], 100, delta=2)
        self.assertAlmostEqual(box[3], 160, delta=2)

    def test_demo(self):
        enabled = bool(os.environ.get('TRACKER_DEMO_ENABLED'))
        if not enabled:
            return
        video_path = os.path.join(
            os.path.dirname(__file__),
            'data',
            'people_walking.mp4'
        )
        key = 0
        while key & 0xff != ord('q'):
            camera = cv2.VideoCapture(video_path)
            ok, frame = camera.read()
            box = (421, 417, 100, 160)
            self.tracker = leapvision.tracker.SingleObjectTracker(
                frame,
                box
            )
            cv2.namedWindow('camera')
            cv2.moveWindow('camera', 0, 0)
            while camera.isOpened():
                ok, frame = camera.read()
                if not ok:
                    break
                tick_count = cv2.getTickCount()
                ok, box = self.tracker.track(frame)
                fps = cv2.getTickFrequency() / (cv2.getTickCount() - tick_count)
                if not ok:
                    continue
                cv2.putText(
                    frame,
                    "fps: {}".format(int(fps)),
                    (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 0),
                    2
                )
                cv2.rectangle(
                    frame,
                    box[:2],
                    (box[0]+box[2], box[1]+box[3]),
                    (0, 255, 0),
                    2
                )
                cv2.imshow('camera', frame)
                key = cv2.waitKey(1)
            self.tracker.clear()


if __name__ == '__main__':
    unittest.main()
