import cv2
import os
import sys
import unittest

if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import leapvision.solfacedetector


class FaceDetectorTest(unittest.TestCase):
    def setUp(self):
        self.solfacedetector = leapvision.solfacedetector.FaceDetector()
        print(self.solfacedetector)
        
    def test_facedetect(self):
        test_video_path = os.path.join(
                os.path.dirname(__file__),
                'data',
                'girl-and-dad-faces.mp4'
            )
        test_video = cv2.VideoCapture(test_video_path)
        ok, frame = test_video.read()
        self.assertTrue(ok)
        result = self.solfacedetector.facedetect(frame)
        print(result)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['label'], 'person')
        self.assertEqual(result[0]['box'], (1010, 3, 448, 448))

if __name__ == '__main__':
    unittest.main()    