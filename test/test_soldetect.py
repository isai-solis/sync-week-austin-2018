import cv2
import os
import sys
import unittest

if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import leapvision.soldetect

class PersonDetectorTest(unittest.TestCase):
    def setUp(self):
        self.soldetect = leapvision.soldetect.PersonDetector()

    def test_initial_state(self):
        self.assertNotEqual(self.soldetect.hog, None)

    def test_single_person(self):
        test_image_path = os.path.join(os.path.dirname(__file__), 'data', 'zeyn-afuang-258471-unsplash.jpg')
        test_image = cv2.imread(test_image_path)
        result = self.soldetect.scan(test_image)
        self.assertAlmostEqual(result[0]['score'], 0.8, delta=1)

if __name__ == '__main__':
    unittest.main()