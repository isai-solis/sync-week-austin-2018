import cv2
import os
import sys
import unittest

if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import leapvision.scanner

class HumanScannerTest(unittest.TestCase):
    def setUp(self):
        self.scanner = leapvision.scanner.HumanScanner()
    def test_initial_state(self):
        self.assertNotEqual(self.scanner.hog, None)
    def test_single_person(self):
        test_image_path = os.path.join(os.path.dirname(__file__), 'data', 'person_beach.jpg')
        test_image = cv2.imread(test_image_path)
        result = self.scanner.scan(test_image)
        self.assertAlmostEqual(result, {'score':0.9, 'label':'person'})

if __name__ == '__main__':
    unittest.main()

