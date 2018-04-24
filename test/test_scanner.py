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
        self.assertEqual(len(result), 9)
        self.assertEqual(result[0]['label'], 'person')
        self.assertAlmostEqual(result[0]['score'], 1.65, delta=1)
        self.assertEqual(result[0]['box'], (3939, 1820,  107,  214))

if __name__ == '__main__':
    unittest.main()

