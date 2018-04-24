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
        self.assertEqual(self.scanner.scoreThreshold, 0.6)
        self.assertEqual(self.scanner.winStride, (8, 8))
        self.assertEqual(self.scanner.padding, (8, 8))
        self.assertEqual(self.scanner.scale, 1.05)
        self.assertNotEqual(self.scanner.hog, None)
    def test_single_person(self):
        test_image_path = os.path.join(os.path.dirname(__file__), 'data', 'person_view.jpg')
        test_image = cv2.imread(test_image_path)
        result = self.scanner.scan(test_image)
        for entry in result:
            box = entry['box']
            pt1 = (box[0], box[1])
            pt2 = (pt1[0]+box[2], pt1[1]+box[3])
            cv2.rectangle(test_image, pt1, pt2, (0,255,0), 2)
            text = "Score : {}".format(entry['score'])
            cv2.putText(test_image, text, pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        cv2.imwrite('/tmp/human_scanner.test.jpg', test_image)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['label'], 'person')
        self.assertAlmostEqual(result[0]['score'], 0.89, delta=1)
        self.assertEqual(result[0]['box'], (2138, 437, 1319, 2638))

if __name__ == '__main__':
    unittest.main()

