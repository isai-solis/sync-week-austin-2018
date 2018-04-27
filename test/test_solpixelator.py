import cv2
import os
import sys
import unittest

if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import leapvision.solpixelator

class PixelatorTest(unittest.TestCase):

    def setUp(self):
        self.video_path = os.path.join(
            os.path.dirname(__file__), 
            'data',
            'girl-and-dad-faces.mp4'
        )
        self.camera = cv2.VideoCapture(self.video_path)
        frameOk, frame = cv2.camera.read()
        if not frameOk:
            print('Error: First frame was not captured')
        cv2.imwrite((os.path.join(os.path.dirname(__file__), 'data')), frame)
        self.pixelator = leapvision.solpixelator.Pixelator()
        cv2.imwrite()

    def test_pixelate(self):

        # pixOk, pix = self.pixelator.pixelate(frame)
        # self.assertTrue(pixOk)
        # self
        
        #absdiff
        #cv2.mean
        pass