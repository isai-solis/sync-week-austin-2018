#!/usr/bin/env python

import cv2

def should_quit():
  return cv2.waitKey(1) & 0xFF == ord('q')

camera = cv2.VideoCapture(0)
cv2.namedWindow('My Video')
while camera.isOpened():
  (ok, frame) = camera.read()
  if not ok:
    break
  cv2.imshow('My Video', frame)
  if should_quit():
    break

