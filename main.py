import numpy as np
import cv2

message_box =cv2.imread('samples/message_box.png')
hallway = cv2.imread('samples/hallway.jpg')
cv2.imshow('image', message_box)
cv2.waitKey(0)
cv2.imshow('image', hallway)
cv2.waitKey(0)
cv2.destroyAllWindows()
