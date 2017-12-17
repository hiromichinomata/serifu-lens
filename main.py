import numpy as np
import cv2

message_box =cv2.imread('samples/message_box.png')
hallway = cv2.imread('samples/hallway.jpg')

cv2.imwrite("output/result.png", hallway)
