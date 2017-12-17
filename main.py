import numpy as np
import cv2

message_box =cv2.imread('samples/message_box.png')
hallway = cv2.imread('samples/hallway.jpg')

m_h, m_w, _ = message_box.shape
h_h, _, _ = hallway.shape
hallway[h_h-m_h:h_h, 0:m_w] = message_box

cv2.imwrite("output/result.png", hallway)
