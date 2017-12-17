import numpy as np
import cv2

 # -1 to read alpha channel
# foreground: (210, 960, 4)
message_box_with_alpha =cv2.imread('samples/message_box.png', -1)
# background: (640, 960, 3)
hallway = cv2.imread('samples/hallway.jpg')

alpha = message_box_with_alpha[:,:,3]
alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR) # 1 -> 3
alpha = alpha / 255.0

message_box = message_box_with_alpha[:,:,:3]

m_h, m_w, _ = message_box.shape
h_h, _, _ = hallway.shape
hallway[h_h-m_h:h_h, 0:m_w] = (hallway[h_h-m_h:h_h, 0:m_w] * (1.0 - alpha)).astype('uint8')
hallway[h_h-m_h:h_h, 0:m_w] = (hallway[h_h-m_h:h_h, 0:m_w] + (message_box * alpha)).astype('uint8')

cv2.imwrite("output/result.png", hallway)
