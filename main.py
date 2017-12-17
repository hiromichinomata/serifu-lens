import numpy as np
import cv2

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

# create text

def draw_text(img, text):
  draw = ImageDraw.Draw(img)
  draw.font = ImageFont.truetype("/Library/Fonts/ヒラギノ丸ゴ ProN W4.ttc", 20)

  position = np.array([0, 0])
  draw.text(position, text, (38, 57, 54))

img = Image.new("RGBA", (750, 80))
text = "いったいアイツはどこにいったんだ。まだOpenCVのビルドの途中だってのに、、、"
draw_text(img, text)
img.save('output/text.png')

# merge images

 # -1 to read alpha channel
# foreground: (210, 960, 4)
message_box_with_alpha =cv2.imread('samples/message_box.png', -1)
# background: (640, 960, 3)
hallway = cv2.imread('samples/hallway.jpg')
# message (80, 750, 3)
text = cv2.imread('output/text.png')

alpha = message_box_with_alpha[:,:,3]
alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR) # 1 -> 3
alpha = alpha / 255.0

message_box = message_box_with_alpha[:,:,:3]

m_h, m_w, _ = message_box.shape
h_h, _, _ = hallway.shape
hallway[h_h-m_h:h_h, 0:m_w] = (hallway[h_h-m_h:h_h, 0:m_w] * (1.0 - alpha)).astype('uint8')
hallway[h_h-m_h:h_h, 0:m_w] = (hallway[h_h-m_h:h_h, 0:m_w] + (message_box * alpha)).astype('uint8')

t_h, t_w, _ = text.shape
hallway[h_h-t_h-40:h_h-40, 105:t_w+105] += text

cv2.imwrite("output/result.png", hallway)
