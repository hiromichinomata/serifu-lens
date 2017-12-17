import numpy as np
import cv2

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

# create text

def draw_text(img, text, color):
  draw = ImageDraw.Draw(img)
  draw.font = ImageFont.truetype("/Library/Fonts/ヒラギノ丸ゴ ProN W4.ttc", 20)

  position = np.array([0, 0])
  draw.text(position, text, color)

img = Image.new("RGBA", (750, 80))
text = "いったいアイツはどこにいったんだ。まだOpenCVのビルドの途中だってのに、、、"
draw_text(img, text, (38, 57, 54))
img.save('output/text.png')

img = Image.new("RGBA", (250, 30))
text = "CTO"
draw_text(img, text, (255, 255, 255))
img.save('output/speaker.png')

# merge images

def merge_images(background, foreground_with_alpha, o_h, o_w):
    alpha = foreground_with_alpha[:,:,3]
    alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR) # 1 -> 3
    alpha = alpha / 255.0

    foreground = foreground_with_alpha[:,:,:3]

    f_h, f_w, _ = foreground.shape
    b_h, _, _ = background.shape
    background[b_h-f_h-o_h:b_h-o_h, o_w:f_w+o_w] = (background[b_h-f_h-o_h:b_h-o_h, o_w:f_w+o_w] * (1.0 - alpha)).astype('uint8')
    background[b_h-f_h-o_h:b_h-o_h, o_w:f_w+o_w] = (background[b_h-f_h-o_h:b_h-o_h, o_w:f_w+o_w] + (foreground * alpha)).astype('uint8')

    return background

 # -1 to read alpha channel
# foreground: (210, 960, 4)
message_box_with_alpha =cv2.imread('samples/message_box.png', -1)
# background: (640, 960, 3)
hallway = cv2.imread('samples/hallway.jpg')
hallway = merge_images(hallway, message_box_with_alpha, 0, 0)

# text (80, 750, 3)
text_with_alpha = cv2.imread('output/text.png', -1)
hallway = merge_images(hallway, text_with_alpha, 40, 105)

# speaker (30, 250, 3)
speaker_with_alpha = cv2.imread('output/speaker.png', -1)
hallway = merge_images(hallway, speaker_with_alpha, 145, 50)

cv2.imwrite("output/result.png", hallway)
