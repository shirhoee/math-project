import numpy as np
from PIL import Image, ImageDraw, ImageFont
img = Image.open('tests/fixtures/sample.jpg').convert('RGB')
draw = ImageDraw.Draw(img)
try:
    font = ImageFont.truetype("arial.ttf", 20)
except:
    font = ImageFont.load_default()
draw.rectangle([300, 300, 500, 500], outline="red", width=3)
draw.text((310, 310), "Near (Person Box)", fill="red", font=font)
draw.rectangle([0, 0, 200, 100], outline="blue", width=3)
draw.text((10, 10), "Far (Sky Box)", fill="blue", font=font)
img.save('docs/results/depth_direction_check.png')
