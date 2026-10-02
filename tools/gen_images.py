import numpy as np
from PIL import Image, ImageDraw
import os

os.makedirs('tests/fixtures/gen', exist_ok=True)

# 1. Indoor Scene (Corridor perspective)
img1 = Image.new('RGB', (800, 534), color=(200, 200, 200))
d1 = ImageDraw.Draw(img1)
# walls
d1.polygon([(0, 0), (200, 200), (200, 334), (0, 534)], fill=(150, 150, 150))
d1.polygon([(800, 0), (600, 200), (600, 334), (800, 534)], fill=(150, 150, 150))
# floor
d1.polygon([(0, 534), (200, 334), (600, 334), (800, 534)], fill=(100, 100, 100))
# back wall
d1.rectangle([200, 200, 600, 334], fill=(220, 220, 220))
img1.save('tests/fixtures/gen/indoor.jpg')

# 2. Landscape (Horizon)
img2 = Image.new('RGB', (800, 534), color=(135, 206, 235)) # Sky
d2 = ImageDraw.Draw(img2)
# Mountains
d2.polygon([(0, 300), (200, 150), (400, 300)], fill=(105, 105, 105))
d2.polygon([(300, 300), (500, 100), (800, 300)], fill=(120, 120, 120))
# Ground
d2.rectangle([0, 300, 800, 534], fill=(34, 139, 34))
img2.save('tests/fixtures/gen/landscape.jpg')

# 3. Close-up portrait/object
img3 = Image.new('RGB', (800, 534), color=(255, 255, 255))
d3 = ImageDraw.Draw(img3)
d3.ellipse([200, 50, 600, 450], fill=(255, 200, 150)) # Face/Object
d3.rectangle([300, 450, 500, 534], fill=(255, 150, 100)) # Body
img3.save('tests/fixtures/gen/closeup.jpg')

print("Generated 3 test images.")
