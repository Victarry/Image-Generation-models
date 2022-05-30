from PIL import Image
from pathlib import Path
import os

path = Path(".")
for x in path.glob("**/*.png"):
    img = Image.open(x)
    os.remove(x)
    new_x = x.with_suffix(".jpg")
    img.save(new_x)