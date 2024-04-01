import os
from PIL import Image
import pyheif


def heic_to_jpeg(heic_file):
    # Open HEIC image using pyheif
    heif_file = pyheif.read(heic_file)
    
    # Convert it to other image format (e.g. png)
    image = Image.frombytes(
        heif_file.mode, 
        heif_file.size, 
        heif_file.data,
        "raw",
        heif_file.mode,
        heif_file.stride,
    )
    
    # Get the file name without extension and add .jpg
    jpg_file = os.path.splitext(heic_file)[0] + '.jpg'
    
    # Save as JPEG with high quality
    image.save(jpg_file, 'JPEG', quality=95)

heic_to_jpeg('20240118_123155501_iOS.heic')