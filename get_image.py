from PIL import Image
import numpy
import requests
import cv2
class Get_Image:
    def __init__(self):
        self.img = None
    def get_image_from_link(self,url):
        img = Image.open(requests.get(url, stream=True).raw).convert('RGB')
        open_cv_image = numpy.array(img)
        # Convert RGB to BGR
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        return open_cv_image