import streamlit as st
import urllib.request
import cv2
import time
import numpy as np
# import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO


def load_image(path: str) -> np.ndarray:
    """
    Loads an image from `path` and returns it as BGR numpy array.

    :param path: path to an image filename or url
    :return: image as numpy array, with BGR channel order
    """
    if type(path) != str:
        uploaded_file = path
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype="uint8")
        image = cv2.imdecode(file_bytes, -1)
    else:
        if path.startswith("http"):
            # Set User-Agent to Mozilla because some websites block requests
            # with User-Agent Python
            request = urllib.request.Request(path, headers={"User-Agent": "Mozilla/5.0"})
            response = urllib.request.urlopen(request)
            array = np.asarray(bytearray(response.read()), dtype="uint8")
            image = cv2.imdecode(array, -1)  # Loads the image as BGR
        else:
            image = cv2.imread(path)
    return image


def write_text_on_image(image: np.ndarray, text: str) -> np.ndarray:
    """
    Write the specified text in the top left corner of the image
    as white text with a black border.

    :param image: image as numpy arry with HWC shape, RGB or BGR
    :param text: text to write
    :return: image with written text, as numpy array
    """
    font = cv2.FONT_HERSHEY_PLAIN
    org = (20, 20)
    font_scale = 4
    font_color = (255, 255, 255)
    line_type = 1
    font_thickness = 2
    text_color_bg = (0, 0, 0)
    x, y = org

    image = cv2.UMat(image)
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    result_im = cv2.rectangle(image, org, (x + text_w, y + text_h), text_color_bg, -1)

    textim = cv2.putText(
        result_im,
        text,
        (x, y + text_h + font_scale - 1),
        font,
        font_scale,
        font_color,
        font_thickness,
        line_type,
    )
    return textim.get()


def generate_gif(img1, img2):
    st.header("Animated GIF Comparison")
    image_bicubic = write_text_on_image(image=img1, text="ORIGIN")
    image_super = write_text_on_image(image=img2, text="SUPER")

    img_array = [image_bicubic, image_super]
    image_gif = st.empty()
    i = 0
    while True:
        image_gif.image(img_array[i])
        i = not i
        time.sleep(2)


def pil_to_bytes(image):
    pil_image = Image.fromarray(np.squeeze(image).astype(np.uint8))
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG")
    byte_image = buffer.getvalue()
    return byte_image


def to_rgb(image_data) -> np.ndarray:
    """
    Convert image_data from BGR to RGB
    """
    return cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)


def list_to_np_arr(image):
    return np.array(image, dtype="uint8")
