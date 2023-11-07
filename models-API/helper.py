import urllib.request
import cv2
import numpy as np
import layoutparser as lp


# import matplotlib.pyplot as plt


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


def convert_result_to_image(result) -> np.ndarray:
    """
    Convert network result of floating point numbers to image with integer
    values from 0-255. Values outside this range are clipped to 0 and 255.

    :param result: a single superresolution network result in N,C,H,W shape
    """
    result = result.squeeze(0).transpose(1, 2, 0)
    result *= 255
    result[result < 0] = 0
    result[result > 255] = 255
    result = result.astype(np.uint8)
    return result


def plot_layout(image, layout):
    # Show the detected layout of the input image
    pil_img = lp.draw_box(image, layout, box_width=3, color_map={"TextRegion": "green", "ImageRegion": "orange"},
                          show_element_type=True)
    # convert result to np array
    lp_img = np.array(pil_img)
    return lp_img


def encode_images(img_arr):
    encoded_img_list = []
    for np_img in img_arr:
        _, encoded_img = cv2.imencode('.png', np_img)
        encoded_img_list.append(encoded_img.tobytes())
    return encoded_img_list


def to_rgb(image_data) -> np.ndarray:
    """
    Convert image_data from BGR to RGB
    """
    return cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
