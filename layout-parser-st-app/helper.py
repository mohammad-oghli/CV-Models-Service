import urllib.request
import cv2
import numpy as np
#import matplotlib.pyplot as plt


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


def to_rgb(image_data) -> np.ndarray:
    """
    Convert image_data from BGR to RGB
    """
    return cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)


def list_to_np_arr(image):
    return np.array(image, dtype="uint8")


# def plt_show(raw_image):
#     """
#     Use matplot to show image inline
#     raw_image: input image
#
#     :param: raw_image:image array
#     """
#     plt.figure(figsize=(10, 6))
#     plt.axis("off")
#     plt.imshow(raw_image)
#     plt.show()