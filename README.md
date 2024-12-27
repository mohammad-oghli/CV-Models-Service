# Computer Vision Models As Service 

This project implements different **Computer Vision** Deep Learning Models as a service. It presents the capabilities of these models for creating ML applications.

It contains 3 pre-trained DL models which can be used in image processing and recognition.

The DL models are deployed as microservices and you can embed them on any application with one line of code using the provided Models API endpoints.

## Models
Currently, Deployed Deep Learning Models:

### Image Super Resolution

![superresulotion](https://i.imgur.com/IYuTz0t.png)

Super Resolution service is a Deep Learning model to enhance low resolution image to high quality image.

The service is based on **Single Image Super Resolution (SISR)** deep learning model which is available on Open Model Zoo, check this [page](https://docs.openvino.ai/latest/omz_models_model_single_image_super_resolution_1032.html) for more info.

**Super Resolution** is the process of enhancing the quality of an image by increasing the pixel count using deep learning.

* The model (Neural Network) expects inputs with a width of **480**, height of **270**.
* The model returns images with a width of **1920**, height of **1080**.
* The image sides are upsampled by a factor **4**. The new image is **16** times as large as the original image.

In brief:
* Input image should be: **480x270** resolution.
* Output image : **1920x1080** resolution.

It has applications in a number of domains including surveillance and security, medical imagery and enhancing Satellite images from the space.

You can use image samples in the **/sample_images** directory to test it on the model.


### Layout Parser
 ![layout_parser](https://i.imgur.com/29hBfTz.png)

 Layout Parser is a Deep Learning model for **Document Image Analysis**.

It provides the following functionality:

* layoutparser can be used for conveniently OCR documents and convert the output into structured data.

* With the help of Deep Learning, layoutparser supports the analysis of very complex documents and processing of the hierarchical structure in the layouts.

You can check full documentation of the project on this [GitHub repository](https://github.com/Layout-Parser/layout-parser).

## Project Structure

The project consists of the following:

* **Models API** : Flask web service that provides inference endpoints for each deep learning model. This service is the core of the project and other services will be integrated with it in order to process input image and recognize it.

* **Super Resolution App**: Streamlit data service integrated with models API service to enhance low resolution image to high resolution image by requesting single image super resolution inference endpoint of models API for each input image.

* **Layout Parser App**: Streamlit data service integrated with modles API service to detect layout different regions (**Text region**, **Image region**) for any document image. Also it can recognize text in text region using **Layout OCR** model and return the recognized text. Layout Parser deep learning model can be used to recognize and segment figure and text region in any document image and extract the data from them.

## How to call it

### Super Resolution Model

You can call the model by sending post request to this Models API endpoint with the input image as paramater
<pre>
endpoint = "http://localhost:5000/models/super_resolution"
image_source = "images/space.jpg"
image = cv2.imread(image_source)
res = requests.post(endpoint, json={'image': image.tolist()})
</pre>
It will return response in **json** format containing the key `super` for result image.

### Layout Parser Model
You can call the model detect layout inference by sending post request to this Models API endpoint with the input image as paramater

<pre>
endpoint = "http://localhost:5000/models/layout_parser"
image_source = "images/newspaper.jpg"
image = cv2.imread(image_source)
res = requests.post(endpoint, json={'image': image.tolist()})
</pre>
It will return response in **json** format containing the key `layout` for result image.

Also You can extract the text data from layout using OCR agent by sending post request to this Models API endpoint with the input image as paramater

<pre>
endpoint = "http://localhost:5000/models/layout_parser/get_text"
image_source = "images/newspaper.jpg"
image = cv2.imread(image_source)
res = requests.post(endpoint, json={'image': image.tolist()})
</pre>

It will return response in **json** format containing the key `layout_text` for result text.

You can check [this notebook](./models-API-notebook.ipynb) for live demo of calling models inference by Models API endpoints.

## Models Containerizing using Docker [MLOps]

The project root directory containing different `docker-compose` files for running each model application.

Each service has its own `Dockerfile` for deploying and running it using Docker engine.

The docker compose files:
* **docker-compose.sr**: This compose file for deploying and running **Super Resolution** Streamlit App with the integrated Models API service.

    Run the following command to build and run it in detach mode:

    `docker-compose -f docker-compose.sr.yml up -d`

    to check the running containers:

    `docker-compose -f docker-compose.sr.yml ps`

    or just use:

    `docker ps`

    You can stop running it using the command:

    `docker-compose -f docker-compose.sr.yml down`

* **docker-compose.lp**: This compose file for deploying and running **Layout Parser** Streamlit App with the integrated Models API service.

    Run the following command to build and run it in detach mode:

    `docker-compose -f docker-compose.lp.yml up -d`

* **docker-compose**: Default compose file for just deploying and running **Models API** service.

    Run the following command to build and run it in detach mode:

    `docker-compose up -d`

**Note**:

`Models API` service listen to port `5000` using the url:

 `http://localhost:5000`

`Streamlit` application listen to the port `8501` using the url:

`http://localhost:8501`

## References

[Single Image Super Resolution Research Paper](https://arxiv.org/abs/1807.06779)

[Layout Parser Toolkit](https://layout-parser.github.io/)

[Layout Parser Documentation](https://layout-parser.readthedocs.io/en/latest/index.html)

[Layout Parser Research Paper](https://arxiv.org/pdf/2103.15348.pdf)
