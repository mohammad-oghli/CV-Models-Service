import cv2
import numpy as np
from helper import convert_result_to_image, to_rgb
import layoutparser as lp


def detect_layout(image, model):
    # Detect the layout of the input image
    layout = model.detect(image)
    return layout


def lp_ocr(image, lang):
    ocr_agent = lp.TesseractAgent(languages=lang)
    text = ocr_agent.detect(image)
    return text


def layout_parse_text(image, model):
    extracted_text = ""
    layout = detect_layout(image, model)
    text_blocks = lp.Layout([b for b in layout if b.type == 'TextRegion'])
    for block in text_blocks:
        # add padding in each image segment can help
        # improve robustness
        segment_image = (block
                         .pad(left=5, right=5, top=5, bottom=5)
                         .crop_image(image))
        text = lp_ocr(segment_image, 'eng')
        extracted_text += text + "\n---\n"
    # print(extracted_text)
    return extracted_text


def superresolution(full_image, exec_net):
    # Set the number of lines to crop from the network result to prevent
    # boundary effects. Should be an integer >= 1.
    CROPLINES = 2
    # See Superresolution on one crop of the image for description of CROP_FACTOR
    CROP_FACTOR = 1

    full_image_height, full_image_width = full_image.shape[:2]

    # Network inputs and outputs are dictionaries. Get the keys for the
    # dictionaries.
    original_image_key = list(exec_net.input_info)[0]
    bicubic_image_key = list(exec_net.input_info)[1]
    output_key = list(exec_net.outputs.keys())[0]

    # Get the expected input and target shape. `.dims[2:]` returns the height
    # and width. OpenCV's resize function expects the shape as (width, height),
    # so we reverse the shape with `[::-1]` and convert it to a tuple
    input_height, input_width = tuple(exec_net.input_info[original_image_key].tensor_desc.dims[2:])
    target_height, target_width = tuple(exec_net.input_info[bicubic_image_key].tensor_desc.dims[2:])

    upsample_factor = int(target_height / input_height)

    print(f"The network expects inputs with a width of {input_width}, " f"height of {input_height}")
    print(f"The network returns images with a width of {target_width}, " f"height of {target_height}")

    print(
        f"The image sides are upsampled by a factor {upsample_factor}. "
        f"The new image is {upsample_factor ** 2} times as large as the "
        "original image"
    )

    # Compute x and y coordinates of left top of image tiles
    x_coords = list(range(0, full_image_width, input_width * CROP_FACTOR - CROPLINES * 2))
    while full_image_width - x_coords[-1] < input_width * CROP_FACTOR:
        x_coords.pop(-1)
    y_coords = list(range(0, full_image_height, input_height * CROP_FACTOR - CROPLINES * 2))
    while full_image_height - y_coords[-1] < input_height * CROP_FACTOR:
        y_coords.pop(-1)

    # Compute the width and height to crop the full image. The full image is
    # cropped at the border to tiles of input size
    crop_width = x_coords[-1] + input_width * CROP_FACTOR
    crop_height = y_coords[-1] + input_height * CROP_FACTOR

    # Compute the width and height of the target superresolution image
    new_width = (
            x_coords[-1] * (upsample_factor // CROP_FACTOR)
            + target_width
            - CROPLINES * 2 * (upsample_factor // CROP_FACTOR)
    )
    new_height = (
            y_coords[-1] * (upsample_factor // CROP_FACTOR)
            + target_height
            - CROPLINES * 2 * (upsample_factor // CROP_FACTOR)
    )
    # print(f"The output image will have a width of {new_width} " f"and a height of {new_height}")

    # start_time = time.perf_counter()
    # patch_nr = 0
    # num_patches = len(x_coords) * len(y_coords)
    # progress_bar = ProgressBar(total=num_patches)
    # progress_bar.display()

    # Crop image to fit tiles of input size
    full_image_crop = full_image.copy()[:crop_height, :crop_width, :]

    # Create empty array of target size.
    full_superresolution_image = np.empty((new_height, new_width, 3), dtype=np.uint8)

    # Create bicubic upsampled image of target size for comparison
    full_bicubic_image = cv2.resize(
        src=full_image_crop[CROPLINES:-CROPLINES, CROPLINES:-CROPLINES, :],
        dsize=(new_width, new_height),
        interpolation=cv2.INTER_CUBIC,
    )

    total_inference_duration = 0
    for y in y_coords:
        for x in x_coords:
            # patch_nr += 1

            # Take a crop of the input image
            image_crop = full_image_crop[
                         y: y + input_height * CROP_FACTOR,
                         x: x + input_width * CROP_FACTOR,
                         ]

            # Resize the images to the target shape with bicubic interpolation
            bicubic_image = cv2.resize(
                src=image_crop,
                dsize=(target_width, target_height),
                interpolation=cv2.INTER_CUBIC,
            )

            if CROP_FACTOR > 1:
                image_crop = cv2.resize(src=image_crop, dsize=(input_width, input_height))

            input_image_original = np.expand_dims(image_crop.transpose(2, 0, 1), axis=0)

            input_image_bicubic = np.expand_dims(bicubic_image.transpose(2, 0, 1), axis=0)

            # Do inference
            # inference_start_time = time.perf_counter()

            network_result = exec_net.infer(
                inputs={
                    original_image_key: input_image_original,
                    bicubic_image_key: input_image_bicubic,
                }
            )

            # inference_stop_time = time.perf_counter()
            # inference_duration = inference_stop_time - inference_start_time
            # total_inference_duration += inference_duration

            # Reshape inference result to image shape and data type
            result = network_result[output_key]
            result_image = convert_result_to_image(result)

            # Add the inference result of this patch to the full superresolution
            # image
            adjusted_upsample_factor = upsample_factor // CROP_FACTOR
            new_y = y * adjusted_upsample_factor
            new_x = x * adjusted_upsample_factor
            full_superresolution_image[
            new_y: new_y + target_height - CROPLINES * adjusted_upsample_factor * 2,
            new_x: new_x + target_width - CROPLINES * adjusted_upsample_factor * 2,
            ] = result_image[
                CROPLINES * adjusted_upsample_factor: -CROPLINES * adjusted_upsample_factor,
                CROPLINES * adjusted_upsample_factor: -CROPLINES * adjusted_upsample_factor,
                :,
                ]

    #         progress_bar.progress = patch_nr
    #         progress_bar.update()
    #
    #         if patch_nr % 10 == 0:
    #             clear_output(wait=True)
    #             progress_bar.display()
    #             display(
    #                 Pretty(
    #                     f"Processed patch {patch_nr}/{num_patches}. "
    #                     f"Inference time: {inference_duration:.2f} seconds "
    #                     f"({1 / inference_duration:.2f} FPS)"
    #                 )
    #             )
    #
    # end_time = time.perf_counter()
    # duration = end_time - start_time
    # clear_output(wait=True)
    # print(
    #     f"Processed {num_patches} patches in {duration:.2f} seconds. "
    #     f"Total patches per second (including processing): "
    #     f"{num_patches / duration:.2f}.\nInference patches per second: "
    #     f"{num_patches / total_inference_duration:.2f} "
    # )
    return to_rgb(full_bicubic_image), to_rgb(full_superresolution_image)
