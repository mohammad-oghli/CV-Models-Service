import streamlit as st
import requests
import threading
# import matplotlib.pyplot as plt
from helper import load_image, generate_gif, pil_to_bytes, to_rgb, list_to_np_arr


def run_model_endpoint(image_src):
    '''
    Run Model Inference endpoint
    '''
    end_point = "http://172.17.0.1:5000/models/super_resolution"
    full_image = load_image(image_src)
    if full_image is not None:
        try:
            result = requests.post(end_point, json={'image': full_image.tolist()})
            bicubic_image, superresolution_image = result.json()['origin'], result.json()['super']
            return list_to_np_arr(bicubic_image), list_to_np_arr(superresolution_image)
        except Exception:
            return "Endpoint Connection Error!"
    return "Sorry, loading image failed!"





def st_ui():
    '''
    Render the User Interface of the application endpoints
    '''
    st.title("Image Super Resolution")
    st.caption("Image Quality Enhancement")
    st.info("Single Image Super Resolution (SISR) Implementation by Oghli")
    container_hints = st.empty()
    with container_hints.container():
        st.markdown("* The model expects inputs with a width of 480, height of 270")
        st.markdown("* The model returns images with a width of 1920, height of 1080")
        st.markdown("* It enhances the resolution of the input image by a factor of 4")

    st.sidebar.subheader("Upload image to enhance its quality")
    uploaded_image = st.sidebar.file_uploader("Upload image", type=["png", "jpg"],
                                              accept_multiple_files=False, key=None,
                                              help="Image to enhance its quality")
    gen_btn = st.sidebar.button("Generate")
    col1, col2 = st.columns(2)
    s_msg = st.empty()
    example_image = load_image('example_image/space.jpg')
    ex_sub = col1.subheader("Image Sample")
    example = col1.image(to_rgb(example_image), use_column_width=True)
    if uploaded_image:
        s_msg = st.sidebar.success("Image uploaded successfully")
        example.empty()
        ex_sub.empty()

    if gen_btn:
        s_msg.empty()
        example.empty()
        ex_sub.empty()
        full_image = 'example_image/space.jpg'
        if uploaded_image:
            full_image = uploaded_image
        container_hints.empty()
        with st.spinner('Generating Super Resolution Image ...'):
            response = run_model_endpoint(full_image)
        if type(response) is tuple:
            bicubic_image, superresolution_image = response
            col1.subheader("Original Image")
            col1.image(bicubic_image, use_column_width=True)
            col2.subheader("SISR Image")
            col2.image(superresolution_image, use_column_width=True)
            byte_super_img = pil_to_bytes(superresolution_image)
            st.download_button(label="Download Result", data=byte_super_img,
                               file_name="superres_image_4x.jpeg", mime="image/jpeg")
            t = threading.Thread(target=generate_gif(bicubic_image, superresolution_image), name="gif_thread")
            t.daemon = True
            t.start()
        else:
            st.error(response)


if __name__ == "__main__":
    # render the app using streamlit ui function
    st_ui()
    # image_source = "images/mass_effect.jpg"
    # #image_source = "https://i.imgur.com/R5ovXDO.jpg"
    # full_bicubic_image, full_superresolution_image = cv_superresolution(image_source)
    # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(30, 15))
    # ax[0].imshow(full_bicubic_image)
    # ax[1].imshow(full_superresolution_image)
    # ax[0].set_title("Origin")
    # ax[1].set_title("Superresolution")
    # plt.show()
