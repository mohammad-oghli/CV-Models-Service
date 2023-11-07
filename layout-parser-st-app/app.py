import streamlit as st
import requests
from helper import load_image, to_rgb, list_to_np_arr


def detect_layout_endpoint(full_image):
    '''
    Run Layout parser Inference endpoint
    '''
    end_point = "http://172.17.0.1:5000/models/layout_parser"
    #full_image = load_image(image_src)
    if full_image is not None:
        try:
            result = requests.post(end_point, json={'image': full_image.tolist()})
            layout_image = result.json()['layout']
            return list_to_np_arr(layout_image)
        except Exception:
            return "Endpoint Connection Error!"
    return "Sorry, loading image failed!"


def parse_text_endpoint(full_image):
    '''
    Run Layout parser get_text Inference endpoint
    '''
    end_point = "http://172.17.0.1:5000/models/layout_parser/get_text"
    #full_image = load_image(image_src)
    if full_image is not None:
        try:
            result = requests.post(end_point, json={'image': full_image.tolist()})
            layout_img_text = result.json()['layout_text']
            return layout_img_text
        except Exception:
            return "Endpoint Connection Error!"
    return "Sorry, loading image failed!"


def st_ui():
    '''
    Render the User Interface of the application endpoints
    '''
    st.title("Layout Document Image Analysis")
    st.caption("Document Recognition")
    st.info("LayoutParser Implementation by Oghli")
    # hint = st.empty()
    # with hint.container():
    #     st.write("### The model detects vehicles and classify it according to:")
    #     st.write("####  • Type")
    #     st.write("####  • Color")
    st.sidebar.subheader("Upload document image to detect layout")
    uploaded_image = st.sidebar.file_uploader("Upload image", type=["png", "jpg"],
                                              accept_multiple_files=False, key=None,
                                              help="Image to detect layout")
    s_msg = st.empty()
    example_image = load_image('example_image/newspaper.jpg')
    st.subheader("Input Image")
    example = st.image(to_rgb(example_image))
    if example:
        placeholder = st.empty()
        de_btn = placeholder.button('Detect', key='1')
    if uploaded_image:
        placeholder.empty()
        example.empty()
        # hint.empty()
        st.image(uploaded_image)
        de_btn = st.button("Detect")
        s_msg = st.sidebar.success("Image uploaded successfully")

    if de_btn:
        s_msg.empty()
        # hint.empty()
        doc_image = load_image('example_image/newspaper.jpg')
        if uploaded_image:
            doc_image = load_image(uploaded_image)
        with st.spinner('Processing Image ...'):
            layout_image = detect_layout_endpoint(doc_image)
            if not example:
                example.empty()
                st.image(doc_image)
            st.subheader("Layout Image")
            st.image(layout_image)
            # byte_detect_img = pil_to_bytes(detection_image)
            # st.download_button(label="Download Result", data=byte_super_img,
            #                    file_name="detect_image.jpeg", mime="image/jpeg")
            layout_text = parse_text_endpoint(doc_image)
            st.subheader("Extracted Layout Text Region")
            st.text(layout_text)


if __name__ == "__main__":
    # render the app using streamlit ui function
    st_ui()
    # image_source = "example_image/newspaper.jpg"
    # layout_image = detect_layout_endpoint(image_source)
    # plt_show(layout_image)
