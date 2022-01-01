import os
import numpy as np
import cv2
import streamlit as st
import streamlit.components.v1 as stc
from PIL import Image


@st.cache
def load_image(image_file):
    img = Image.open(image_file)
    return img


def read_image(path):
    img = cv2.imread(path)
    return img


def upload():
    image_file = st.file_uploader("Upload An Image", type=['png', 'jpeg', 'jpg'])
    if image_file is not None:
        file_details = {"FileName": image_file.name, "FileType": image_file.type}
        st.write(file_details)
        # img = load_image(image_file)
        # st.image(img)
        with open(os.path.join(r"..\Cartoonify_Streamlit\Saved Image", image_file.name), "wb") as f:
            f.write(image_file.getbuffer())
            path = os.path.join(r"..\Cartoonify_Streamlit\Saved Image", image_file.name)
        st.success("Saved File")
        return path


def quantization(img, k):
    iD = np.float32(img).reshape((-1, 3))
    iC = (cv2.TermCriteria_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    ret, label, center = cv2.kmeans(iD, k, None, iC, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    iN = center[label.flatten()]
    iN = iN.reshape(img.shape)
    return iN


def img_edge(img, edge_width, blur):
    # convert color image to gray scale
    gC = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # covert gray scale image to blur image
    gB = cv2.medianBlur(gC, blur)

    # calculate and store the image edges
    iE = cv2.adaptiveThreshold(gB, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, edge_width, blur)
    return iE


def main():
    path = upload()
    image = read_image(path)
    edge_width = 9
    blur_value = 9
    totalColors = 10  # play with this number to achieve your desired output

    if image is not None:
        img_Edge = img_edge(image, edge_width, blur_value)
        image = quantization(image, totalColors)
        blurred = cv2.bilateralFilter(image, d=7, sigmaColor=200, sigmaSpace=200)
        cartoonify = cv2.bitwise_and(blurred, blurred, mask=img_Edge)
        st.image(cartoonify)



if __name__ == '__main__':
    main()
