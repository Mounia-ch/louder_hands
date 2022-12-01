import streamlit as st
import cv2
import mediapipe as mp
from only_hands import handTracker
from tensorflow import models
from google.colab import drive

model 

#App layout
col1, col2, col3 = st.columns(3)

#Header with 3 columns to center image
with col1:
    st.write(' ')

with col2:
    st.image('/Users/manuel/Pictures/only_hands_logo.png')

with col3:
    st.write(' ')

#Main body with 2 columns to split webcam and prediction
col4, col5 = st.columns(2)

#Left column to show webcam
with col4:
    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)
    tracker = handTracker()
    p = st.empty()
    while run:
        _, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = tracker.handsFinder(frame)
        lmList = tracker.positionFinder(frame)
        FRAME_WINDOW.image(frame)
        #print(lmList)
        if len(lmList)>0:
            p.write(str(lmList))
    #st.camera_input('Webcam', label_visibility='hidden')

#Right column to show prediction
with col5:
    with st.container():
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.markdown("<h1 style='text-align: center; color: grey; vertical-align:middle;'>Translated letter:</h1>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; color: grey; vertical-align:middle;'>B</h2>", unsafe_allow_html=True)
        
        
        