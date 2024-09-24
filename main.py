import cvzone
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import google.generativeai as genai
from PIL import Image
import streamlit as st
 
st.set_page_config(layout="wide")

# Renaming column variables and checkbox
colA, colB = st.columns([3,2])
with colA:
    toggle_run = st.checkbox('Run', value=True)
    IMAGE_PLACEHOLDER = st.image([])

with colB:
    st.title("Answer")
    # Changed from subheader to markdown for better text wrapping
    answer_text = st.markdown("")

# Renaming the generative AI configuration and model
genai.configure(api_key="AIzaSyCC5ScKkeproeZeshDo9w74iOH-tK5d7bw")
ai_model = genai.GenerativeModel('gemini-1.5-flash')

# Renaming webcam initialization
webcam = cv2.VideoCapture(0)
webcam.set(3, 1280)
webcam.set(4, 720)

# Renaming the HandDetector class
hand_finder = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

# Renaming the function to get hand information
def extract_hand_data(frame):
    hand_list, frame = hand_finder.findHands(frame, draw=False, flipType=True)
    if hand_list:
        first_hand = hand_list[0]
        landmarks = first_hand["lmList"]
        fingers_state = hand_finder.fingersUp(first_hand)
        return fingers_state, landmarks
    else:
        return None

# Renaming the drawing function
def draw_on_canvas(hand_data, previous_position, blank_canvas):
    fingers_state, landmarks = hand_data
    current_position = None
    if fingers_state == [0, 1, 0, 0, 0]:
        current_position = landmarks[8][0:2]
        if previous_position is None:
            previous_position = current_position
        cv2.line(blank_canvas, current_position, previous_position, (255, 0, 255), 10)
    elif fingers_state == [1, 0, 0, 0, 0]:
        blank_canvas = np.zeros_like(frame)
    return current_position, blank_canvas

# Renaming the AI integration function
def generate_ai_response(ai_model, blank_canvas, fingers_state):
    if fingers_state == [1, 1, 1, 1, 0]:
        img_for_ai = Image.fromarray(blank_canvas)
        ai_result = ai_model.generate_content(["Solve this math problem", img_for_ai])
        return ai_result.text

# Renaming main variables
previous_position = None
drawing_canvas = None
combined_image = None
response_text = ""

# Main loop
while True:
    if toggle_run:
        # Capture frame
        successful_capture, frame = webcam.read()
        frame = cv2.flip(frame, 1)

        if drawing_canvas is None:
            drawing_canvas = np.zeros_like(frame)

        hand_data = extract_hand_data(frame)
        if hand_data:
            fingers_state, landmarks = hand_data
            previous_position, drawing_canvas = draw_on_canvas(hand_data, previous_position, drawing_canvas)
            response_text = generate_ai_response(ai_model, drawing_canvas, fingers_state)

        combined_image = cv2.addWeighted(frame, 0.7, drawing_canvas, 0.3, 0)
        IMAGE_PLACEHOLDER.image(combined_image, channels="BGR")

        if response_text:
            answer_text.markdown(response_text)

    else:
        st.warning("Webcam is paused. Check 'Run' to start.")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
webcam.release()
cv2.destroyAllWindows()
