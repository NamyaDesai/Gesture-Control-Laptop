
import cv2
import mediapipe as mp
import time, math, numpy as np
import pyautogui
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import screen_brightness_control as sbc
import HandTrackingModule as htm

# Initialize variables
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

# Mediapipe hand detector
mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.75)
mpDraw = mp.solutions.drawing_utils

# Audio and volume setup
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()  # (-63.5, 0.0, 0.5) min max
minVol, maxVol = -63, volRange[1]
hmin, hmax = 50, 200

# PyAutoGUI settings
pyautogui.FAILSAFE = False

# Gesture recognition IDs
tipIds = [4, 8, 12, 16, 20]

# Variables for cursor smoothing
prev_cursorX, prev_cursorY = 0, 0
smooth_factor = 5  # Higher value for smoother, slower movement

while True:
    success, img = cap.read()
    if not success:
        print("Error: Failed to capture image.")
        break
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    left_lmList, right_lmList = [], []
    if results.multi_hand_landmarks and results.multi_handedness:
        for i, hand_handedness in enumerate(results.multi_handedness):
            label = hand_handedness.classification[0].label
            hand_landmarks = results.multi_hand_landmarks[i]
            h, w, _ = img.shape
            lmList = [[int(lm.x * w), int(lm.y * h)] for lm in hand_landmarks.landmark]
            if label == "Left":
                left_lmList = lmList
            elif label == "Right":
                right_lmList = lmList
            mpDraw.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS)

    # Brightness control with left hand (thumb and index finger)
    if left_lmList:
        x1, y1 = left_lmList[4]
        x2, y2 = left_lmList[8]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        length = math.hypot(x2 - x1, y2 - y1)
        bright = np.interp(length, [15, 200], [0, 100])
        sbc.set_brightness(int(bright))

    # Volume control with right hand (thumb and index finger)
    if right_lmList:
        x1, y1 = right_lmList[4]
        x2, y2 = right_lmList[8]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        length = math.hypot(x2 - x1, y2 - y1)
        vol = np.interp(length, [15, 200], [minVol, maxVol])
        volume.SetMasterVolumeLevel(vol, None)

    # Cursor control with right hand (thumb and index finger)
    if right_lmList and len(right_lmList) >= 9:
        x, y = right_lmList[8]
        w, h = pyautogui.size()

        # Smooth cursor movement
        cursorX = np.interp(x, [0, wCam], [0, w])
        cursorY = np.interp(y, [0, hCam], [0, h])
        cursorX = prev_cursorX + (cursorX - prev_cursorX) / smooth_factor
        cursorY = prev_cursorY + (cursorY - prev_cursorY) / smooth_factor
        pyautogui.moveTo(cursorX, cursorY)
        prev_cursorX, prev_cursorY = cursorX, cursorY

        # Click gesture: double pinch or half-folding of index finger
        distance_thumb_index = math.hypot(
            right_lmList[4][0] - right_lmList[8][0],
            right_lmList[4][1] - right_lmList[8][1]
        )
        distance_index_middle = math.hypot(
            right_lmList[8][0] - right_lmList[12][0],
            right_lmList[8][1] - right_lmList[12][1]
        )
        
        if distance_thumb_index < 25 and distance_index_middle < 25:
            pyautogui.click()

    # Scroll control: Thumb outwards for scroll up, index finger up for scroll down
    if right_lmList and len(right_lmList) >= 13:
        # Check if thumb is pointing outward (scroll up gesture)
        thumb_out = abs(right_lmList[4][0] - right_lmList[2][0]) > 50 and right_lmList[4][1] > right_lmList[3][1]

        # Check if index finger is straight (scroll down gesture)
        index_up = right_lmList[8][1] < right_lmList[6][1]

        if thumb_out:
            pyautogui.scroll(300)  # Scroll up when thumb is pointing outward
        elif index_up:
            pyautogui.scroll(-300)  # Scroll down when index finger is straight


    # FPS display
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f"FPS: {int(fps)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Gesture Control", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()