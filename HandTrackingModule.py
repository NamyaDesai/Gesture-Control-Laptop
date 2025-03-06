import cv2
import mediapipe as mp
import math

class HandTrackingModule:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        """
        Initializes the hand detector.
        :param mode: Whether to run in static mode.
        :param maxHands: Maximum number of hands to detect.
        :param detectionCon: Minimum detection confidence.
        :param trackCon: Minimum tracking confidence.
        """
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        """
        Detect hands in the given image and optionally draw landmarks.
        :param img: Input image.
        :param draw: Whether to draw hand landmarks on the image.
        :return: Processed image.
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        """
        Find positions of hand landmarks for a specific hand.
        :param img: Input image.
        :param handNo: Index of the hand to process (default is 0).
        :param draw: Whether to draw points on the image.
        :return: List of landmark positions (id, x, y).
        """
        lmList = []
        if self.results.multi_hand_landmarks:
            try:
                myHand = self.results.multi_hand_landmarks[handNo]
                h, w, c = img.shape
                for id, lm in enumerate(myHand.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            except IndexError:
                pass  # Handle cases where no hand is detected
        return lmList

    def calculateDistance(self, p1, p2, lmList):
        """
        Calculate the Euclidean distance between two landmarks.
        :param p1: ID of the first landmark.
        :param p2: ID of the second landmark.
        :param lmList: List of landmarks.
        :return: Distance between the two points.
        """
        x1, y1 = lmList[p1][1], lmList[p1][2]
        x2, y2 = lmList[p2][1], lmList[p2][2]
        distance = math.hypot(x2 - x1, y2 - y1)
        return distance

    def recognizeGesture(self, lmList):
        """
        Recognize gestures based on hand landmarks.
        :param lmList: List of landmarks.
        :return: Gesture name (string) or None if no gesture is recognized.
        """
        if len(lmList) == 0:
            return None

        # Example gestures: Thumb and index pinch
        thumb_tip = 4
        index_tip = 8
        middle_tip = 12
        distance_thumb_index = self.calculateDistance(thumb_tip, index_tip, lmList)

        # Detect pinch gesture
        if distance_thumb_index < 25:
            return "Pinch"

        # Add more gestures here based on landmark positions

        return None

# Example usage
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    detector = HandTrackingModule(detectionCon=0.8, trackCon=0.8)

    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)

        if len(lmList) != 0:
            gesture = detector.recognizeGesture(lmList)
            if gesture:
                print(f"Gesture Detected: {gesture}")

        cv2.imshow("Hand Tracking", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
