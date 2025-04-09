import cv2
import mediapipe as mp
import math
import controller
import time  # For delay control
from cvfpscalc import CvFpsCalc

# Utility functions
def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def map_value(value, from_low, from_high, to_low, to_high):
    mapped = (value - from_low) * (to_high - to_low) / (from_high - from_low) + to_low
    return max(to_low, min(to_high, mapped))  # Ensure value stays within bounds

# MediaPipe Initialization
mp_draw = mp.solutions.drawing_utils
mp_hand = mp.solutions.hands
hands = mp_hand.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5, max_num_hands=1)
video = cv2.VideoCapture(0)

FRAME_CENTER_X = int(video.get(3) / 2)
FRAME_CENTER_Y = int(video.get(4) / 2)

# Servo mappings
SERVO_LATERAL = 1
SERVO_VERTICAL_1 = 2
SERVO_VERTICAL_2 = 3
SERVO_VERTICAL_3 = 4
SERVO_CLAW = 6  # Pin 11 for claw control

# Initial servo positions
servo_angles = {
    SERVO_LATERAL: 0,
    SERVO_VERTICAL_1: 0,
    SERVO_VERTICAL_2: 0,
    4: 0,
    SERVO_VERTICAL_3: 0,
}

# Gradual Calibration to 90° at Startup
calibration_speed = 1
while any(angle != 90 for angle in servo_angles.values()):
    for servo in servo_angles:
        if servo_angles[servo] < 90:
            servo_angles[servo] += calibration_speed
        elif servo_angles[servo] > 90:
            servo_angles[servo] -= calibration_speed

        controller.set_servo_angle(servo, servo_angles[servo])

    time.sleep(0.02)

# Sensitivity controls
sensitivity = 180
cvFpsCalc = CvFpsCalc(buffer_len=20)

while True:
    fps = cvFpsCalc.get()
    ret, image = video.read()
    if not ret:
        break

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # Draw center point
    cv2.circle(image, (FRAME_CENTER_X, FRAME_CENTER_Y), 5, (0, 0, 255), -1)

    if results.multi_hand_landmarks:
        hand_landmark = results.multi_hand_landmarks[0]
        lmList = [[id, int(lm.x * image.shape[1]), int(lm.y * image.shape[0])] for id, lm in enumerate(hand_landmark.landmark)]

        mp_draw.draw_landmarks(image, hand_landmark, mp_hand.HAND_CONNECTIONS)

        if lmList:
            # Code 1's tracking logic
            hand_x, hand_y = lmList[9][1], lmList[9][2]
            lateral_offset = (FRAME_CENTER_X - hand_x) / sensitivity
            vertical_offset = (hand_y - FRAME_CENTER_Y) / sensitivity

            servo_angles[SERVO_LATERAL] -= lateral_offset
            servo_angles[SERVO_VERTICAL_1] -= vertical_offset
            servo_angles[SERVO_VERTICAL_2] += vertical_offset
            servo_angles[SERVO_VERTICAL_3] -= vertical_offset

            # Code 2's tracking logic (Claw control)
            thumb_tip = lmList[4]
            middle_finger_tip = lmList[12]
            wrist = lmList[0]
            index_mcp = lmList[5]
            center_Hand = lmList[9]
            center_HandYAxis = center_Hand[2]

            center_HandMap = round(map_value(center_HandYAxis, 450, 50, 0, 100))
            print(center_HandYAxis)
            cv2.putText(image, f"Y Axis: {center_HandMap}", (10, 150), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 200, 200), 2, cv2.LINE_AA)
            controller.set_servo_angle(SERVO_VERTICAL_1, center_HandMap)
            
            reference_distance = calculate_distance((wrist[1], wrist[2]), (index_mcp[1], index_mcp[2]))
            distanceMapped = round(map_value(reference_distance, 150, 100, 90, 180), 2)
            controller.set_servo_angle(SERVO_VERTICAL_2, distanceMapped)
            cv2.putText(image, f"Distance: {distanceMapped}", (10, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 200, 200), 2, cv2.LINE_AA)



            
            if reference_distance > 0:
                normalized_distance = calculate_distance((thumb_tip[1], thumb_tip[2]), (middle_finger_tip[1], middle_finger_tip[2])) / reference_distance
                mapped_distance = map_value(normalized_distance, 0.1, 1.5, 0, 100)
            else:
                mapped_distance = 0

            cv2.line(image, (thumb_tip[1], thumb_tip[2]), (middle_finger_tip[1], middle_finger_tip[2]), (0, 255, 0), 2)
            cv2.line(image, (wrist[1], wrist[2]), (index_mcp[1], index_mcp[2]), (0, 255, 0), 2)
            cv2.putText(image, f"Angle: {mapped_distance}", (10, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (100, 255, 100), 2, cv2.LINE_AA)
            controller.set_servo_angle(SERVO_CLAW, mapped_distance)

    # Clamp angles between 0 and 180
    # for servo in servo_angles:
    #     servo_angles[servo] = max(0, min(180, servo_angles[servo]))
    #     controller.set_servo_angle(servo, servo_angles[servo])

    # Display debug info
    cv2.putText(image, f"Lateral: {servo_angles[SERVO_LATERAL]:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(image, f"FPS: {fps}", (10, 90), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 100, 100), 2, cv2.LINE_AA)
    

    cv2.imshow("Frame", image)
    if cv2.waitKey(1) == 27:
        break

video.release()
cv2.destroyAllWindows()
controller.cleanup()
