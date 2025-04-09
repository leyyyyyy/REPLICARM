import cv2
import mediapipe as mp
import math
from cvfpscalc import CvFpsCalc
import controller1


def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def map_value(value, from_low, from_high, to_low, to_high):
    mapped = (value - from_low) * (to_high - to_low) / (from_high - from_low) + to_low
    mapped = max(to_low, min(to_high, mapped))  # Ensure value stays within bounds
    return round(mapped / 10) * 10  # Round to nearest multiple of 10


def main():
    mp_draw = mp.solutions.drawing_utils
    mp_hand = mp.solutions.hands
    hands = mp_hand.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5, max_num_hands=1)
    video = cv2.VideoCapture(0)
    cvFpsCalc = CvFpsCalc(buffer_len=20)
    
    while True:
        fps = cvFpsCalc.get()
        ret, image = video.read()
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            hand_landmark = results.multi_hand_landmarks[0]  # Only process one hand
            lmList = [[id, int(lm.x * image.shape[1]), int(lm.y * image.shape[0])] for id, lm in enumerate(hand_landmark.landmark)]
            
            mp_draw.draw_landmarks(image, hand_landmark, mp_hand.HAND_CONNECTIONS)
            
            if lmList:
                thumb_tip = lmList[4]
                middle_finger_tip = lmList[12]
                wrist = lmList[0]
                index_mcp = lmList[5]
                
                # Use wrist to index MCP as a reference for normalization
                reference_distance = calculate_distance((wrist[1], wrist[2]), (index_mcp[1], index_mcp[2]))
                
                if reference_distance > 0:
                    normalized_distance = calculate_distance((thumb_tip[1], thumb_tip[2]), (middle_finger_tip[1], middle_finger_tip[2])) / reference_distance
                    mapped_distance = map_value(normalized_distance, 0.1, 1.5, 0, 100)
                else:
                    mapped_distance = 0
                
                # Draw a green line between the thumb tip and middle finger tip
                cv2.line(image, (thumb_tip[1], thumb_tip[2]), (middle_finger_tip[1], middle_finger_tip[2]), (0, 255, 0), 2)
                
                cv2.putText(image, f"Angle: {mapped_distance}", (10, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (100, 255, 100), 2, cv2.LINE_AA)
                
                # Send mapped distance to controller
                controller1.set_servo_angle(mapped_distance)
        
        cv2.putText(image, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 100, 100), 2, cv2.LINE_AA)
        cv2.imshow("Frame", image)
        if cv2.waitKey(1) == 27:
            break
    
    video.release()
    cv2.destroyAllWindows()
    controller1.cleanup()


if __name__ == "__main__":
    main()