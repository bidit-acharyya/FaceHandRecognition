import cv2
import mediapipe as mp

vid = cv2.VideoCapture(0)

while True:
    ret, frame = vid.read()
    if frame is None:
        break

    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    tribute = False

    with mp_pose.Pose() as pose:
        result = pose.process(frame)

    with mp_hands.Hands(model_complexity=0,min_detection_confidence=0.15,min_tracking_confidence=0.25) as hands:
        result_hands = hands.process(frame)

    red_dot = mp_draw.DrawingSpec(color=(0, 0, 255), thickness=-1, circle_radius=2)
    green_line = mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)

    if result.pose_landmarks:
        right_hand_landmarks = result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y
        left_hand_landmarks = result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y
        right_eye_landmarks = result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE].y
        left_eye_landmarks = result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE].y

        if(right_hand_landmarks < right_eye_landmarks and right_hand_landmarks < left_eye_landmarks):
            if tribute:
                cv2.putText(frame, "Tribute", (150,150), cv2.FONT_HERSHEY_PLAIN, 10, (0,0,0), 5, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Hand is raised", (150,150), cv2.FONT_HERSHEY_PLAIN, 10, (0,0,0), 5, cv2.LINE_AA)

    mp_draw.draw_landmarks(image=frame,
                           landmark_list=result.pose_landmarks,
                           connections=mp_pose.POSE_CONNECTIONS,
                           landmark_drawing_spec=red_dot,
                           connection_drawing_spec=green_line)
    if result_hands.multi_hand_landmarks:
        for hand_landmark in result_hands.multi_hand_landmarks:
            if hand_landmark.landmark[mp_hands.HandLandmark.THUMB_TIP] and hand_landmark.landmark[mp_hands.HandLandmark.PINKY_TIP]:
                thumb_tip = hand_landmark.landmark[mp_hands.HandLandmark.THUMB_TIP]
                pinky_tip = hand_landmark.landmark[mp_hands.HandLandmark.PINKY_TIP]
                print("thumb = " + str(thumb_tip) + ", pinky = " + str(pinky_tip))
                if abs(thumb_tip.x - pinky_tip.x) < 0.002:
                    tribute = True
            mp_draw.draw_landmarks(
                image=frame,
                landmark_list=hand_landmark,
                connections=mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=red_dot,
                connection_drawing_spec=green_line)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()