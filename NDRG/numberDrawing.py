import cv2
import mediapipe as mp

cam = cv2.VideoCapture(0)

# while True:
#     ret, frame = cam.read()
#         # out.write(frame)
#     if not ret:
#        break

#     frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)  # Rotate 90 degrees clockwise
 
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
# help(mp_hands.Hands)

with mp_hands.Hands(
    static_image_mode = False,
    model_complexity = 0,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence = 0.5) as hands:
    while cam.isOpened():
        success, image = cam.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        image =cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = hands.process(image_rgb)
        image_rgb.flags.writeable = True
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image_bgr, 
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
            
        cv2.imshow("MediaPipe Hands", image_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()

