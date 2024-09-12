import cv2
import mediapipe as mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Capture the video using the web cam.
cam = cv2.VideoCapture(0)

# Initialize the Hands model
with mp_hands.Hands(
    static_image_mode = False,
    model_complexity = 0,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence = 0.5) as hands:
    while cam.isOpened():
        success, image = cam.read()
        # If the image is not available, skip this iteration
        if not success:
            print("Ignoring empty camera frame.")
            continue
        image =cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        # convert the image from BGR to RGB which is required by MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        # process the image for hand detection and tracking 
        results = hands.process(image_rgb)
        image_rgb.flags.writeable = True

        # Convert the frame back from RGB to BGR (required by OpenCV)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # If hands are detected, draw landmarks and connections on the frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image_bgr, 
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
        
        # Display the image with annotations
        cv2.imshow("MediaPipe Hands", image_bgr)

        # Exit the loops if 'q' key is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cam.release()
    cv2.destroyAllWindows()

