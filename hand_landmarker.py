import os
os.environ['TF_ENBLE_ONEDNN_OPTS'] = '0'

import cv2
import mediapipe as mp

mp_hand_landmark = mp.solutions.hands
hands = mp_hand_landmark.Hands()

cap = cv2.VideoCapture('hands_ip.mp4')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

output_filename = "hands_landmark_detection_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mprv')
out = cv2.VideoWriter(output_filename , fourcc ,fps ,(frame_width,frame_height))

while(True):
    ret,frame =cap.read()
    if not ret :
        break

    rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)


    if results.multi_hand_landmarks:
        # Iterate through each hand detected
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw custom circles on hand landmarks
            for idx, landmark in enumerate(hand_landmarks.landmark):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Green circles for landmarks

            # Draw custom lines between connected landmarks
            for connection in mp_hand_landmark.HAND_CONNECTIONS:
                start_idx, end_idx = connection
                start_point = (int(hand_landmarks.landmark[start_idx].x * frame_width),
                                int(hand_landmarks.landmark[start_idx].y * frame_height))
                end_point = (int(hand_landmarks.landmark[end_idx].x * frame_width),
                                int(hand_landmarks.landmark[end_idx].y * frame_height))
                cv2.line(frame, start_point, end_point, (255, 0, 0), 2)  # Blue lines for connections

    #Show the Frame 
    cv2.imshow("Custom Hand Landmarks ", frame)
    out.write(frame)

    cv2.imwrite("output.jpg" ,frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break


cap.release()
out.release()
cv2.destroyAllWindows()
print("Finisted Processing Frames !")

