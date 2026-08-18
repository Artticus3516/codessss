import cv2 
import mediapipe as mp
import time 

def callcamera():
    baseoptions = mp.tasks.BaseOptions(model_asset_path="pose_landmarker_lite.task")
    runningMode = mp.tasks.vision.RunningMode

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=baseoptions,
        running_mode=runningMode.VIDEO,
        num_poses=5,
        min_pose_detection_confidence=0.2,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                print("Camera could not be opened")
                break

            h, w, _ = frame.shape
            clr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=clr_frame)
            frame_timestamp_ms = int(time.time() * 1000)

            result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            if result.pose_landmarks:
                for pose_landmarks in result.pose_landmarks:
                    x_coords = [lm.x * w for lm in pose_landmarks]
                    y_coords = [lm.y * h for lm in pose_landmarks]

                    x_min, x_max = int(min(x_coords)), int(max(x_coords))
                    y_min, y_max = int(min(y_coords)), int(max(y_coords))

                    # Draw green bounding box around each detected person
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            cv2.imshow("Pose Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    callcamera()
