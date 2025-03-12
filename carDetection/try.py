import cv2
from ultralytics import YOLO

#import model
model = YOLO("runs/detect/train/weights/best.pt")

# give a video path
path = "C:/Users/LENOVO/Desktop/carDetection/trafficjam.mp4"

# or you can your pc's camera (default=0) 
cap = cv2.VideoCapture(path)

# check is video opened
if not cap.isOpened():
    print("Exception, camera didn't open!")
    exit()

# Get vidoe frame and FPS
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Made a video file for output.
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

while cap.isOpened():
    success, frame = cap.read()
    if not success or frame is None:
        break 

    # Run the YOLO model and get the results
    results = model.predict(frame)

    # Visualize the results
    annotated_frame = results[0].plot() if isinstance(results, list) else results.plot()

    # Show the frame
    cv2.imshow("Car detection on traffic", annotated_frame)

    # Save the annotated frame
    out.write(annotated_frame)

    # Press the 'q' button for exit from window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
