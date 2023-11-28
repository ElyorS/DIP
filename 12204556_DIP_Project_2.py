#Saidmuorodov Elyor 12204556    DIP   Project 2
import cv2

def main():
    # Creating a CSRT tracker instance (Channel and Spatial Reliability Tracker)
    tracker = cv2.legacy.TrackerCSRT_create()

    # Defining the camera index
    CAMERA_INDEX = 0

    # Flag to indicate if the video is paused
    paused = False

    # Initializing the bounding box as None
    bbox = None

    # Opening a video file for reading
    #Example 1:
    video = cv2.VideoCapture("/Users/elyoretto/Downloads/People Walking Free Stock Footage, Royalty-Free No Copyright Content-2.mp4")

   
    # Alternatively, it is possible to use the camera index to capture video
    # video = cv2.VideoCapture(CAMERA_INDEX)

    # Checking if the video file or camera is opened successfully
    if not video.isOpened():
        print('Error: Cannot open camera or video source')
        return

    # Loop to process each frame in the video
    while True:
        # Reading a frame from the video
        ret, frame = video.read()

        # Checking if the frame is successfully reading
        if not ret:
            print('Error: Something went wrong while reading the video')
            break

        # Getting the height and width of the frame
        frame_height, frame_width = frame.shape[:2]

        # Resizing the frame 
        resized_frame = cv2.resize(frame, (frame_width // 2, frame_height // 2))

        # If not paused, perform object tracking
        if not paused:
            if bbox is not None:
                # Start timer to calculate FPS
                timer = cv2.getTickCount()

                # Update the tracker with the resized frame
                ret, bbox = tracker.update(resized_frame)

                # If tracking is successful, draw bounding box
                if ret:
                    bbox = tuple(map(int, bbox))  # Convert bbox values to integers
                    p1 = (bbox[0], bbox[1])
                    p2 = (bbox[0] + bbox[2], bbox[1] + bbox[3])
                    cv2.rectangle(resized_frame, p1, p2, (255, 0, 0), 2, 1)
                else:
                    # If tracking fails, display a message
                    cv2.putText(resized_frame, "Tracking failure detected", (100, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

                # Calculate and display FPS
                fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
                cv2.putText(resized_frame, "CSRT Tracker", (100, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
                cv2.putText(resized_frame, "FPS : " + str(int(fps)), (100, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

                # Display the frame with tracking information
                cv2.imshow("Tracking", resized_frame)

                # Press'Esc' key to exit
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
            else:
                # Display the frame without tracking if the bounding box is not defined
                cv2.imshow("Tracking", resized_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break

        # Check for user input to pause and reselect the object to track
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            paused = not paused
            if paused:
                # Allow the user to select a new bounding box
                bbox = cv2.selectROI("Tracking", resized_frame, False)
                if bbox != (0, 0, 0, 0):
                    # Initialize the tracker with the new bounding box
                    tracker = cv2.legacy.TrackerCSRT_create()
                    tracker.init(resized_frame, bbox)
            else:
                bbox = None

        # Check for user input to resume tracking after selecting ROI
        if (key == 13 or key == 32) and paused and bbox is not None:
            paused = not paused

    # Release the video source and close all windows
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()





