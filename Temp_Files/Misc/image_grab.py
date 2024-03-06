import cv2 as cv

def save_frame(video_path, save_frame):
    cap = cv.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    if save_frame < 0 or save_frame >= frame_count:
        print("Error: Invalid save_frame value. It should be in the range [0, {}]".format(frame_count - 1))
        cap.release()
        return

    # Set the position to the desired frame
    cap.set(cv.CAP_PROP_POS_FRAMES, save_frame)

    # Read the frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        cap.release()
        return

    # Save the frame as "screenshot.jpg"
    cv.imwrite("screenshot.jpg", frame)

    print("Screenshot saved as 'screenshot.jpg'.")

    # Release the video capture object
    cap.release()

if __name__ == "__main__":
    # Specify the path to the video file
    video_path = '../Videos/scenario_alt2.mp4'

    # Specify the frame number you want to save
    save_frame_number = 1200  # Change this to the desired frame number

    # Call the function to save the specified frame
    save_frame(video_path, save_frame_number)
