import cv2 as cv
import numpy as np

# TO RUN:
# open terminal with "CTRL + `"
# enter 'python demo2_pixelwise_video_sync.py' and press enter
# press 'q' to exit the video window

def find_frame_difference(frame1, frame2):
    # Convert frames to grayscale for simplicity
    gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

    # Display the original frames
    cv.imshow('Frame 1', gray1)
    cv.imshow('Frame 2', gray2)
    key = cv.waitKey(0)

    # Save the original frames if 's' is pressed
    if key == ord('s'):
        cv.imwrite('frame1.png', gray1)
        cv.imwrite('frame2.png', gray2)

    # Find pixel-wise differences in the two frames
    diff = cv.absdiff(gray1, gray2)
    diff_resized = rescaleFrame(diff, 3)
    cv.imshow('Difference', diff_resized)
    key = cv.waitKey(0)

    # Save the difference image if 's' is pressed
    if key == ord('s'):
        cv.imwrite('difference.png', diff_resized)

    # Create a binary image, setting pixels above a threshold to white and below to black
    _, thresholded_diff = cv.threshold(diff, 125, 255, cv.THRESH_BINARY)
    thresholded_diff_resized = rescaleFrame(thresholded_diff, 3)
    cv.imshow('Thresholded Difference', thresholded_diff_resized)
    key = cv.waitKey(0)

    # Save the thresholded difference image if 's' is pressed
    if key == ord('s'):
        cv.imwrite('thresholded_difference.png', thresholded_diff_resized)

    # Count the number of non-black pixels in the thresholded_diff
    non_zero_count = np.count_nonzero(thresholded_diff)

    # Return true if more than "threshold" non-black pixels are found
    threshold = 200
    return non_zero_count > threshold

def find_matching_frame(frame1, frame2):
    # Convert frames to grayscale for simplicity
    gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)  

    # Find pixel-wise differences in the two frames
    diff = cv.absdiff(gray1, gray2)
    diff_resized = rescaleFrame(diff, 3)
    cv.imshow('alt diff', diff_resized)
    cv.waitKey(10)

    # Create a binary image, setting pixels above a threshold to white and below to black
    _, thresholded_diff = cv.threshold(diff, 10, 255, cv.THRESH_BINARY)
    thresholded_diff_resized = rescaleFrame(thresholded_diff, 3)
    cv.imshow('alt thresholded_diff', thresholded_diff_resized)
    cv.waitKey(10)

    # Count the number of non-black pixels in the thresholded_diff
    non_zero_count = np.count_nonzero(thresholded_diff)

    # Return true if less than "threshold" non-black pixels are found
    threshold = 20
    return non_zero_count < threshold

def find_sync_frame_number(first_frame, base_vid):
    # Reset video capture to the beginning
    base_vid.set(cv.CAP_PROP_POS_FRAMES, 0)

    frame_number = 0

    while True:
        ret, frame = base_vid.read()

        if not ret:
            break

        frame_number += 1

        # If a non-matching frame is found, print the frame number and return
        if find_frame_difference(first_frame, frame):
            # print(f"Non-matching frame found at frame number {frame_number}")
            return frame_number

    return None

def get_frame_at_number(frame_number, vid):
    # Set video capture to the specified frame number
    vid.set(cv.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = vid.read()
    
    if ret:
        return frame
    else:
        return None

def find_matching_frame_number(sync_frame, alt_vid):
    frame_number = 0

    while True:
        ret, frame = alt_vid.read()

        if not ret:
            break

        frame_number += 1

        # Return the frame number if a matching frame is found
        if find_matching_frame(sync_frame, frame):
            return frame_number

    return None

def rescaleFrame(frame, scale):
    # images, videos and live videos
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)  # tuple of width and height

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def main():
    # Open video capture for base and alternative videos
    base_vid = cv.VideoCapture('../Videos/scenario_base.mp4')
    alt_vid = cv.VideoCapture('../Videos/scenario_alt2.mp4')

    # Get and set frame rates to match for synchronization
    base_frame_rate = base_vid.get(cv.CAP_PROP_FPS)
    alt_vid.set(cv.CAP_PROP_FPS, base_frame_rate)

    # Capture the first frame of the base video
    ret, first_frame = base_vid.read()
    first_frame_resized = rescaleFrame(first_frame, 3)
    cv.imshow('first frame', first_frame_resized)
    cv.waitKey(0)
    

    # Find frame numbers where videos are synchronized
    sync_frame_number = find_sync_frame_number(first_frame, base_vid)
    sync_frame = get_frame_at_number(sync_frame_number, base_vid)
    print("Base video reference frame found at frame number: ", sync_frame_number)
    sync_frame_resized = rescaleFrame(sync_frame, 3)
    cv.imshow('reference frame', sync_frame_resized)
    cv.waitKey(0)

    
    alt_sync_frame_number = find_matching_frame_number(sync_frame, alt_vid)
    
    print("Alt video reference frame found at frame number: ", alt_sync_frame_number)

    # Set the minimum frame number for synchronization
    min_frame_number = min(sync_frame_number, alt_sync_frame_number)

    # Print frame numbers for reference
    print("Minumum frame number: ", min_frame_number)
    print("Starting base video from frame number: ", sync_frame_number - min_frame_number)
    print("Starting alt video from frame number: ", alt_sync_frame_number - min_frame_number)

    # Set starting frame numbers for both videos
    # base_vid.set(cv.CAP_PROP_POS_FRAMES, sync_frame_number - min_frame_number)
    # alt_vid.set(cv.CAP_PROP_POS_FRAMES, alt_sync_frame_number - min_frame_number)
    base_vid.set(cv.CAP_PROP_POS_FRAMES, 288)
    alt_vid.set(cv.CAP_PROP_POS_FRAMES, 285)

    while True:
        ret_base, frame_base = base_vid.read()
        ret_alt, frame_alt = alt_vid.read()

        if not ret_base or not ret_alt:
            break

        # Add a border around each video frame
        frame_base = cv.copyMakeBorder(frame_base, 10, 50, 10, 10, cv.BORDER_CONSTANT, value=(255, 255, 255))
        frame_alt = cv.copyMakeBorder(frame_alt, 10, 50, 10, 10, cv.BORDER_CONSTANT, value=(255, 255, 255))

        # Add text below each video to indicate which is which
        cv.putText(frame_base, "Video 1", (100, frame_base.shape[0] - 20), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)
        cv.putText(frame_alt, "Video 2", (100, frame_alt.shape[0] - 20), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)

        # Create a video by concatenating frames of both videos
        concatenated_frame = np.concatenate((frame_base, frame_alt), axis=1)
        concatenated_frame_resized = rescaleFrame(concatenated_frame, 3)

        cv.imshow('Synchronized Videos', concatenated_frame_resized)

        # Break the loop if 'q' is pressed
        if cv.waitKey(30) & 0xFF == ord('q'):
            break

    # Release video captures and close windows
    base_vid.release()
    alt_vid.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
