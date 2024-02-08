import cv2 as cv
import numpy as np

# TO RUN:
# open terminal with "CTRL + `"
# enter 'python pixel_wise_video_sync.py' and press enter
# press 'q' to exit the video window

def find_frame_difference(frame1, frame2):
    # Convert frames to grayscale for simplicity
    gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)  

    # Find pixel-wise differences in the two frames
    diff = cv.absdiff(gray1, gray2)

    # Create a binary image, setting pixels above a threshold to white and below to black
    _, thresholded_diff = cv.threshold(diff, 25, 255, cv.THRESH_BINARY)

    # Count the number of non-black pixels in the thresholded_diff
    non_zero_count = np.count_nonzero(thresholded_diff)

    # Return true if more than 100 non-black pixels are found
    return non_zero_count > 200

def find_matching_frame(frame1, frame2):
    # Convert frames to grayscale for simplicity
    gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)  

    # Find pixel-wise differences in the two frames
    diff = cv.absdiff(gray1, gray2)

    # Create a binary image, setting pixels above a threshold to white and below to black
    _, thresholded_diff = cv.threshold(diff, 25, 255, cv.THRESH_BINARY)

    # Count the number of non-black pixels in the thresholded_diff
    non_zero_count = np.count_nonzero(thresholded_diff)

    # Return true if more than 200 non-black pixels are found
    return non_zero_count > 10

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
        if not find_matching_frame(sync_frame, frame):
            return frame_number

    return None

def main():
    # Open video capture for base and alternative videos
    base_vid = cv.VideoCapture('Videos/scenario_base.mp4')
    alt_vid = cv.VideoCapture('Videos/scenario_alt2.mp4')

    # Get and set frame rates to match for synchronization
    base_frame_rate = base_vid.get(cv.CAP_PROP_FPS)
    alt_vid.set(cv.CAP_PROP_FPS, base_frame_rate)

    # Capture the first frame of the base video
    ret, first_frame = base_vid.read()

    # Find frame numbers where videos are synchronized
    sync_frame_number = find_sync_frame_number(first_frame, base_vid)
    sync_frame = get_frame_at_number(sync_frame_number, base_vid)

    alt_sync_frame_number = find_matching_frame_number(sync_frame, alt_vid)
    
    
    # Release video captures and close windows
    base_vid.release()
    alt_vid.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
