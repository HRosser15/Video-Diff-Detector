import cv2 as cv
import numpy as np


def subtract_frames(frame1, frame2):
    # Convert frames to grayscale for simplicity
    gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)  

    # Perform simple frame subtraction
    diff = cv.subtract(gray1, gray2)

    # Set pixels below a threshold to black
    _, thresholded_diff = cv.threshold(diff, 25, 255, cv.THRESH_BINARY)

    # Count the number of non-black pixels in the thresholded_diff
    non_zero_count = np.count_nonzero(thresholded_diff)

    # Return true if more than 500 non-black pixels are found
    return non_zero_count > 500

def find_sync_frame(first_frame, base_vid):
    # Reset video capture to the beginning
    base_vid.set(cv.CAP_PROP_POS_FRAMES, 0)

    frame_number = 0

    while True:
        ret, frame = base_vid.read()

        if not ret:
            break

        frame_number += 1

        # If a non-matching frame is found, print the frame number and return
        if subtract_frames(first_frame, frame):
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
        if not subtract_frames(sync_frame, frame):
            return frame_number

    return None

def main():

    base_vid = cv.VideoCapture('../../Videos/scenario_base.mp4')  # instance of base video
    alt_vid = cv.VideoCapture('../../Videos/scenario_alt2.mp4')  # instance of alt video

    # Get the frame rate of the base video
    base_frame_rate = base_vid.get(cv.CAP_PROP_FPS)
    # Set the frame rate of alt_vid2 to match base_vid
    alt_vid.set(cv.CAP_PROP_FPS, base_frame_rate)

    # Capture image of first frame of base_vid
        # ret - (return) boolean that indicates if the .read operation was successful.
    ret, first_frame = base_vid.read()

    sync_frame_number = find_sync_frame(first_frame, base_vid)
    sync_frame = get_frame_at_number(sync_frame_number, base_vid)

    alt_sync_frame_number = find_matching_frame_number(sync_frame, alt_vid)

    min_frame_number = min(sync_frame_number, alt_sync_frame_number)
    # print("min frame number: ", min_frame_number)
    # print("base_frame: ", sync_frame_number)
    # print("alt frame: ", alt_sync_frame_number)

    # Set the starting frame number for both videos
    base_vid.set(cv.CAP_PROP_POS_FRAMES, sync_frame_number - min_frame_number)
    alt_vid.set(cv.CAP_PROP_POS_FRAMES, alt_sync_frame_number - min_frame_number)


if __name__ == '__main__':
    main()