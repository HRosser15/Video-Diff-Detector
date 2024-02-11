import cv2 as cv
import numpy as np

# TO RUN:
# open terminal with "CTRL + `"
# enter 'python pixel_wise_video_sync.py' and press enter
# press 'q' to exit the video window

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
    # Open video capture for base and alternative videos
    base_vid = cv.VideoCapture('../../Videos/scenario_base.mp4')
    alt_vid = cv.VideoCapture('../../Videos/scenario_alt2.mp4')

    # Get and set frame rates to match for synchronization
    base_frame_rate = base_vid.get(cv.CAP_PROP_FPS)
    alt_vid.set(cv.CAP_PROP_FPS, base_frame_rate)

    # Capture the first frame of the base video
    ret, first_frame = base_vid.read()

    # Find frame numbers where videos are synchronized
    sync_frame_number = find_sync_frame(first_frame, base_vid)
    sync_frame = get_frame_at_number(sync_frame_number, base_vid)
    # cv.imshow('sync frame', sync_frame)

    alt_sync_frame_number = find_matching_frame_number(sync_frame, alt_vid)

    # Set the minimum frame number for synchronization
    min_frame_number = min(sync_frame_number, alt_sync_frame_number)

    # Print frame numbers for reference
    # print("min frame number: ", min_frame_number)
    # print("base_frame: ", sync_frame_number)
    # print("alt frame: ", alt_sync_frame_number)

    # Set starting frame numbers for both videos
    base_vid.set(cv.CAP_PROP_POS_FRAMES, sync_frame_number - min_frame_number)
    alt_vid.set(cv.CAP_PROP_POS_FRAMES, alt_sync_frame_number - min_frame_number)

    while True:
        ret_base, frame_base = base_vid.read()
        ret_alt, frame_alt = alt_vid.read()

        if not ret_base or not ret_alt:
            break

        # Add text to both videos to indicate which is which
        cv.putText(frame_base, "Base Video", (10, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (255, 255, 255), 1)
        cv.putText(frame_alt, "Alt Video", (10, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (255, 255, 255), 1)

        # Create a video by concatenating frames of both videos
        concatenated_frame = np.concatenate((frame_base, frame_alt), axis=1)

        cv.imshow('Synchronized Videos', concatenated_frame)

        # Break the loop if 'q' is pressed
        if cv.waitKey(30) & 0xFF == ord('q'):
            break

    # Release video captures and close windows
    base_vid.release()
    alt_vid.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()


