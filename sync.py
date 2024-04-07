import cv2 as cv
import numpy as np
import sys

def find_frame_difference(gray1, frame2):
    # Convert frames to grayscale for simplicity
    gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)  

    # Find pixel-wise differences in the two frames
    diff = cv.absdiff(gray1, gray2)

    # Create a binary image, setting pixels above a threshold to white and below to black
    _, thresholded_diff = cv.threshold(diff, 50, 255, cv.THRESH_BINARY)

    # Count the number of non-black pixels in the thresholded_diff
    non_zero_count = np.count_nonzero(thresholded_diff)

    # Return true if more than "threshold" non-black pixels are found
    threshold = 50
    return non_zero_count > threshold

def find_matching_frame(gray1, frame2):
    # Convert frames to grayscale for simplicity
    gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)  

    # Find pixel-wise differences in the two frames
    diff = cv.absdiff(gray1, gray2)

    # Create a binary image, setting pixels above a threshold to white and below to black
    _, thresholded_diff = cv.threshold(diff, 10, 255, cv.THRESH_BINARY)

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
        # Read the next frame
        ret, frame = base_vid.read()

        # If no frame is found, break the loop
        if not ret:
            break

        frame_number += 1

        # If a non-matching frame is found, return the current frame number
        if find_frame_difference(first_frame, frame):
            return frame_number

    return None

def get_frame_at_number(frame_number, vid):
    # Set video capture to the specified frame number
    vid.set(cv.CAP_PROP_POS_FRAMES, frame_number)

    # Read the frame at the specified frame number
    ret, frame = vid.read()
    
    # Return the frame if found, otherwise return None
    if ret:
        return frame
    else:
        return None

def find_matching_frame_number(sync_frame, alt_vid):
    frame_number = 0

    while True:
        # Read the next frame
        ret, frame = alt_vid.read()

        # If no frame is found, break the loop
        if not ret:
            break

        frame_number += 1

        # Return the frame number if a matching frame is found
        if find_matching_frame(sync_frame, frame):
            return frame_number

    return None

def main():
    if len(sys.argv) < 3:
        print("Usage: python video_sync.py <video1_path> <video2_path>")
        return
    
    # Set resolution of the video
    width, height = 1920, 1080

    base_vido_path = 'Videos/'
    video1_path = sys.argv[1]
    video2_path = sys.argv[2]
    
    # Open video capture for base and alternative videos
    base_vid = cv.VideoCapture(base_vido_path + video1_path)
    alt_vid = cv.VideoCapture(base_vido_path + video2_path)

    # Get and set frame rates to match for synchronization
    base_frame_rate = base_vid.get(cv.CAP_PROP_FPS)
    alt_vid.set(cv.CAP_PROP_FPS, base_frame_rate)

    # Set resolution of the video
    base_vid.set(cv.CAP_PROP_FRAME_WIDTH, width)
    base_vid.set(cv.CAP_PROP_FRAME_HEIGHT, height)

    # Capture the first frame of the base video
    ret, first_frame = base_vid.read()

    # Convert it to grayscale for simplicity
    first_frame = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)  
    
    # Find frame numbers where videos are synchronized
    sync_frame_number = find_sync_frame_number(first_frame, base_vid)

    # Get the frame at the sync_frame_number
    sync_frame = get_frame_at_number(sync_frame_number, base_vid)
    # print("Base video reference frame found at frame number: ", sync_frame_number)

    # Convert it to grayscale for simplicity
    sync_frame = cv.cvtColor(sync_frame, cv.COLOR_BGR2GRAY)
    
    # Find the frame number in the alt video that contains the sync frame
    alt_sync_frame_number = find_matching_frame_number(sync_frame, alt_vid)
    # print("Alt video reference frame found at frame number: ", alt_sync_frame_number)

    # If no alt_sync_frame_number is found, switch the video paths and try again
    if alt_sync_frame_number is None:
        print("No alt sync frame found, switching video paths...")
        base_vid.release()
        alt_vid.release()
        video1_path, video2_path = video2_path, video1_path
        base_vid = cv.VideoCapture(base_vido_path + video1_path)
        alt_vid = cv.VideoCapture(base_vido_path + video2_path)
        base_vid.set(cv.CAP_PROP_FPS, base_frame_rate)
        base_vid.set(cv.CAP_PROP_FRAME_WIDTH, width)
        base_vid.set(cv.CAP_PROP_FRAME_HEIGHT, height)
        ret, first_frame = base_vid.read()
        first_frame = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
        sync_frame_number = find_sync_frame_number(first_frame, base_vid)
        sync_frame = get_frame_at_number(sync_frame_number, base_vid)
        sync_frame = cv.cvtColor(sync_frame, cv.COLOR_BGR2GRAY)
        alt_sync_frame_number = find_matching_frame_number(sync_frame, alt_vid)
        videos_switched = True
    else:
        videos_switched = False

    # If still no alt_sync frame is found, return None
    if alt_sync_frame_number is None:
        print("Videos could not be synced. No matching frame was found.")
        return None, None, None

    # find the minimum frame number
    min_frame_number = min(sync_frame_number, alt_sync_frame_number)

        # Print frame numbers for reference
    # print("Starting base video from frame number: ", sync_frame_number - min_frame_number)
    # print("Starting alt video from frame number: ", alt_sync_frame_number - min_frame_number)
    base_start_frame = sync_frame_number - min_frame_number
    alt_start_frame = alt_sync_frame_number - min_frame_number

    # Set starting frame numbers for both videos
    base_vid.set(cv.CAP_PROP_POS_FRAMES, base_start_frame)
    alt_vid.set(cv.CAP_PROP_POS_FRAMES, alt_start_frame)

    return base_start_frame, alt_start_frame, videos_switched

if __name__ == '__main__':
    base_start_frame, alt_start_frame, videos_switched = main()
    # Format the output as a comma-separated string
    output = f"{base_start_frame},{alt_start_frame},{videos_switched}"
    print(output)