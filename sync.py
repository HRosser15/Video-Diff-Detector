import cv2 as cv
import numpy as np
import sys
import argparse

def find_frame_difference(gray1, frame2, threshold_find):
    # Convert frames to grayscale for simplicity
    gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)  

    # Find pixel-wise differences in the two frames
    diff = cv.absdiff(gray1, gray2)

    # Create a binary image, setting pixels above a threshold to white and below to black
    _, thresholded_diff = cv.threshold(diff, threshold_find, 255, cv.THRESH_BINARY)
    cv.imshow('thresholded_diff', thresholded_diff)
    cv.waitKey(1)

    # Count the number of non-black pixels in the thresholded_diff
    non_zero_count = np.count_nonzero(thresholded_diff)

    # Return true if more than "threshold_find" non-black pixels are found
    return non_zero_count > 0

def find_matching_frame(gray1, frame2, threshold_match):
    # Convert frames to grayscale for simplicity
    gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)  

    # Find pixel-wise differences in the two frames
    diff = cv.absdiff(gray1, gray2)

    # Create a binary image, setting pixels above a threshold to white and below to black
    _, thresholded_diff = cv.threshold(diff, threshold_match, 255, cv.THRESH_BINARY)
    cv.imshow('finding match', thresholded_diff)
    cv.waitKey(30)

    # Count the number of non-black pixels in the thresholded_diff
    non_zero_count = np.count_nonzero(thresholded_diff)

    # Return true if less than "threshold_match" non-black pixels are found
    return non_zero_count < threshold_match

def find_sync_frame_number(first_frame, base_vid, threshold_find):
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
        if find_frame_difference(first_frame, frame, threshold_find):
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

def find_matching_frame_number(sync_frame, alt_vid, threshold_match):
    frame_number = 0

    while True:
        # Read the next frame
        ret, frame = alt_vid.read()

        # If no frame is found, break the loop
        if not ret:
            break

        frame_number += 1

        # Return the frame number if a matching frame is found
        if find_matching_frame(sync_frame, frame, threshold_match):
            return frame_number

    return None

def set_sync_properties(base_vid, threshold_find, threshold_match):
    frame_width = int(base_vid.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(base_vid.get(cv.CAP_PROP_FRAME_HEIGHT))

    if threshold_find is None:
        if frame_width < 360:
            threshold_find = 100
        elif frame_width < 720:
            threshold_find = 200
        elif frame_width < 1080:
            threshold_find = 280
        else:
            threshold_find = 350

    if threshold_match is None:
        if frame_width < 360:
            threshold_match = 2
        elif frame_width < 720:
            threshold_match = 5
        elif frame_width < 1080:
            threshold_match = 10
        else:
            threshold_match = 15

    return threshold_find, threshold_match

def main():
    parser = argparse.ArgumentParser(description='Video Synchronization')
    parser.add_argument('video1_path', help='Path to the first video file')
    parser.add_argument('video2_path', help='Path to the second video file')
    parser.add_argument('-stf', '--sync-threshold-find', type=int, help='Set the threshold value finding a reference frame in the base video (default: based on video resolution).')
    parser.add_argument('-stm', '--sync-threshold-match', type=int, help='Set the threshold value finding a matching frame in the delta video (default: based on video resolution).')   
    args = parser.parse_args()

    video1_path = args.video1_path
    video2_path = args.video2_path
    threshold_find = args.sync_threshold_find
    threshold_match = args.sync_threshold_match

    base_video_path = 'Videos/'
    full_video1_path = base_video_path + video1_path
    full_video2_path = base_video_path + video2_path

    # Open video capture for base and alternative videos
    base_vid = cv.VideoCapture(full_video1_path)
    alt_vid = cv.VideoCapture(full_video2_path)

    if not (base_vid.isOpened() and alt_vid.isOpened()):
        print("Error: Unable to open videos.")
        return

    # Get and set frame rates to match for synchronization
    base_frame_rate = base_vid.get(cv.CAP_PROP_FPS)
    alt_vid.set(cv.CAP_PROP_FPS, base_frame_rate)

    # Set sync properties based on video resolution or user input
    threshold_find, threshold_match = set_sync_properties(base_vid, threshold_find, threshold_match)

    # Capture the first frame of the base video
    ret, first_frame = base_vid.read()

    # Convert it to grayscale for simplicity
    first_frame = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)  

    # Find frame numbers where videos are synchronized
    sync_frame_number = find_sync_frame_number(first_frame, base_vid, threshold_find)

    # Get the frame at the sync_frame_number
    sync_frame = get_frame_at_number(sync_frame_number, base_vid)

    # Convert it to grayscale for simplicity
    sync_frame = cv.cvtColor(sync_frame, cv.COLOR_BGR2GRAY)
    
    # Find the frame number in the alt video that contains the sync frame
    alt_sync_frame_number = find_matching_frame_number(sync_frame, alt_vid, threshold_match)

    # If no alt_sync_frame_number is found, switch the video paths and try again
    if alt_sync_frame_number is None:
        # print("No alt sync frame found, switching video paths...")
        base_vid.release()
        alt_vid.release()
        video1_path, video2_path = video2_path, video1_path
        base_vid = cv.VideoCapture(base_video_path + video1_path)
        alt_vid = cv.VideoCapture(base_video_path + video2_path)
        base_vid.set(cv.CAP_PROP_FPS, base_frame_rate)
        ret, first_frame = base_vid.read()
        first_frame = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
        sync_frame_number = find_sync_frame_number(first_frame, base_vid, threshold_find)
        sync_frame = get_frame_at_number(sync_frame_number, base_vid)
        sync_frame = cv.cvtColor(sync_frame, cv.COLOR_BGR2GRAY)
        alt_sync_frame_number = find_matching_frame_number(sync_frame, alt_vid, threshold_match)
        videos_switched = True
    else:
        videos_switched = False

    # If still no alt_sync frame is found, return None
    if alt_sync_frame_number is None:
        return None, None, None

    # Check if sync_frame_number is also None
    if sync_frame_number is None:
        return None, None, None

    # find the minimum frame number
    min_frame_number = min(sync_frame_number, alt_sync_frame_number)

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