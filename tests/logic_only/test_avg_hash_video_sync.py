import cv2 as cv
import numpy as np


def average_hash(image, hash_size=8):
    resized = cv.resize(image, (hash_size, hash_size), interpolation=cv.INTER_AREA)
    gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
    average = gray.mean()
    hash_value = (gray > average).astype(np.uint8)
    return hash_value.flatten()

def find_frame_difference(frame1, frame2):
    hash1 = average_hash(frame1)
    hash2 = average_hash(frame2)

    hamming_distance = np.sum(hash1 != hash2)

    if hamming_distance > 0:
        return True
    else:
        return False


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
        if find_frame_difference(first_frame, frame):
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
        if not find_frame_difference(sync_frame, frame):
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
    print("min frame number: ", min_frame_number)
    print("base_frame: ", sync_frame_number)
    print("alt frame: ", alt_sync_frame_number)

    # Set the starting frame number for both videos
    base_vid.set(cv.CAP_PROP_POS_FRAMES, sync_frame_number - min_frame_number)
    alt_vid.set(cv.CAP_PROP_POS_FRAMES, alt_sync_frame_number - min_frame_number)


if __name__ == '__main__':
    main()