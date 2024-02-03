import cv2 as cv
import numpy as np
from PIL import Image
import imagehash

# TO RUN:
# open terminal with "CTRL + `"
# enter 'python dct_hash_video_sync.py' and press enter
# press 'q' to exit the video window

def dct_hash(image, hash_size=8):
    resized = cv.resize(image, (hash_size, hash_size), interpolation=cv.INTER_AREA)
    gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)

    # Compute DCT
    dct_result = cv.dct(np.float32(gray))

    # Use top-left hash_size x hash_size coefficients as the hash
    hash_result = dct_result[:hash_size, :hash_size]

    # Convert the hash to a binary NumPy array
    hash_array = hash_result.flatten() > 0

    return hash_array.astype(np.uint8)

def find_frame_difference(frame1, frame2):
    hash1 = dct_hash(frame1)
    hash2 = dct_hash(frame2)

    hamming_distance = np.sum(hash1 != hash2)

    if hamming_distance > 0:
        print(f"Hamming distance for non-matching frame: {hamming_distance}")
        return True
    else:
        print(f"Hamming distance for matching frame: {hamming_distance}")
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
        if not find_frame_difference(sync_frame, frame):
            return frame_number

    return None


def main():
    base_vid = cv.VideoCapture('Videos/scenario_base.mp4')  # instance of base video
    alt_vid = cv.VideoCapture('Videos/scenario_alt2.mp4')  # instance of alt video

    # Get the frame rate of the base video
    base_frame_rate = base_vid.get(cv.CAP_PROP_FPS)
    # Set the frame rate of alt_vid2 to match base_vid
    alt_vid.set(cv.CAP_PROP_FPS, base_frame_rate)

    # Capture image of first frame of base_vid
        # ret - (return) boolean that indicates if the .read operation was successful.
    ret, first_frame = base_vid.read()

    sync_frame_number = find_sync_frame(first_frame, base_vid)
    sync_frame = get_frame_at_number(sync_frame_number, base_vid)
    # cv.imshow('sync frame', sync_frame)

    alt_sync_frame_number = find_matching_frame_number(sync_frame, alt_vid)

    min_frame_number = min(sync_frame_number, alt_sync_frame_number)
    # print("min frame number: ", min_frame_number)
    # print("base_frame: ", sync_frame_number)
    # print("alt frame: ", alt_sync_frame_number)

    # Set the starting frame number for both videos
    base_vid.set(cv.CAP_PROP_POS_FRAMES, sync_frame_number - min_frame_number)
    alt_vid.set(cv.CAP_PROP_POS_FRAMES, alt_sync_frame_number - min_frame_number)

    while True:
        ret_base, frame_base = base_vid.read()  # read through the base video
        ret_alt, frame_alt = alt_vid.read()  # read through the alt video

        if not ret_base or not ret_alt:  # if either video ends, break
            break

        # Add text to both videos to signify which is which
        cv.putText(frame_base, "Base Video", (10, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (255, 255, 255), 1)
        cv.putText(frame_alt, "Alt Video", (10, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (255, 255, 255), 1)  # Corrected text here

        # Create a video of both videos stitched togetherq
        concatenated_frame = np.concatenate((frame_base, frame_alt), axis=1)        

        cv.imshow('Synchronized Videos', concatenated_frame)

        if cv.waitKey(30) & 0xFF == ord('q'):
            break

    base_vid.release()
    alt_vid.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()