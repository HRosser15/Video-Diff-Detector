import cv2 as cv
import numpy as np

# TO RUN:
# open terminal with "CTRL + `"
# enter 'python demo1_no_video_sync.py' and press enter
# press 'q' to exit the video window


def main():
    # Open video capture for base and alternative videos
    base_vid = cv.VideoCapture('Videos/scenario_base.mp4')
    alt_vid = cv.VideoCapture('Videos/scenario_alt2.mp4')

    # Set frame rate to 15 fps for both videos
    desired_frame_rate = 1.0
    base_vid.set(cv.CAP_PROP_POS_FRAMES, 250)
    alt_vid.set(cv.CAP_PROP_POS_FRAMES, 250)


    while True:
        ret_base, frame_base = base_vid.read()
        ret_alt, frame_alt = alt_vid.read()

        if not ret_base or not ret_alt:
            break

        # Add a border around each video frame
        frame_base = cv.copyMakeBorder(frame_base, 10, 50, 10, 10, cv.BORDER_CONSTANT, value=(255, 255, 255))
        frame_alt = cv.copyMakeBorder(frame_alt, 10, 50, 10, 10, cv.BORDER_CONSTANT, value=(255, 255, 255))

        # Add text below each video to indicate which is which
        cv.putText(frame_base, "Base Video", (100, frame_base.shape[0] - 20), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)
        cv.putText(frame_alt, "Alt Video", (100, frame_alt.shape[0] - 20), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)

        # Create a video by concatenating frames of both videos
        concatenated_frame = np.concatenate((frame_base, frame_alt), axis=1)
        
        cv.imshow('Unsynchronized Videos', concatenated_frame)

        # Break the loop if 'q' is pressed
        if cv.waitKey(90) & 0xFF == ord('q'):
            break

    # Release video captures and close windows
    base_vid.release()
    alt_vid.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
