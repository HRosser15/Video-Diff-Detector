import cv2 as cv
import numpy as np

# h_favg is short for histogram difference and frame average as this program uses a 
# combination of the two.

'''This function is not sensitive enough to detect small changes in the video.
It is not a viable option for the project with the thresholds I tried.'''

def histogram_difference(frame1, frame2):
    hist1 = cv.calcHist([frame1], [0], None, [256], [0, 256])
    hist2 = cv.calcHist([frame2], [0], None, [256], [0, 256])
    

    # Normalize histograms to have the same scale
    cv.normalize(hist1, hist1, 0, 1, cv.NORM_MINMAX)
    cv.normalize(hist2, hist2, 0, 1, cv.NORM_MINMAX)
    cv.imshow('hist1', hist1)
    cv.imshow('hist2', hist2)

    # Calculate histogram difference
    diff = cv.compareHist(hist1, hist2, cv.HISTCMP_CORREL)
    cv.imshow('diff', diff)

    return diff

def frame_average_difference(frame1, frame2):
    avg1 = np.mean(frame1)
    avg2 = np.mean(frame2)

    # Calculate the absolute difference between average pixel values
    diff = np.abs(avg1 - avg2)
    cv.imshow('diff', diff)

    return diff

def main():
    video1_path = '../../Videos/scenario_base.mp4'
    video2_path = '../../Videos/scenario_alt1.mp4'

    cap1 = cv.VideoCapture(video1_path)
    cap2 = cv.VideoCapture(video2_path)

    if not (cap1.isOpened() and cap2.isOpened()):
        print("Error: Unable to open videos.")
        return

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not (ret1 and ret2):
            break

        # Convert frames to grayscale
        gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

        # Perform histogram comparison
        hist_diff = histogram_difference(gray1, gray2)

        # Perform frame averaging
        avg_diff = frame_average_difference(gray1, gray2)

        # Adjust the threshold based on your preference
        threshold = 0.001

        # Check if the difference is beyond the threshold
        if hist_diff < threshold and avg_diff < threshold:
            # Draw rectangle around the region of difference
            cv.rectangle(frame1, (0, 0), (frame1.shape[1], frame1.shape[0]), (0, 0, 255), 2)
            cv.rectangle(frame2, (0, 0), (frame2.shape[1], frame2.shape[0]), (0, 0, 255), 2)

        # Display synchronized frames with rectangles
        cv.imshow('Synchronized Videos', np.hstack((frame1, frame2)))

        # Press 'q' to exit
        if cv.waitKey(20) & 0xFF == ord('q'):
            break

    cap1.release()
    cap2.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
