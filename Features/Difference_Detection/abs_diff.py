import cv2 as cv
import numpy as np

def frame_difference(frame1, frame2, threshold=50):
    # Convert frames to grayscale for simplicity
    gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

    # Find pixel-wise differences in the two frames
    diff = cv.absdiff(gray1, gray2)

    # Create a binary image, setting pixels above a threshold to white and below to black
    _, thresholded_diff = cv.threshold(diff, threshold, 255, cv.THRESH_BINARY)

    # Find contours in the thresholded difference image
    contours, _ = cv.findContours(thresholded_diff, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Filter out small contours
    min_contour_area = 100  # Adjust this value based on your requirements
    contours = [contour for contour in contours if cv.contourArea(contour) > min_contour_area]

    return thresholded_diff, contours

def visualize_difference(frame1, frame2, diff_image, contours, diff_value):
    # Draw rectangles around the detected differences
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        cv.rectangle(frame1, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv.rectangle(frame2, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Display the frames with rectangles
    cv.imshow('Synchronized Videos', np.hstack((frame1, frame2)))

    # Display the difference image
    cv.imshow('Difference Image', diff_image)

    # Print the calculated difference value
    print(f'Difference Value: {diff_value}')

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
            print("Error: Unable to read frames.")
            break

        # Perform pixel-wise difference and get contours
        diff_image, contours = frame_difference(frame1, frame2)

        # Check if any differences are found
        if contours:
            # Visualize the differences
            visualize_difference(frame1, frame2, diff_image, contours, len(contours))

        # Press 'q' to exit
        if cv.waitKey(30) & 0xFF == ord('q'):
            break

    cap1.release()
    cap2.release()

    # Add a delay to keep the windows open
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
