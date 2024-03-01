import cv2 as cv
import numpy as np

def frame_difference(frame1, frame2, threshold=15):
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
    min_contour_area = 40  # Adjust this value based on requirements of video
    contours = [contour for contour in contours if cv.contourArea(contour) > min_contour_area]

    return thresholded_diff, contours

def rescaleFrame(frame, scale=0.75):
    # images, videos, and live videos
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)  # tuple of width and height

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def visualize_difference(frame1, frame2, diff_image, contours, diff_value):
    # Copy frames and rescale them to avoid modifying the original frames
    frame1_with_border = rescaleFrame(frame1)
    frame2_with_border = rescaleFrame(frame2)

    # Draw rectangles around the detected differences
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        
        # Draw rectangles on frames with borders
        cv.rectangle(frame1_with_border, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv.rectangle(frame2_with_border, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Add a border around each video frame
    frame1_with_border = cv.copyMakeBorder(frame1_with_border, 10, 50, 10, 10, cv.BORDER_CONSTANT, value=(255, 255, 255))
    frame2_with_border = cv.copyMakeBorder(frame2_with_border, 10, 50, 10, 10, cv.BORDER_CONSTANT, value=(255, 255, 255))

    # Add text under each video
    cv.putText(frame1_with_border, "Video 1", (100, frame1_with_border.shape[0] - 20), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)
    cv.putText(frame2_with_border, "Video 2", (100, frame2_with_border.shape[0] - 20), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)

    # Display the frames with rectangles, borders, and text
    cv.imshow('Synchronized Videos', np.hstack((frame1_with_border, frame2_with_border)))
    cv.waitKey(0)

    # Display the difference image
    cv.imshow('Difference Image', diff_image)

    # Print the calculated difference value
    print(f'Difference Value: {diff_value}')


def main():
    video1_path = '../../Videos/mochi1.mov'
    video2_path = '../../Videos/mochi2.mov'

    cap1 = cv.VideoCapture(video1_path)
    cap2 = cv.VideoCapture(video2_path)

    if not (cap1.isOpened() and cap2.isOpened()):
        print("Error: Unable to open videos.")
        return
    
    # Set starting frame for each video (we will adjust this by passing in the 
    # synchronization frames found in the video synchronizer)
    cap1.set(cv.CAP_PROP_POS_FRAMES, 0)
    cap2.set(cv.CAP_PROP_POS_FRAMES, 32)

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not (ret1 and ret2):
            print("No more frames detected. Terminating process...")
            break

        # Perform pixel-wise difference and get contours
        diff_image, contours = frame_difference(frame1, frame2)

        # Check if any differences are found
        if contours:
            # Visualize the differences
            visualize_difference(frame1, frame2, diff_image, contours, len(contours))

        # Press 'q' to exit
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap1.release()
    cap2.release()

    # Add a delay to keep the windows open
    # cv.waitKey(0)

    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
