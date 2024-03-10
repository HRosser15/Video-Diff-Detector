import cv2 as cv
import numpy as np
import sys

# Global variables to store detected contours and their corresponding frame counts
detected_contour_list = []
frame_count_list = []

# Set the desired margin for the bounding rectangles
MARGIN = 0  # Adjust this value to change the margin

def handle_key_events():
    key = cv.waitKey(1)
    if key == ord('p'):  # Pause execution
        print("Paused. Press 'l' to resume.")
        while cv.waitKey(0) != ord('l'):  # Wait for 'l' to resume
            pass

def process_contours(contours, frame_count):
    global detected_contour_list
    global frame_count_list

    new_detected_contour_list = []
    new_frame_count_list = []

    # Update the durations for existing contours
    for i, detected_contour in enumerate(detected_contour_list):
        contour_area = cv.contourArea(detected_contour)
        found_match = False
        for cnt in contours:
            if abs(cv.contourArea(cnt) - contour_area) < 1:  # Threshold here can be adjusted
                new_detected_contour_list.append(cnt)
                new_frame_count_list.append(frame_count_list[i])
                found_match = True
                break
        if not found_match:
            duration = frame_count - frame_count_list[i]
            if duration > 150:
                print(f"Contour removed after {duration} frames")

    # Add new contours
    for cnt in contours:
        contour_area = cv.contourArea(cnt)
        if not any(abs(cv.contourArea(detected_contour) - contour_area) < 1 for detected_contour in new_detected_contour_list):
            new_detected_contour_list.append(cnt)
            new_frame_count_list.append(frame_count)

    detected_contour_list = new_detected_contour_list
    frame_count_list = new_frame_count_list


def split_contours(larger_contour, smaller_contour):
    # Split the larger contour list into two lists based on the smaller contour
    mask = np.zeros_like(larger_contour)
    cv.drawContours(mask, [smaller_contour], 0, (255), thickness=cv.FILLED)
    new_half = cv.bitwise_and(larger_contour, larger_contour, mask=mask)
    orig_half = cv.subtract(larger_contour, new_half)
    return new_half, orig_half

def remove_subset(original_contour, subset_contour):
    # Remove subset contour from the original contour
    mask = np.zeros_like(original_contour)
    cv.drawContours(mask, [subset_contour], 0, (255), thickness=cv.FILLED)
    new_contour_half = cv.subtract(original_contour, mask)
    return new_contour_half

def frame_difference(frame1, frame2, threshold=25):
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
    min_contour_area = 50  # Adjust this value based on your requirements
    contours = [contour for contour in contours if cv.contourArea(contour) > min_contour_area]
    return thresholded_diff, contours

def visualize_difference(frame1, frame2, diff_image, contours, diff_value, cap1):
    # Copy frames to avoid modifying the original frames
    frame1_with_border = frame1.copy()
    frame2_with_border = frame2.copy()

    # Get the frame rate
    frame_rate = cap1.get(cv.CAP_PROP_FPS)
    duration_threshold_frames = int(frame_rate * 5)  # Set the duration threshold in frames

    # Draw rectangles around all detected contours
    bounding_rects = []
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        bounding_rects.append((x - MARGIN, y - MARGIN, w + 2 * MARGIN, h + 2 * MARGIN))

    # Combine overlapping or touching rectangles
    combined_rects = combine_rectangles(bounding_rects)

    # Draw the combined rectangles on the frames
    for rect in combined_rects:
        x, y, w, h = rect
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
    cv.waitKey(30)

    # Display the difference image
    cv.imshow('Difference Image', diff_image)

    # Print the calculated difference value
    # print(f'Difference Value: {diff_value}')

def combine_rectangles(rects):
    # Sort the rectangles by their top-left corner coordinates
    rects.sort(key=lambda rect: (rect[1], rect[0]))

    combined_rects = []
    i = 0
    while i < len(rects):
        x1, y1, w1, h1 = rects[i]
        x2, y2 = x1 + w1, y1 + h1

        j = i + 1
        while j < len(rects):
            xj, yj, wj, hj = rects[j]
            xj2, yj2 = xj + wj, yj + hj

            # Check if the rectangles overlap or touch
            if max(x1, xj) < min(x2, xj2) and max(y1, yj) < min(y2, yj2):
                # Combine the rectangles
                x1 = min(x1, xj)
                y1 = min(y1, yj)
                x2 = max(x2, xj2)
                y2 = max(y2, yj2)
                w1 = x2 - x1
                h1 = y2 - y1
                j += 1
            else:
                break

        combined_rects.append((x1, y1, w1, h1))
        i = j

    return combined_rects

def main():
    global detected_contour_list
    global frame_count_list

    if len(sys.argv) < 5:
        print("Usage: python diff_find.py video_path1 video_path2 base_start_frame alt_start_frame")
        return
    
    video1_path = sys.argv[1]
    video2_path = sys.argv[2]
    base_start_frame = int(sys.argv[3])
    alt_start_frame = int(sys.argv[4])

    video_base_path = 'Videos/'
    full_video1_path = video_base_path + video1_path
    full_video2_path = video_base_path + video2_path

    cap1 = cv.VideoCapture(full_video1_path)
    cap2 = cv.VideoCapture(full_video2_path)

    if not (cap1.isOpened() and cap2.isOpened()):
        print("No more frames to read. Terminating Process")
        return

    frame_rate = cap1.get(cv.CAP_PROP_FPS)
    duration_threshold_frames = int(frame_rate * 5)  # Set the duration threshold in frames

    # Set starting frame for each video (we will adjust this by passing in the 
    # synchronization frames found in the video synchronizer)
    cap1.set(cv.CAP_PROP_POS_FRAMES, base_start_frame)
    cap2.set(cv.CAP_PROP_POS_FRAMES, alt_start_frame)
    


    frame_count = 0

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
            visualize_difference(frame1, frame2, diff_image, contours, len(contours), cap1)

            # Process the contours and update the detected_contour_list
            process_contours(contours, frame_count)

            # Increment the frame count
            frame_count += 1

            # Handle key events to pause and resume execution
            handle_key_events()
        
    cap1.release()
    cap2.release()

    cv.destroyAllWindows()

if __name__ == '__main__':
    main()