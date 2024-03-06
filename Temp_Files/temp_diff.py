import cv2 as cv
import numpy as np
import sys

'''
Currently the app looks like it is not catching when contours end once the first difference is visualized.
It might be helpful to use cv.imshow to watch the videos play as the differences are detected.
using cv.imshow to show other things could also be helpful
'''

'''CURRENTLY the second difference never stops because there is always some small difference between the two videos.
To remedy this, we can focus on JUST the contours that were returned (maybe). If those specific contours stop being present, 
then the difference has ended'''

def frame_difference(frame1, frame2, threshold=25):
    # Convert frames to grayscale for simplicity
    gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

    # Find pixel-wise differences in the two frames
    diff = cv.absdiff(gray1, gray2)

    # Create a binary image, setting pixels above a threshold to white and below to black
    _, thresholded_diff = cv.threshold(diff, threshold, 255, cv.THRESH_BINARY)

    cv.imshow("Thresholded diff", thresholded_diff)

    # Find contours in the thresholded difference image
    contours, _ = cv.findContours(thresholded_diff, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Filter out small contours
    min_contour_area = 10  
    contours = [contour for contour in contours if cv.contourArea(contour) > min_contour_area]

    return thresholded_diff, contours

def visualize_difference(frame1, frame2, diff_image, contours, diff_value, frame_count):
    # Copy frames to avoid modifying the original frames
    frame1_with_border = frame1.copy()
    frame2_with_border = frame2.copy()

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
    cv.putText(frame1_with_border, "Video 1", (10, frame1_with_border.shape[0] - 20), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)
    cv.putText(frame2_with_border, "Video 2", (10, frame2_with_border.shape[0] - 20), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)

    # Display the frames with rectangles, borders, and text
    cv.imshow('Synchronized Videos', np.hstack((frame1_with_border, frame2_with_border)))
    # cv.waitKey(0)

    # Display the difference image
    cv.imshow('Difference Image', diff_image)

    # Print the calculated difference value and frame count
    print(f'Difference Value: {diff_value}, Frame Count: {frame_count}')

def main():
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
        print("difference detected at frame: ", diff_detect_frame)
        print("total frame count: ", total_frame_count)
        frames_present = total_frame_count - diff_detect_frame
        print(f"Difference present for {frames_present} frames")
        print("No more frames to read. Terminating Process")
        return
    
    # Set starting frame for each video (we will adjust this by passing in the 
    # synchronization frames found in the video synchronizer)
    cap1.set(cv.CAP_PROP_POS_FRAMES, base_start_frame)
    cap2.set(cv.CAP_PROP_POS_FRAMES, alt_start_frame)

    frame_rate = cap1.get(cv.CAP_PROP_FPS)
    duration_threshold_frames = int(frame_rate * 5)  # Set the duration threshold in frames

    frame_count = 0
    diff_detect_frame = 0
    total_frame_count = 0
    frames_present = 0
    five_sec_flag = False
    difference_count = 0
    ret3, temp_frame1 = cap1.read()
    ret4, temp_frame2 = cap2.read()

    while True:  # We can edit this to only read every 5th or 10th frame to make it more efficient
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not (ret1 and ret2):
            print("No more frames detected. Terminating process...")
            break

        # Perform pixel-wise difference and get contours
        diff_image, contours = frame_difference(frame1, frame2)
        total_frame_count += 1
        

        # Check if any differences are found
        if contours:
            frame_count += 1
            if frame_count % 50 == 0:
                print("frame count: ", frame_count)

            # if frame_count == 1:
            #     diff_detect_frame = total_frame_count
            #     print("set the difference detection frame")
            
            if frame_count == 150:
                difference_count += 1
                print("difference count: ", difference_count)

            if (frame_count >= duration_threshold_frames) and (not five_sec_flag):
                five_sec_flag = True
                temp_frame1 = frame1
                temp_frame2 = frame2
                # visualize_difference(temp_frame1, temp_frame2, diff_image, contours, len(contours), frame_count)
                
                print("set five_sec_flag to TRUE for difference number: ", difference_count)
            
        elif five_sec_flag:
            print(f"difference #{difference_count} has ended and should be visualizing now...")
            visualize_difference(temp_frame1, temp_frame2, diff_image, contours, len(contours), frame_count)
            # print("difference detected at frame: ", diff_detect_frame)
            # print("total frame count: ", total_frame_count)
            # frames_present = total_frame_count - diff_detect_frame
            # print(f"Difference present for {frames_present} frames")
            frame_count = 0
            five_sec_flag = False
            print("flag set to FALSE")

        elif (not five_sec_flag) and (frame_count > 0):
            frame_count = 0    
            # print("set frame count to ZERO")

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
