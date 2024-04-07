import cv2 as cv
import numpy as np
import tkinter as tk
import sys

# Global variables to store detected contours and their corresponding frame counts
detected_contour_list = []
frame_count_list = []

def get_screen_resolution():
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()
    return screen_width, screen_height

def rescaleFrame(frame, scale):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

MARGIN = 5  # Adjust this value to change the margin of the bounding box

def handle_key_events(key, paused):
    if key == ord('q'):
        return True, paused  # Quit the program
    elif key == ord('p'):
        paused = not paused  # Toggle pause/resume
        if paused:
            print("Paused. Press 'p' to resume.")
        else:
            print("Resuming...")
    return False, paused

def process_contours(contours, frame_count, total_frame_count):
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
            if duration > 30:  # duration can be adjusted
                print(f"Difference found at {((total_frame_count - duration)/30):.2f} seconds")
                print(f"Difference present for {(duration/30):.2f} seconds\n")


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
    print("new half: ", new_half)
    print("original half: ", orig_half)
    
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
    min_contour_area = 20  # Adjust this value based how strict you need the filtering to be. higher number = more contours are required for a difference to be counted.
    contours = [contour for contour in contours if cv.contourArea(contour) > min_contour_area]
    return thresholded_diff, contours

def visualize_difference(frame1, frame2, diff_image, contours, cap1, fps, frame_scale, border_size, bottom_border_size, text_x_pos, text_y_pos, text_size, text_weight, output_width, output_height, frame_width, frame_height):
    # Copy frames to avoid modifying the original frames
    frame1_with_border = frame1.copy()
    frame2_with_border = frame2.copy()
    # print("frame1_without_border size:", frame1_with_border.shape)

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
    frame1_with_border = cv.copyMakeBorder(frame1_with_border, border_size, bottom_border_size, border_size, border_size, cv.BORDER_CONSTANT, value=(255, 255, 255))
    frame2_with_border = cv.copyMakeBorder(frame2_with_border, border_size, bottom_border_size, border_size, border_size, cv.BORDER_CONSTANT, value=(255, 255, 255))

    # # Print the size of frame1_with_border
    # print("frame1_with_border size:", frame1_with_border.shape)

    # # Print border size and bottom border size
    # print("Border Size:", border_size)
    # print("Bottom Border Size:", bottom_border_size)

    # # Print dimensions of the output video
    # print("Output Video Dimensions:", output_width, "x", output_height)

    # # Print dimensions of the displayed window
    # window_width = frame1_with_border.shape[1] + frame2_with_border.shape[1]
    # window_height = max(frame1_with_border.shape[0], frame2_with_border.shape[0])
    # print("Displayed Window Dimensions:", window_width, "x", window_height)

    # # Print placement of the text
    # print("Text Position (X, Y):", text_x_pos, ",", text_y_pos)

    # Print height of the text
    text_height = cv.getTextSize("Video", cv.FONT_HERSHEY_SIMPLEX, text_size, text_weight)[0][1]
    # print("Text Height:", text_height)

    cv.putText(frame1_with_border, "Video 1", (text_x_pos, text_y_pos), cv.FONT_HERSHEY_SIMPLEX, text_size, (0, 0, 0), text_weight, cv.LINE_AA)
    cv.putText(frame2_with_border, "Video 2", (text_x_pos, text_y_pos), cv.FONT_HERSHEY_SIMPLEX, text_size, (0, 0, 0), text_weight, cv.LINE_AA)

    concatenated_frame = np.concatenate((frame1_with_border, frame2_with_border), axis=1)
    concatenated_frame = rescaleFrame(concatenated_frame, frame_scale)

    cv.imshow('Difference Detection', concatenated_frame)

    # print("=============================================================")

    return concatenated_frame


def combine_rectangles(rects):
    # Sort the rectangles by their top-left corner coordinates
    rects.sort(key=lambda rect: (rect[1], rect[0]))

    combined_rects = []
    for rect in rects:
        x1, y1, w1, h1 = rect
        x2, y2 = x1 + w1, y1 + h1

        merged = False
        for j in range(len(combined_rects)):
            xj, yj, wj, hj = combined_rects[j]
            xj2, yj2 = xj + wj, yj + hj

            # Check if the rectangle overlaps with the existing combined rectangle
            if max(x1, xj) < min(x2, xj2) and max(y1, yj) < min(y2, yj2):
                # Merge the rectangles
                x1 = min(x1, xj)
                y1 = min(y1, yj)
                x2 = max(x2, xj2)
                y2 = max(y2, yj2)
                w1 = x2 - x1
                h1 = y2 - y1
                combined_rects[j] = (x1, y1, w1, h1)
                merged = True
                break

        if not merged:
            combined_rects.append((x1, y1, w1, h1))

    return combined_rects

def set_display_properties(cap1):
    screen_width, screen_height = get_screen_resolution()
    print("Screen resolution: {}x{}".format(screen_width, screen_height))

    frame_width = int(cap1.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap1.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap1.get(cv.CAP_PROP_FPS))

    # Adjust frame size to better fit current user's screen
    screen_div_by_three = screen_width / 4
    frame_scale = screen_div_by_three / frame_width
    # print(" FRAME SCALE: ", frame_scale)

    # Define border sizes and output video properties
    border_size = int(frame_width/30)
    output_width = int((frame_width + (border_size * 2)) * 2 * frame_scale)
    # print("OUTPUT WIDTH: ", output_width)

    text_size = 1
    text_weight = 1

    if frame_width < 360:
        text_size = 1
        text_weight = 1

    elif frame_width < 720:
        text_size = 2
        text_weight = 2

    elif frame_width < 1080:
        text_size = 3
        text_weight = 2

    else: 
        text_size = 4
        text_weight = 3

    text_dimensions = cv.getTextSize("Video", cv.FONT_HERSHEY_TRIPLEX, text_size, text_weight)
    text_height = cv.getTextSize("Video", cv.FONT_HERSHEY_TRIPLEX, text_size, text_weight)[0][1]
    text_tuple = text_dimensions[0]
    text_width = text_tuple[0]
    bottom_border_size = int(text_height * 2.5)
    output_height = int((frame_height + border_size + bottom_border_size) * frame_scale)
    # print("Text size: ", text_size)
    # print("Text weight: ", text_weight)
    # print("frame width: ", frame_width)
    # print("frame height: ", frame_height)
    # print("Output width: ", output_width)
    # print("Output height: ", output_height)
    # print("Text width: ", text_width)
    # print("Text height inside set_display_properties: ", text_height)
    

    # Define coordinates and properties for text placement
    video_width = frame_width + border_size * 2 
    text_x_offset = int(text_width / 2)
    text_x_pos = int((video_width / 2 - text_x_offset))
    text_y_pos = int(frame_height + border_size + text_height + (frame_height / 30)) 
    # print(f"text_y_pos = {frame_height} + {border_size} + {text_height} + 10)")
    # print(f"text y pos with no int = {frame_height + border_size + text_height + 10}")
    # print(f"text y pos with int = {int(frame_height + border_size + text_height + 10)}")

    # print("text_x_pos = int((video_width / 2 - text_x_offset))")
    # print(f"Text x position: {text_x_pos} = ({video_width} / 2 - {text_x_offset})")  

    # print("Frame width: ", frame_width)
    # print("Frame height: ", frame_height)
    # print("Border size: ", border_size)
    
    # print("Text x position: ", text_x_pos)
    # print("Text y position: ", text_y_pos)

    return fps, frame_scale, border_size, bottom_border_size, text_x_pos, text_y_pos, text_size, text_weight, output_width, output_height, frame_width, frame_height 


def write_output_video(frame, output_video):
    output_video.write(frame)


def main():
    if len(sys.argv) < 5:
        print("Usage: python diff.py video1_file_name.ext video2_file_name.ext base_start_frame alt_start_frame")
        print(" ** temp dev use: base_start_frame and alt_start_frame are passed in from sync.py")
        print(" ** temp dev use: application will add the path to the Videos folder by default. Please just add the file name and extension.")
        return
    
    print("============================================")
    print("||    Starting difference detection...    ||")
    print("||     Press 'q' to quit the program.     ||")
    print("||  Press 'p' to pause/resume the video.  ||")
    print("============================================")
    
    global detected_contour_list
    global frame_count_list
    total_frame_count = 0

    video1_path = sys.argv[1]
    video2_path = sys.argv[2]
    base_start_frame = int(sys.argv[3])
    alt_start_frame = int(sys.argv[4])


    video_base_path = 'Videos/'
    full_video1_path = video_base_path + video1_path
    full_video2_path = video_base_path + video2_path

    
    # Open video capture for base and alternative videos
    cap1 = cv.VideoCapture(full_video1_path)
    cap2 = cv.VideoCapture(full_video2_path)

    if not (cap1.isOpened() and cap2.isOpened()):
        print("Error: Unable to open videos.")
        return

    frame_rate = cap1.get(cv.CAP_PROP_FPS)
    duration_threshold_frames = int(frame_rate * 5)  # Set the duration threshold in frames

    fps, frame_scale, border_size, bottom_border_size, text_x_pos, text_y_pos, text_size, text_weight, output_width, output_height, frame_width, frame_height = set_display_properties(cap1)

    # Set starting frame for each video
    cap1.set(cv.CAP_PROP_POS_FRAMES, base_start_frame)
    cap2.set(cv.CAP_PROP_POS_FRAMES, alt_start_frame)

    frame_count = 0
    paused = False

    codec = cv.VideoWriter_fourcc(*'mp4v')
    output_video = cv.VideoWriter('Output_Videos/Differences.mp4', codec, fps, (output_width, output_height), True)

    while True:
        if not paused:
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
                concatenated_frame = visualize_difference(frame1, frame2, diff_image, contours, cap1, fps, frame_scale, border_size, bottom_border_size, text_x_pos, text_y_pos, text_size, text_weight, output_width, output_height, frame_width, frame_height)

                # Process the contours and update the detected_contour_list
                process_contours(contours, frame_count, total_frame_count)

                # Write the frame to the output video
                write_output_video(concatenated_frame, output_video)

                # Increment the frame count
                frame_count += 1

        # Handle key events to pause, resume, or quit
        key = cv.waitKey(1)
        quit_program, paused = handle_key_events(key, paused)
        if quit_program:
            break

        total_frame_count += 1

    cap1.release()
    cap2.release()
    output_video.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()