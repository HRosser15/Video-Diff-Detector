import cv2 as cv
import numpy as np
import tkinter as tk

def get_screen_resolution():
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()
    return screen_width, screen_height

def find_frame_difference(gray1, frame2):
    # Convert frames to grayscale for simplicityx
    gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)  

    # Find pixel-wise differences in the two frames
    diff = cv.absdiff(gray1, gray2)

    # Create a binary image, setting pixels above a threshold to white and below to black
    _, thresholded_diff = cv.threshold(diff, 50, 255, cv.THRESH_BINARY)

    # Count the number of non-black pixels in the thresholded_diff
    non_zero_count = np.count_nonzero(thresholded_diff)

    # Return true if more than "threshold" non-black pixels are found
    threshold = 20
    return non_zero_count > threshold

def find_matching_frame(gray1, frame2):
    # Convert frames to grayscale for simplicity
    gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)  

    # Find pixel-wise differences in the two frames
    diff = cv.absdiff(gray1, gray2)

    # Create a binary image, setting pixels above a threshold to white and below to black
    _, thresholded_diff = cv.threshold(diff, 50, 255, cv.THRESH_BINARY)

    # Count the number of non-black pixels in the thresholded_diff
    non_zero_count = np.count_nonzero(thresholded_diff)

    # Return true if less than "threshold" non-black pixels are found
    threshold = 40
    return non_zero_count < threshold

def find_sync_frame_number(first_frame, base_vid):
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
        if find_frame_difference(first_frame, frame):
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

def find_matching_frame_number(sync_frame, alt_vid):
    frame_number = 0

    while True:
        # Read the next frame
        ret, frame = alt_vid.read()

        # If no frame is found, break the loop
        if not ret:
            break

        frame_number += 1

        # Return the frame number if a matching frame is found
        if find_matching_frame(sync_frame, frame):
            return frame_number

    return None

def rescaleFrame(frame, scale):
    # images, videos and live videos
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)  # tuple of width and height

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def add_border(frame, border_size):
    border_color = (255, 255, 255)  
    bordered_frame = cv.copyMakeBorder(frame, border_size, border_size, border_size, border_size, cv.BORDER_CONSTANT, value=border_color)
    return bordered_frame

def main():
    screen_width, screen_height = get_screen_resolution()
    print("Screen resolution: {}x{}".format(screen_width, screen_height))

    # Open video capture for base and alternative videos
    # base_vid = cv.VideoCapture('../../Videos/scenario_base.mp4')
    # alt_vid = cv.VideoCapture('../../Videos/scenario_alt2.mp4')
    # base_vid = cv.VideoCapture('../../Videos/Gauge_base.mp4')
    # alt_vid = cv.VideoCapture('../../Videos/Gauge_diff2.mp4')
    base_vid = cv.VideoCapture('../../Videos/3D_cat1.mov')
    alt_vid = cv.VideoCapture('../../Videos/3D_cat2.mov')
    # base_vid = cv.VideoCapture('../../Videos/3D_Cockpit_3.mp4')
    # alt_vid = cv.VideoCapture('../../Videos/3D_Cockpit_4.mp4')

    # Get and set frame rates to match for synchronization
    base_frame_rate = base_vid.get(cv.CAP_PROP_FPS)
    alt_vid.set(cv.CAP_PROP_FPS, base_frame_rate)

    # Get video properties for output
    frame_width = int(base_vid.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(base_vid.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(base_vid.get(cv.CAP_PROP_FPS))

    # Adjust frame size to better fit current user's screen
    screen_div_by_three = screen_width / 4
    frame_scale = screen_div_by_three / frame_width

    # Capture the first frame of the base video
    ret, first_frame = base_vid.read()

    # Convert it to grayscale for simplicity
    first_frame = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY) 
    
    # Find frame numbers where videos are synchronized
    sync_frame_number = find_sync_frame_number(first_frame, base_vid)

    # Get the frame at the sync_frame_number
    sync_frame = get_frame_at_number(sync_frame_number, base_vid)
    print("Base video reference frame found at frame number: ", sync_frame_number)

    # Convert it to grayscale for simplicity
    sync_frame = cv.cvtColor(sync_frame, cv.COLOR_BGR2GRAY)
    
    # Find the frame number in the alt video that contains the sync frame
    alt_sync_frame_number = find_matching_frame_number(sync_frame, alt_vid)
    print("Alt video reference frame found at frame number: ", alt_sync_frame_number)

    # find the minimum frame number
    min_frame_number = min(sync_frame_number, alt_sync_frame_number)

    # Print frame numbers for reference
    print("Minumum frame number: ", min_frame_number)
    print("Starting base video from frame number: ", sync_frame_number - min_frame_number)
    print("Starting alt video from frame number: ", alt_sync_frame_number - min_frame_number)

    # Set starting frame numbers for both videos
    base_vid.set(cv.CAP_PROP_POS_FRAMES, sync_frame_number - min_frame_number)
    alt_vid.set(cv.CAP_PROP_POS_FRAMES, alt_sync_frame_number - min_frame_number)

    

    
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

    # Define border sizes and output video properties
    border_size = int(frame_width/30)
    bottom_border_size = int(frame_height/10 + (text_size * 8))
    output_width = int((frame_width + (border_size * 2)) * 2 * frame_scale)
    output_height = int((frame_height + border_size + bottom_border_size) * frame_scale)
    
    print("frame height: ", frame_height)
    print("Output width: ", output_width)
    print("Output height: ", output_height)


    # Define coordinates and properties for text placement
    video_width = frame_width + border_size * 2 
    text_x_offset = int(frame_width / 8)
    text_y_offset = (frame_height / 20)
    text_x_pos = int((video_width / 2 - text_x_offset))
    text_y_pos = int((output_height - text_y_offset + (text_size * 4)) / frame_scale)

    codec = cv.VideoWriter_fourcc(*'mp4v')
    output_video = cv.VideoWriter('Videos/3D_cat_synced.mp4', codec, fps, (output_width, output_height), True)
    
    frame_number = 0    

    print("Frame width: ", frame_width)
    print("Frame height: ", frame_height)

    while True:
        ret_base, frame_base = base_vid.read()
        ret_alt, frame_alt = alt_vid.read()

        if not ret_base or not ret_alt:
            break

        # Increment frame number
        frame_number += 1

        # Stop if frame number reaches 240
        if frame_number > 180:
            break

        # Add a borders and text
        bordered_frame_base = cv.copyMakeBorder(frame_base, border_size, bottom_border_size, border_size, border_size, cv.BORDER_CONSTANT, value=(255, 255, 255))
        bordered_frame_alt = cv.copyMakeBorder(frame_alt, border_size, bottom_border_size, border_size, border_size, cv.BORDER_CONSTANT, value=(255, 255, 255))
        cv.putText(bordered_frame_base, "Video 1", (text_x_pos, text_y_pos), cv.FONT_HERSHEY_COMPLEX_SMALL, text_size, (0, 0, 0), text_weight)
        cv.putText(bordered_frame_alt, "Video 2", (text_x_pos, text_y_pos), cv.FONT_HERSHEY_COMPLEX_SMALL, text_size, (0, 0, 0), text_weight)

        concatenated_frame = np.concatenate((bordered_frame_base, bordered_frame_alt), axis=1)
        concatenated_frame = rescaleFrame(concatenated_frame, frame_scale)

        cv.imshow('Synced Video', concatenated_frame)

        # Write the frame to the output video
        output_video.write(concatenated_frame)

        # play frames with 1 ms delay between each frame
        # Break the loop if 'q' or 'Q' is pressed
        key = cv.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break

    # Release video captures and close windows
    base_vid.release()
    alt_vid.release()
    cv.destroyAllWindows()

    # Release the output video
    output_video.release()

if __name__ == '__main__':
    main()