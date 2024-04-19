import cv2 as cv
import numpy as np
import tkinter as tk
import abc
import csv
import sys
import os
from datetime import datetime
import argparse


# Global variables to store detected contours and their corresponding frame counts
detected_contour_list = []
frame_count_list = []
MARGIN = None
WEIGHT = None
previous_frame = None
BBOX_COLOR = None


class Logger(abc.ABC):
    @abc.abstractmethod
    def log(self, message):
        pass


class CSVLogger(Logger):
    def __init__(self, file_path):
        self.file_path = file_path
        self.is_header_written = False
        self.is_details_written = False
        self.details = []

    def log(self, message):
        with open(self.file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            if not self.is_header_written:
                writer.writerow(["Details"])
                self.is_header_written = True
            elif message[0] == "User's screen resolution:":
                writer.writerow(message)
            elif not self.is_details_written:
                self.details.append(message)
            else:
                writer.writerow(message)

    def write_details(self):
        with open(self.file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            for detail in self.details:
                writer.writerow(detail)
            writer.writerow([])
            writer.writerow([])
            writer.writerow([])
            writer.writerow(["Timestamp", "Duration"])
            self.is_details_written = True


class ConsoleLogger(Logger):
    def log(self, message):
        print(message)


class NullLogger(Logger):
    def log(self, message):
        pass


class LoggerFactory:
    @staticmethod
    def create_logger(logger_type, file_path=None):
        if logger_type == 'csv':
            return CSVLogger(file_path)
        elif logger_type == 'console':
            return ConsoleLogger()
        else:
            return NullLogger()
        

def create_folder_structure(log_directory=None, video1_path=None):
    if log_directory:
        current_datetime_folder = log_directory
    else:
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_folder = "Output"
        video1_folder = os.path.splitext(os.path.basename(video1_path))[0]  # Get the filename without extension
        current_datetime_folder = os.path.join(output_folder, video1_folder, current_datetime)

    difference_images_folder = os.path.join(current_datetime_folder, "Difference_Images")
    os.makedirs(difference_images_folder, exist_ok=True)

    return current_datetime_folder, difference_images_folder

# When video ends, save the remaining differences if they meet the duration threshold.
def save_remaining_differences(detected_contour_list, frame_count_list, total_frame_count, fps, duration_threshold, logger, difference_images_folder, last_frame1, last_frame2, frame_scale, border_size, bottom_border_size, text_x_pos, text_y_pos, text_size, text_weight):
    existing_filenames = {}
    for i, detected_contour in enumerate(detected_contour_list):
        duration = total_frame_count - frame_count_list[i]
        if duration > (fps * duration_threshold):
            timestamp = (total_frame_count - duration) / fps
            duration_seconds = duration / fps
            message = f"Difference found at {timestamp:.2f} seconds, present for {duration_seconds:.2f} seconds"
            logger.log([f"{timestamp:.2f}", f"{duration_seconds:.2f}"])
            print(message)

            # Draw bounding rectangle with margin around the specific contour on both frames
            x, y, w, h = cv.boundingRect(detected_contour)
            frame1_with_rect = last_frame1.copy()
            frame2_with_rect = last_frame2.copy()
            cv.rectangle(frame1_with_rect, (x - MARGIN, y - MARGIN), (x + w + MARGIN, y + h + MARGIN), BBOX_COLOR, WEIGHT)
            cv.rectangle(frame2_with_rect, (x - MARGIN, y - MARGIN), (x + w + MARGIN, y + h + MARGIN), BBOX_COLOR, WEIGHT)

            # Add borders and text to the frames
            frame1_with_border = cv.copyMakeBorder(frame1_with_rect, border_size, bottom_border_size, border_size, border_size, cv.BORDER_CONSTANT, value=(255, 255, 255))
            frame2_with_border = cv.copyMakeBorder(frame2_with_rect, border_size, bottom_border_size, border_size, border_size, cv.BORDER_CONSTANT, value=(255, 255, 255))
            cv.putText(frame1_with_border, "Video 1", (text_x_pos, text_y_pos), cv.FONT_HERSHEY_SIMPLEX, text_size, (0, 0, 0), text_weight, cv.LINE_AA)
            cv.putText(frame2_with_border, "Video 2", (text_x_pos, text_y_pos), cv.FONT_HERSHEY_SIMPLEX, text_size, (0, 0, 0), text_weight, cv.LINE_AA)

            # Concatenate the frames and rescale the image
            concatenated_frame = np.concatenate((frame1_with_border, frame2_with_border), axis=1)
            concatenated_frame = rescaleFrame(concatenated_frame, frame_scale)

            # Generate a unique filename based on timestamp, duration, and an incrementing digit if necessary
            base_filename = f"difference_{timestamp:.2f}_{duration_seconds:.2f}"
            if base_filename not in existing_filenames:
                existing_filenames[base_filename] = 0
            existing_filenames[base_filename] += 1
            counter = existing_filenames[base_filename]

            filename = f"{base_filename}-{counter}.jpg" if counter > 1 else f"{base_filename}.jpg"
            difference_image_path = os.path.join(difference_images_folder, filename)
            cv.imwrite(difference_image_path, concatenated_frame)

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



def handle_key_events(key, paused):
    if key == ord('q'):
        print("Quitting the program...")
        return True, paused  # Quit the program
    elif key == ord('p'):
        paused = not paused  # Toggle pause/resume
        if paused:
            print("Paused. Press 'p' to resume.")
        else:
            print("Resuming...")
    return False, paused

def process_contours(contours, frame_count, total_frame_count, fps, duration_threshold, logger, difference_images_folder, frame1, frame2, frame_scale, border_size, bottom_border_size, text_x_pos, text_y_pos, text_size, text_weight):
    global detected_contour_list
    global frame_count_list
    global previous_frame1
    global previous_frame2

    new_detected_contour_list = []
    new_frame_count_list = []

    existing_filenames = {}

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
            if duration > (fps * duration_threshold):
                timestamp = (total_frame_count - duration) / fps
                duration_seconds = duration / fps
                message = f"Difference found at {timestamp:.2f} seconds, present for {duration_seconds:.2f} seconds"
                logger.log([f"{timestamp:.2f}", f"{duration_seconds:.2f}"])
                print(message)

                x, y, w, h = cv.boundingRect(detected_contour)
                frame1_with_rect = previous_frame1.copy()
                frame2_with_rect = previous_frame2.copy()
                cv.rectangle(frame1_with_rect, (x - MARGIN, y - MARGIN), (x + w + MARGIN, y + h + MARGIN), BBOX_COLOR, WEIGHT)
                cv.rectangle(frame2_with_rect, (x - MARGIN, y - MARGIN), (x + w + MARGIN, y + h + MARGIN), BBOX_COLOR, WEIGHT)

                frame1_with_border = cv.copyMakeBorder(frame1_with_rect, border_size, bottom_border_size, border_size, border_size, cv.BORDER_CONSTANT, value=(255, 255, 255))
                frame2_with_border = cv.copyMakeBorder(frame2_with_rect, border_size, bottom_border_size, border_size, border_size, cv.BORDER_CONSTANT, value=(255, 255, 255))
                cv.putText(frame1_with_border, "Video 1", (text_x_pos, text_y_pos), cv.FONT_HERSHEY_SIMPLEX, text_size, (0, 0, 0), text_weight, cv.LINE_AA)
                cv.putText(frame2_with_border, "Video 2", (text_x_pos, text_y_pos), cv.FONT_HERSHEY_SIMPLEX, text_size, (0, 0, 0), text_weight, cv.LINE_AA)

                concatenated_frame = np.concatenate((frame1_with_border, frame2_with_border), axis=1)
                concatenated_frame = rescaleFrame(concatenated_frame, frame_scale)

                # Generate a unique filename based on timestamp, duration, and an incrementing digit if necessary
                base_filename = f"difference_{timestamp:.2f}_{duration_seconds:.2f}"
                if base_filename not in existing_filenames:
                    existing_filenames[base_filename] = 0
                existing_filenames[base_filename] += 1
                counter = existing_filenames[base_filename]

                filename = f"{base_filename}-{counter}.jpg" if counter > 1 else f"{base_filename}.jpg"
                difference_image_path = os.path.join(difference_images_folder, filename)
                cv.imwrite(difference_image_path, concatenated_frame)

    # Add new contours
    for cnt in contours:
        contour_area = cv.contourArea(cnt)
        if not any(abs(cv.contourArea(detected_contour) - contour_area) < 1 for detected_contour in new_detected_contour_list):
            new_detected_contour_list.append(cnt)
            new_frame_count_list.append(frame_count)

    detected_contour_list = new_detected_contour_list
    frame_count_list = new_frame_count_list
    previous_frame1 = frame1.copy()
    previous_frame2 = frame2.copy()


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

def frame_difference(frame1, frame2, threshold, min_contour_area):
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
        cv.rectangle(frame1_with_border, (x, y), (x + w, y + h), BBOX_COLOR, WEIGHT)
        cv.rectangle(frame2_with_border, (x, y), (x + w, y + h), BBOX_COLOR, WEIGHT)

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

def set_display_properties(cap1, resolution, min_contour_area, threshold):
    screen_width, screen_height = get_screen_resolution()

    frame_width = int(cap1.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap1.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap1.get(cv.CAP_PROP_FPS))

    if resolution is not None:
        output_width, output_height = map(int, resolution.split('x'))

    # Adjust frame size to better fit current user's screen
    screen_div_by_three = screen_width / 4
    frame_scale = screen_div_by_three / frame_width
    # print(" FRAME SCALE: ", frame_scale)

    # Define border sizes and output video properties
    border_size = int(frame_width/30)
    if resolution is None:
        output_width = int((frame_width + (border_size * 2)) * 2 * frame_scale)

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
        text_weight = margin = 4

    else: 
        text_size = 4
        text_weight = 3

    if min_contour_area is None:
        if frame_width < 360:
            min_contour_area = 20
        elif frame_width < 720:
            min_contour_area = 50
        elif frame_width < 1080:
            min_contour_area = 100
        else:
            min_contour_area = 150
        print("Setting minimum contour area to: ", min_contour_area)
    else:
        print("Minimum contour area manually set to: ", min_contour_area)

    if threshold is None:
        if frame_width < 360:
            threshold = 15
        elif frame_width < 720:
            threshold = 20
        elif frame_width < 1080:
            threshold = 25
        else:
            threshold = 30
        print("Setting threshold to: ", threshold)
    else:
        print("Threshold manually set to: ", threshold)

    if WEIGHT is None:
        if frame_width < 360:
            weight = 2
        elif frame_width < 720:
            weight = 3
        elif frame_width < 1080:
            weight = 4
        else:
            weight = 5
        print("Setting bounding box weight to: ", weight)
    else:
        weight = WEIGHT
        print("Bounding box weight manually set to: ", weight)

    if MARGIN is None:
        if frame_width < 360:
            margin = 5
        elif frame_width < 720:
            margin = 8
        elif frame_width < 1080:
            margin = 12
        else:
            margin = 15
        print("Setting bounding box margin to: ", margin)
    else:
        margin = MARGIN
        print("Bounding box margin manually set to: ", margin)

    text_dimensions = cv.getTextSize("Video", cv.FONT_HERSHEY_TRIPLEX, text_size, text_weight)
    text_height = cv.getTextSize("Video", cv.FONT_HERSHEY_TRIPLEX, text_size, text_weight)[0][1]
    text_tuple = text_dimensions[0]
    text_width = text_tuple[0]
    bottom_border_size = int(text_height * 2.5)
    if resolution is None:
        output_height = int((frame_height + border_size + bottom_border_size) * frame_scale)

    video_width = frame_width + border_size * 2 
    text_x_offset = int(text_width / 2)
    text_x_pos = int((video_width / 2 - text_x_offset))
    text_y_pos = int(frame_height + border_size + text_height + (frame_height / 30)) 

    return fps, frame_scale, border_size, bottom_border_size, text_x_pos, text_y_pos, text_size, text_weight, output_width, output_height, frame_width, frame_height, margin, weight, min_contour_area, threshold


def write_output_video(frame, output_video):
    output_video.write(frame)


def main():
    parser = argparse.ArgumentParser(description='Video Difference Detector')
    parser.add_argument('video1_path', help='Path to the first video file')
    parser.add_argument('video2_path', help='Path to the second video file')
    parser.add_argument('base_start_frame', type=int, help='Starting frame number for the base video')
    parser.add_argument('alt_start_frame', type=int, help='Starting frame number for the alternative video')
    parser.add_argument('-w', '--weight', type=int, help='Weight of the bounding boxes (default: (based on input video dimensions))')
    parser.add_argument('-m', '--margin', type=int, help='Margin around the bounding boxes (default: (based on input video dimensions))')
    parser.add_argument('-v', '--video', default='false', help='Save the analysis process to video')
    parser.add_argument('-t', '--threshold', type=int, help='Set the threshold value for pixel-wise difference calculation (default: 25).')
    parser.add_argument('-c', '--contour-area', type=int, help='Set the minimum contour area threshold for contour filtering (default: 100).')
    parser.add_argument('-r', '--resolution', help='Set the resolution for the output video (e.g., 1280x720).')
    parser.add_argument('-l', '--log-file', help='Specify the path to a log file for writing log messages.')
    parser.add_argument('-d', '--duration', type=int, default=1, help='Specify the duration (in seconds) a difference must be present to be logged (default: 1 second).')
    parser.add_argument('-b', '--box-color', default='yellow', help='Set the color of the bounding boxes (default: yellow).')
    parser.add_argument('-nv', '--no-visualization', action='store_true', help='Disable visualization of the analysis process')
    args = parser.parse_args()
    
    global BBOX_COLOR
    
    if args.box_color.lower() == 'red':
        BBOX_COLOR = (0, 0, 255)
    elif args.box_color.lower() == 'green':
        BBOX_COLOR = (0, 255, 0)
    elif args.box_color.lower() == 'blue':
        BBOX_COLOR = (255, 0, 0)
    elif args.box_color.lower() == 'yellow':
        BBOX_COLOR = (0, 255, 255)
    elif args.box_color.lower() == 'pink':
        BBOX_COLOR = (203, 192, 255)
    elif args.box_color.lower() == 'purple':
        BBOX_COLOR = (128, 0, 128)
    else:
        print("Invalid color specified. Using default color (yellow).")
        BBOX_COLOR = (0, 255, 255)

    global detected_contour_list
    global frame_count_list
    global WEIGHT
    global MARGIN
    WEIGHT = args.weight
    MARGIN = args.margin
    previous_frame = None
    print(MARGIN)
       
    print("============================================")
    print("||    Starting difference detection...    ||")
    print("||     Press 'q' to quit the program.     ||")
    print("||  Press 'p' to pause/resume the video.  ||")
    print("============================================")
    

    video1_path = args.video1_path
    video2_path = args.video2_path
    base_start_frame = args.base_start_frame
    alt_start_frame = args.alt_start_frame
    threshold = args.threshold 
    min_contour_area = args.contour_area
    duration_threshold = args.duration 


    video_base_path = 'Videos/'
    full_video1_path = video_base_path + video1_path
    full_video2_path = video_base_path + video2_path

    
    # Open video capture for base and alternative videos
    cap1 = cv.VideoCapture(full_video1_path)
    cap2 = cv.VideoCapture(full_video2_path)

    if not (cap1.isOpened() and cap2.isOpened()):
        print("Error: Unable to open videos.")
        return
    
    fps1 = int(cap1.get(cv.CAP_PROP_FPS))
    fps2 = int(cap2.get(cv.CAP_PROP_FPS))
    frame_width1 = int(cap1.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height1 = int(cap1.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_width2 = int(cap2.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height2 = int(cap2.get(cv.CAP_PROP_FRAME_HEIGHT))

    fps, frame_scale, border_size, bottom_border_size, text_x_pos, text_y_pos, text_size, text_weight, output_width, output_height, frame_width, frame_height, MARGIN, WEIGHT, min_contour_area, threshold = set_display_properties(cap1, args.resolution, min_contour_area, threshold)
    
    # Set starting frame for each video
    cap1.set(cv.CAP_PROP_POS_FRAMES, base_start_frame)
    cap2.set(cv.CAP_PROP_POS_FRAMES, alt_start_frame)

    frame_count = 0
    total_frame_count = 0
    paused = False

    codec = cv.VideoWriter_fourcc(*'mp4v')
    save_video = args.video.lower() == 'true'
    if save_video:
        output_video = cv.VideoWriter(output_video_path, codec, fps, (output_width, output_height), True)
    else:
        output_video = None

    if args.log_file:
        log_directory = os.path.dirname(args.log_file)
        csv_filename = os.path.basename(args.log_file)
    else:
        log_directory = None
        csv_filename = "DifferenceLog.csv"

    current_datetime_folder, difference_images_folder = create_folder_structure(log_directory, args.video1_path)
    output_video_path = os.path.join(current_datetime_folder, "Output_Video.mp4")
    csv_path = os.path.join(current_datetime_folder, csv_filename)

    logger = LoggerFactory.create_logger('csv', csv_path)

    screen_width, screen_height = get_screen_resolution()
    logger.log(["Details"])
    logger.log(["User's screen resolution:", f"{screen_width}x{screen_height}"])
    logger.log(["Base video resolution:", f"{frame_width1}x{frame_height1}"])
    logger.log(["Base video frame rate (fps):", fps1])
    logger.log(["Delta video resolution:", f"{frame_width2}x{frame_height2}"])
    logger.log(["Delta video frame rate (fps):", fps2])
    logger.log(["Delta video frame rate (fps):", fps2])
    logger.log(["Base video starting frame:", base_start_frame])
    logger.log(["Delta video starting frame:", alt_start_frame])
    logger.log(["Set threshold:", threshold])
    logger.log(["Set minimum contour area:", min_contour_area])
    logger.log(["Set duration threshold (in seconds):", duration_threshold])
    logger.log(["Output image (or video) resolution:", f"{output_width}x{output_height}"])

    logger.write_details()
    print("Duration threshold (in seconds): ", duration_threshold)
    
    while True:
        if not paused:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if not (ret1 and ret2):
                print("End of video reached.")
                break

            # Perform pixel-wise difference and get contours
            diff_image, contours = frame_difference(frame1, frame2, threshold, min_contour_area)

            # Check if any differences are found
            if contours:
                # Visualize the differences
                if not args.no_visualization:
                    concatenated_frame = visualize_difference(frame1, frame2, diff_image, contours, cap1, fps, frame_scale, border_size, bottom_border_size, text_x_pos, text_y_pos, text_size, text_weight, output_width, output_height, frame_width, frame_height)
                    cv.imshow('Difference Detection', concatenated_frame)

                # Process the contours and update the detected_contour_list
                process_contours(contours, frame_count, total_frame_count, fps, duration_threshold, logger, difference_images_folder, frame1, frame2, frame_scale, border_size, bottom_border_size, text_x_pos, text_y_pos, text_size, text_weight)

                # Write the frame to the output video
                if save_video:
                    write_output_video(concatenated_frame, output_video)

                # Increment the frame count
                frame_count += 1

        # Handle key events to pause, resume, or quit
        if not args.no_visualization:
            key = cv.waitKey(1)
            quit_program, paused = handle_key_events(key, paused)
            if quit_program:
                break

        total_frame_count += 1
        last_frame1 = frame1
        last_frame2 = frame2

    save_remaining_differences(detected_contour_list, frame_count_list, total_frame_count, fps, duration_threshold, logger, difference_images_folder, last_frame1,  last_frame2, frame_scale, border_size, bottom_border_size, text_x_pos, text_y_pos, text_size, text_weight)

    print("Process terminated.")

    
    cap1.release()
    cap2.release()
    if save_video:
        output_video.release()
    if not args.no_visualization:
        cv.destroyAllWindows()

if __name__ == '__main__':
    main() 