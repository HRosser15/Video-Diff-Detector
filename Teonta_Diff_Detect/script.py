import sys
import cv2

def calculate_frame_difference(video_path1, video_path2, output_video_path, bounding_box_margin=5, min_contour_area=100):
    # Open the videos
    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)

    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: Could not open videos.")
        return

    # Get video properties
    frame_count1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))

    # Determine the minimum number of frames
    min_frame_count = min(frame_count1, frame_count2)

    # Get video properties for output
    frame_width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap1.get(cv2.CAP_PROP_FPS))

    # Create video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width*2, frame_height))

    # Loop through each frame and calculate the absolute difference
    for i in range(min_frame_count):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            print("Error: Failed to read frames.")
            break

        # Convert frames to grayscale for better comparison
        gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Calculate absolute difference between frames
        frame_difference = cv2.absdiff(gray_frame1, gray_frame2)

        # Threshold the difference frame
        _, thresholded = cv2.threshold(frame_difference, 30, 255, cv2.THRESH_BINARY)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding boxes around the changes with added margin
        bounding_boxes_drawn = False
        merged_contours = []
        for contour in contours:
            # Calculate contour area
            contour_area = cv2.contourArea(contour)
            if contour_area < min_contour_area:
                continue  # Skip contours with area below threshold

            x, y, w, h = cv2.boundingRect(contour)

            # Add a margin to the bounding box dimensions
            x -= bounding_box_margin
            y -= bounding_box_margin
            w += 2 * bounding_box_margin
            h += 2 * bounding_box_margin

            # Check if the current bounding box overlaps with any existing ones
            merged = False
            for i, (mx, my, mw, mh) in enumerate(merged_contours):
                if x < mx + mw and x + w > mx and y < my + mh and y + h > my:
                    # If overlap is found, merge the bounding boxes
                    merged_contours[i] = (min(x, mx), min(y, my), max(x + w, mx + mw) - min(x, mx), max(y + h, my + mh) - min(y, my))
                    merged = True
                    break

            if not merged:
                # If no overlap, add the current bounding box to the list of merged contours
                merged_contours.append((x, y, w, h))

        # Draw the merged bounding boxes
        for x, y, w, h in merged_contours:
            cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 0, 255), 2)
            bounding_boxes_drawn = True

        # Concatenate frame1 and frame2 horizontally
        combined_frame = cv2.hconcat([frame1, frame2])

        # Write the frame with bounding boxes to the output video
        out.write(combined_frame)

        # Display the combined frame
        cv2.imshow('Output Video', combined_frame)
        cv2.waitKey(1)  # Wait for a key press

    # Release the video capture objects
    cap1.release()
    cap2.release()

    # Release the video writer object
    out.release()

    # Close OpenCV windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Check if the correct number of arguments are provided
    if len(sys.argv) != 6:
        print(len(sys.argv))
        print("Usage: python script.py video_path1 video_path2 output_video_path bounding_box_margin min_contour_area")
        sys.exit(1)

    # Get the command-line arguments
    video_path1 = sys.argv[1]
    video_path2 = sys.argv[2]
    output_video_path = sys.argv[3]
    bounding_box_margin = int(sys.argv[4])
    min_contour_area = int(sys.argv[5])

    # Call the function to calculate frame difference
    calculate_frame_difference(video_path1, video_path2, output_video_path, bounding_box_margin, min_contour_area)
