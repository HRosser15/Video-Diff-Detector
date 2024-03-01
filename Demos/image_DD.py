import cv2 as cv
import numpy as np

def compare_images(image1, image2):
    # Convert images to grayscale for simplicity
    gray1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)

    # Display the original images
    cv.imshow('Image 1', gray1)
    cv.imshow('Image 2', gray2)
    key = cv.waitKey(0)

    # Save the original images if 's' is pressed
    if key == ord('s'):
        cv.imwrite('image1.png', gray1)
        cv.imwrite('image2.png', gray2)

    # Find pixel-wise differences in the two images
    diff = cv.absdiff(gray1, gray2)
    cv.imshow('Difference', diff)
    key = cv.waitKey(0)

    # Save the difference image if 's' is pressed
    if key == ord('s'):
        cv.imwrite('difference.png', diff)

    # Create a binary image, setting pixels above a threshold to white and below to black
    _, thresholded_diff = cv.threshold(diff, 65, 255, cv.THRESH_BINARY)
    cv.imshow('Thresholded Difference', thresholded_diff)
    key = cv.waitKey(0)

    # Save the thresholded difference image if 's' is pressed
    if key == ord('s'):
        cv.imwrite('thresholded_difference.png', thresholded_diff)

    # Wait for 'q' to close the windows
    cv.waitKey(0)
    cv.destroyAllWindows()

# Replace the image paths with your own file paths
image_path1 = '../Photos/edit1.jpg'
image_path2 = '../Photos/edit2.jpg'

# Read the images
image1 = cv.imread(image_path1)
image2 = cv.imread(image_path2)

# Check if the images are valid
if image1 is not None and image2 is not None:
    compare_images(image1, image2)
else:
    print("Error: One or both images could not be loaded.")
