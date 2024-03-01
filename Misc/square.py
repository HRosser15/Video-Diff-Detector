import cv2 as cv
import numpy as np

def add_filled_square(input_image_path, output_image_path):
    # Read the input image
    image = cv.imread(input_image_path)

    if image is None:
        print("Error: Could not read the input image.")
        return

    # Define the coordinates and size of the square
    x, y = 100, 100
    square_size = 25

    # Create an image filled with dark green color (BGR format)
    green = (0, 200, 0)
    square = np.full((square_size, square_size, 3), green, dtype=np.uint8)

    # Add the square to the input image at specified coordinates
    image[y:y + square_size, x:x + square_size] = square

    # Save the modified image
    cv.imwrite(output_image_path, image)

    print("Modified image saved as '{}'.".format(output_image_path))

if __name__ == "__main__":
    # Specify the path to the input image
    input_image_path = '../Photos/screenshot2.jpg'  # Replace with the actual path

    # Specify the path for the output image
    output_image_path = 'edit2.jpg'  # Replace with the desired output path

    # Call the function to add a filled square to the image
    add_filled_square(input_image_path, output_image_path)
