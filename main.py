import cv2
import cv2 as cv
import numpy as np
import imutils
import glob
import os

# If true, display images (rescale image down with a factor of 'scale')
image_debug = False

# Scale down image for debug with a factor of 'scale'
scale = 5

# If true, display information and draw contours
draw_debug = False

# If true, write files
write_files = True

# Max distinctive colors for task 2
colors = {
    '0': (255, 255, 255),
    '1': (0, 100, 0),
    '2': (188, 143, 143),
    '3': (255, 0, 0),
    '4': (255, 215, 0),
    '5': (0, 255, 0),
    '6': (65, 105, 225),
    '7': (0, 255, 255),
    '8': (0, 0, 255),
    '9': (255, 20, 147)
}


# Task number 1
def task1():

    # Local paths
    images_path = ".\\antrenare\\clasic\\"
    destination_path = ".\\results\\Constantinescu_Andrei-Eduard_344\\clasic\\"

    # Count how many images we processed
    images_counter = 1

    # Iterate each image
    for image_path in glob.glob(images_path + '*.jpg'):

        # Debug single image
        if image_debug:
            debug_image_number = 1
            if str(debug_image_number) not in image_path:
                continue

        # Read image
        image = cv.imread(image_path)
        file_name = str(images_counter) + "_predicted.txt"
        images_counter += 1

        # Image padding
        image_padding_horizontal = 100
        image_padding_vertical = 0

        # Expand canvas to add padding to image
        old_image_height, old_image_width, channels = image.shape

        # New size of padded image
        new_image_width = old_image_width + image_padding_horizontal
        new_image_height = old_image_height + image_padding_vertical

        # Create new array for padded image
        padded_image = np.full((new_image_height, new_image_width, channels), (200, 200, 200), dtype=np.uint8)

        # Calculate the center of the padded image
        x_center = (new_image_width - old_image_width) // 2
        y_center = (new_image_height - old_image_height) // 2

        # Paste original image into the center
        padded_image[y_center:y_center + old_image_height, x_center:x_center + old_image_width] = image

        # Gray, blur and threshold padded image
        grayed_image = cv.cvtColor(padded_image, cv.COLOR_BGR2GRAY)
        blurred_image = cv.GaussianBlur(grayed_image, (15, 15), 6)
        thresholded_image = cv.adaptiveThreshold(blurred_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 33, 4)
        thresholded_image = cv.bitwise_not(thresholded_image)

        # Get contours
        contours = cv.findContours(thresholded_image.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv.contourArea, reverse=True)

        # If we find a sudoku square save it in sudoku_contour
        sudoku_contour = None

        # Iterate through contours
        for c in contours:

            # Convex Hull
            epsilon = 0.02 * cv.arcLength(c, True)
            approx = cv.approxPolyDP(c, epsilon, True)

            # Find the bounding rectangle of contour to check its size
            x, y, w, h = cv.boundingRect(c)

            # Draw contour if square and if size of box is higher than threshold (so that text cannot be picked up)
            if len(approx) == 4 and w * h > 500000:

                # Draw the contour and display bounding square size
                if draw_debug:
                    cv.putText(padded_image, f'Box size: {str(w * h)} pixels', (15, 60), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                    cv.drawContours(padded_image, [approx], -1, (0, 255, 0), 4)

                # Save the contour as the sudoku_contour
                sudoku_contour = approx
                break

        # If we found a sudoku contour then proceed to wrap image so that it contains only the sudoku contour
        if sudoku_contour is not None:

            # Order points from contour
            rect = np.zeros((4, 2), dtype='float32')
            sudoku_contour_reshaped = sudoku_contour.reshape(4, 2)

            # Calculate the sum and difference of x and y of each corner
            points_sum = sudoku_contour_reshaped.sum(axis=1)
            points_diff = np.diff(sudoku_contour_reshaped, axis=1)

            # First element will be top left and third will be bottom right (minimum sum and maximum sum)
            rect[0] = sudoku_contour_reshaped[np.argmin(points_sum)]
            rect[2] = sudoku_contour_reshaped[np.argmax(points_sum)]

            # Second element will be top right and last will be bottom left (minimum and maximum diff)
            rect[1] = sudoku_contour_reshaped[np.argmin(points_diff)]
            rect[3] = sudoku_contour_reshaped[np.argmax(points_diff)]

            # Calculate the width of the new reshaped image
            width_bottom = np.sqrt(((rect[2][0] - rect[3][0]) ** 2) + ((rect[2][1] - rect[3][1]) ** 2))
            width_top = np.sqrt(((rect[1][0] - rect[0][0]) ** 2) + ((rect[1][1] - rect[0][1]) ** 2))
            width_max = max(int(width_top), int(width_bottom))

            # Calculate the height of the new reshaped image
            height_right = np.sqrt(((rect[1][0] - rect[2][0]) ** 2) + ((rect[1][1] - rect[2][1]) ** 2))
            height_left = np.sqrt(((rect[0][0] - rect[3][0]) ** 2) + ((rect[0][1] - rect[3][1]) ** 2))
            height_max = max(int(height_left), int(height_right))

            # Put text in each corner of the sudoku box
            if image_debug:
                if draw_debug:
                    for i, r in enumerate(rect):
                        cv.putText(padded_image, str(i), (int(r[0]), int(r[1])), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

                # Draw image before transformation
                dims = (padded_image.shape[1] // scale, padded_image.shape[0] // scale)
                cv.imshow('image', cv.resize(padded_image, dims))

            # Construct the size of the new image and save it in a matrix
            sudoku_matrix_template = np.array([[0, 0], [width_max - 1, 0], [width_max - 1, height_max - 1], [0, height_max - 1]], dtype='float32')
            perspective_transform = cv.getPerspectiveTransform(rect, sudoku_matrix_template)
            sudoku_contour_warped = cv.warpPerspective(padded_image, perspective_transform, (width_max, height_max))

            # Calculate step size for each cell
            width_step = sudoku_contour_warped.shape[1] // 9
            height_step = sudoku_contour_warped.shape[0] // 9

            # Array to hold each cell upper left corner coord
            coords = []

            # Calculate the upper left coord of each cell
            for c in range(0, 81):
                coord = ((c % 9) * width_step, (c // 9 * height_step))
                coords.append(coord)

                if draw_debug:
                    sudoku_contour_warped = cv.circle(sudoku_contour_warped, coord, 12, (0, 0, 255), -1)

            # Array to hold indices of cells that contain numbers
            cells_with_numbers = []

            for i, coord in enumerate(coords):

                # Add padding to remove borders
                padding = 40
                cell_mean_bias = 10

                cell = sudoku_contour_warped[coord[1] + padding:coord[1] + height_step - padding, coord[0] + padding:coord[0] + width_step - padding].copy()
                cell_grayed = cv.cvtColor(cell, cv.COLOR_BGR2GRAY)
                cell_threshold = cv.threshold(cell_grayed, 145, 255, cv.THRESH_BINARY_INV)[1]

                # If there is something inside the cell (if the mean of the cell is higher than the cell_mean_bias) append to the final array
                if cell_threshold.mean() > cell_mean_bias:
                    cells_with_numbers.append(i)

                # Display some cells if debug
                if image_debug:
                    number_of_cells = 1
                    if i + 81 - number_of_cells < len(coords):
                        cv.imshow('cell' + str(i), cell_threshold)

            # Generate final answer array
            answer = []
            for i in range(81):
                if i in cells_with_numbers:
                    answer.append('x')
                else:
                    answer.append('o')

            # Create folder if not exists
            if write_files:
                if not os.path.exists(destination_path):
                    os.makedirs(destination_path)

                # Save answer and create file
                with open(destination_path + file_name, 'w+') as file:
                    for i, val in enumerate(answer):
                        file.write(val)
                        if (i + 1) % 9 == 0 and i < len(answer) - 1:
                            file.write('\n')

            # Display the warped image
            if image_debug:
                sudoku_dims = (sudoku_contour_warped.shape[1] // scale, sudoku_contour_warped.shape[0] // scale)
                cv.imshow('warped', cv.resize(sudoku_contour_warped, sudoku_dims))

        else:
            print(f"Could not find sudoku in image with name {image_path}!")

        if image_debug:
            cv.waitKey(0)
            return  # Display only one image


# Task number 2
def task2():
    # Local paths
    images_path = ".\\antrenare\\jigsaw\\"
    destination_path = ".\\results\\Constantinescu_Andrei-Eduard_344\\jigsaw\\"

    # Count how many images we processed
    images_counter = 1

    # Iterate each image
    for image_path in glob.glob(images_path + '*.jpg'):

        # Debug single image
        if image_debug:
            debug_image_number = 20
            if str(debug_image_number) not in image_path:
                continue

        # Image padding
        image_padding_horizontal = 100
        image_padding_vertical = 0

        # Read image
        image = cv.imread(image_path)
        file_name = str(images_counter) + "_predicted.txt"
        images_counter += 1

        # Expand canvas to add padding to image
        old_image_height, old_image_width, channels = image.shape

        # New size of padded image
        new_image_width = old_image_width + image_padding_horizontal
        new_image_height = old_image_height + image_padding_vertical

        # Create new array for padded image
        padded_image = np.full((new_image_height, new_image_width, channels), (200, 200, 200), dtype=np.uint8)

        # Calculate the center of the padded image
        x_center = (new_image_width - old_image_width) // 2
        y_center = (new_image_height - old_image_height) // 2

        # Paste original image into the center
        padded_image[y_center:y_center + old_image_height, x_center:x_center + old_image_width] = image

        # Gray, blur and threshold padded image
        grayed_image = cv.cvtColor(padded_image, cv.COLOR_BGR2GRAY)
        blurred_image = cv.GaussianBlur(grayed_image, (15, 15), 6)
        thresholded_image = cv.adaptiveThreshold(blurred_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 33, 4)
        thresholded_image = cv.bitwise_not(thresholded_image)

        # Get contours
        contours = cv.findContours(thresholded_image.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv.contourArea, reverse=True)

        # If we find a sudoku square save it in sudoku_contour
        sudoku_contour = None

        # Iterate through contours
        for c in contours:

            # Convex Hull
            epsilon = 0.02 * cv.arcLength(c, True)
            approx = cv.approxPolyDP(c, epsilon, True)

            # Find the bounding rectangle of contour to check its size
            x, y, w, h = cv.boundingRect(c)

            # Draw contour if square and if size of box is higher than threshold (so that text cannot be picked up)
            if len(approx) == 4 and 10000000 > w * h > 500000:

                # Draw the contour and display bounding square size
                if draw_debug:
                    cv.putText(padded_image, f'Box size: {str(w * h)} pixels', (15, 60), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                    cv.drawContours(padded_image, [approx], -1, (0, 255, 0), 4)

                # Save the contour as the sudoku_contour
                sudoku_contour = approx
                break

        # If we found a sudoku contour then proceed to wrap image so that it contains only the sudoku contour
        if sudoku_contour is not None:

            # Order points from contour
            rect = np.zeros((4, 2), dtype='float32')
            sudoku_contour_reshaped = sudoku_contour.reshape(4, 2)

            # Calculate the sum and difference of x and y of each corner
            points_sum = sudoku_contour_reshaped.sum(axis=1)
            points_diff = np.diff(sudoku_contour_reshaped, axis=1)

            # First element will be top left and third will be bottom right (minimum sum and maximum sum)
            rect[0] = sudoku_contour_reshaped[np.argmin(points_sum)]
            rect[2] = sudoku_contour_reshaped[np.argmax(points_sum)]

            # Second element will be top right and last will be bottom left (minimum and maximum diff)
            rect[1] = sudoku_contour_reshaped[np.argmin(points_diff)]
            rect[3] = sudoku_contour_reshaped[np.argmax(points_diff)]

            # Calculate the width of the new reshaped image
            width_bottom = np.sqrt(((rect[2][0] - rect[3][0]) ** 2) + ((rect[2][1] - rect[3][1]) ** 2))
            width_top = np.sqrt(((rect[1][0] - rect[0][0]) ** 2) + ((rect[1][1] - rect[0][1]) ** 2))
            width_max = max(int(width_top), int(width_bottom))

            # Calculate the height of the new reshaped image
            height_right = np.sqrt(((rect[1][0] - rect[2][0]) ** 2) + ((rect[1][1] - rect[2][1]) ** 2))
            height_left = np.sqrt(((rect[0][0] - rect[3][0]) ** 2) + ((rect[0][1] - rect[3][1]) ** 2))
            height_max = max(int(height_left), int(height_right))

            # Put text in each corner of the sudoku box
            if image_debug:
                if draw_debug:
                    for i, r in enumerate(rect):
                        cv.putText(padded_image, str(i), (int(r[0]), int(r[1])), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

                # Draw image before transformation
                dims = (padded_image.shape[1] // scale, padded_image.shape[0] // scale)
                cv.imshow('image', cv.resize(padded_image, dims))

            # Construct the size of the new image and save it in a matrix
            sudoku_matrix_template = np.array([[0, 0], [width_max - 1, 0], [width_max - 1, height_max - 1], [0, height_max - 1]], dtype='float32')
            perspective_transform = cv.getPerspectiveTransform(rect, sudoku_matrix_template)
            sudoku_contour_warped = cv.warpPerspective(padded_image, perspective_transform, (width_max, height_max))

            # Gray and blur image
            sudoku_grayed_image = cv.cvtColor(sudoku_contour_warped, cv.COLOR_BGR2GRAY)
            sudoku_blurred_image = cv.GaussianBlur(sudoku_grayed_image, (5, 5), 3)

            # Do opening of image (erode then dilate) to remove thin lines and keep the thick ones
            sudoku_kernel_erode = np.ones((19, 19), np.uint8)
            T, sudoku_thresholded_image = cv.threshold(sudoku_blurred_image, 80, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
            sudoku_opened = cv.morphologyEx(sudoku_thresholded_image, cv2.MORPH_OPEN, sudoku_kernel_erode)

            # Invert the opened image and convert it to rgb
            sudoku_opened = cv.bitwise_not(sudoku_opened)

            # Draw border around sudoku table to prevent small gaps
            border_size = 30
            top_left = (border_size // 2, border_size // 2)
            bottom_right = (sudoku_opened.shape[1] - border_size // 2, sudoku_opened.shape[0] - border_size // 2)
            sudoku_opened = cv.rectangle(sudoku_opened, top_left, bottom_right, (0, 0, 0), border_size)

            # Get contours
            contours = cv.findContours(sudoku_opened.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(contours)

            # Convert image to rgb to color it
            sudoku_opened = cv.cvtColor(sudoku_opened, cv2.COLOR_GRAY2RGB)

            # Iterate through different contours to fill white
            for number, c in enumerate(contours):

                # Convex Hull
                epsilon = 0.00002 * cv.arcLength(c, True)
                approx = cv.approxPolyDP(c, epsilon, True)

                # Fill inside of contour with white to create a canvas
                cv.drawContours(sudoku_opened, [approx], -1, (255, 255, 255), cv.FILLED)

                if draw_debug:

                    # Get center of contour
                    M = cv.moments(c)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])

                    # Put contour number
                    cv.putText(sudoku_opened, str(number + 1), (cX, cY), cv.FONT_HERSHEY_SIMPLEX, 7, (0, 255, 0), 20)

            # Display the color-zone image
            if image_debug:
                sudoku_dims = (sudoku_opened.shape[1] // scale, sudoku_opened.shape[0] // scale)
                cv.imshow('zoned', cv.resize(sudoku_opened, sudoku_dims))

            # Calculate step size for each cell
            width_step = sudoku_contour_warped.shape[1] // 9
            height_step = sudoku_contour_warped.shape[0] // 9

            # Array to hold each cell upper left corner coord
            coords = []

            # Calculate the upper left coord of each cell
            for c in range(0, 81):
                coord = ((c % 9) * width_step, (c // 9 * height_step))
                coords.append(coord)

                if draw_debug:
                    sudoku_contour_warped = cv.circle(sudoku_contour_warped, coord, 12, (0, 0, 255), -1)

            # Iterate through each cell and verify if it is colored, if not, color the whole contour that contains the cell
            current_zone = 1
            for coord in coords:

                padding = 100
                cell = sudoku_opened[coord[1] + padding:coord[1] + height_step - padding, coord[0] + padding:coord[0] + width_step - padding].copy()
                average_color = cv.mean(cell)[:3]

                # Use epsilon to check for small errors between colors
                color_epsilon = (5, 5, 5)

                # Check if cell is not colored
                if abs(average_color[0] - colors['0'][0]) < color_epsilon[0] and abs(average_color[1] - colors['0'][1]) < color_epsilon[1] and abs(average_color[2] - colors['0'][2]) < color_epsilon[2]:
                    # Iterate through each contour
                    for c in contours:
                        # Check if cell is inside the current contour
                        if cv.pointPolygonTest(c, (coord[0] + padding, coord[1] + padding), False) > 0:
                            # Color the zone according to its id
                            cv.drawContours(sudoku_opened, [c], -1, colors[str(current_zone)], cv.FILLED)
                            current_zone += 1
                            break

            # Display the true color-zone image
            if image_debug:
                sudoku_dims = (sudoku_opened.shape[1] // scale, sudoku_opened.shape[0] // scale)
                cv.imshow('true zoned', cv.resize(sudoku_opened, sudoku_dims))

            # Array to hold each cell color-zone
            cells_to_zone = []

            # Assign color-zone to each cell based on sudoku_opened colors
            for i, coord in enumerate(coords):

                # Add padding to remove borders
                padding = 100

                cell = sudoku_opened[coord[1] + padding:coord[1] + height_step - padding, coord[0] + padding:coord[0] + width_step - padding].copy()
                average_color = cv.mean(cell)[:3]

                # Use epsilon to check for small errors between colors
                color_epsilon = (5, 5, 5)

                # Iterate through each color
                for color in colors.values():
                    # If average color is close to a defined color
                    if abs(average_color[0] - color[0]) < color_epsilon[0] and abs(average_color[1] - color[1]) < color_epsilon[1] and abs(average_color[2] - color[2]) < color_epsilon[2]:
                        cells_to_zone.append(list(colors.keys())[list(colors.values()).index(color)])

            # Array to hold indices of cells that contain numbers
            cells_with_numbers = []

            # Check if cell contains number
            for i, coord in enumerate(coords):

                # Add padding to remove borders
                padding = 40
                cell_mean_bias = 10

                cell = sudoku_contour_warped[coord[1] + padding:coord[1] + height_step - padding, coord[0] + padding:coord[0] + width_step - padding].copy()
                cell_grayed = cv.cvtColor(cell, cv.COLOR_BGR2GRAY)
                cell_threshold = cv.threshold(cell_grayed, 145, 255, cv.THRESH_BINARY_INV)[1]

                # If there is something inside the cell (if the mean of the cell is higher than the cell_mean_bias) append to the final array
                if cell_threshold.mean() > cell_mean_bias:
                    cells_with_numbers.append(i)

            # Generate final answer array
            answer = []
            for i in range(81):
                answer.append(cells_to_zone[i])
                if i in cells_with_numbers:
                    answer.append('x')
                else:
                    answer.append('o')

            if write_files:
                # Create folder if not exists
                if not os.path.exists(destination_path):
                    os.makedirs(destination_path)

                # Save answer and create file
                with open(destination_path + file_name, 'w+') as file:
                    for i, val in enumerate(answer):
                        file.write(val)
                        if (i + 1) % 18 == 0 and i < len(answer) - 1:
                            file.write('\n')

            # Display the warped image
            if image_debug:
                sudoku_dims = (sudoku_contour_warped.shape[1] // scale, sudoku_contour_warped.shape[0] // scale)
                cv.imshow('warped', cv.resize(sudoku_contour_warped, sudoku_dims))

        else:
            print(f"Could not find sudoku in image with name {image_path}!")

        if image_debug:
            cv.waitKey(0)
            return  # Display only one image


if __name__ == "__main__":
    # task1()
    # task2()
    pass
