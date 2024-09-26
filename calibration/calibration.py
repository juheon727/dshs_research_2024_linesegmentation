from line_segmentation.keypoint.keypoint import find_extremities, SegmentationModel
import os
import cv2
import json
import torch
import numpy as np

def draw_points(img: np.ndarray, keypoints_dict: dict, config: dict, rvec, tvec, camera_matrix, dist_coeffs) -> None:
    """
    Draws multiple categories of keypoints on the image and projects coordinate axes onto the image.

    Parameters:
        img (np.ndarray): The image on which to draw.
        keypoints_dict (dict): A dictionary where keys are category names and values are dictionaries
                               mapping point names to (x, y) tuples.
        config (dict): Configuration dictionary containing drawing parameters.
        rvec (np.ndarray): Rotation vector from camera calibration.
        tvec (np.ndarray): Translation vector from camera calibration.
        camera_matrix (np.ndarray): Camera intrinsic matrix.
        dist_coeffs (np.ndarray): Distortion coefficients.
    """
    # Define colors for each category (BGR format)
    category_colors = {
        'cones': (0, 255, 0),        # Green for cones
        'goal': (255, 0, 0),         # Blue for goal keypoints
        # Add more categories and colors as needed
    }

    # Define font properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    line_type = cv2.LINE_AA

    for category, points in keypoints_dict.items():
        color = category_colors.get(category, (0, 255, 255))  # Default to Yellow if category not found
        for point_name, (x, y) in points.items():
            # Draw a circle at each point
            cv2.circle(img, (x, y), radius=10, color=color, thickness=-1)  # Filled circle

            # Calculate text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(point_name, font, font_scale, thickness)
            # Position the text slightly offset from the point
            text_x = x + 15
            text_y = y - 15
            # Ensure text does not go out of image bounds
            text_x = min(text_x, img.shape[1] - text_width - 10)
            text_y = max(text_y, text_height + 10)

            # Draw a filled rectangle as background for better visibility
            cv2.rectangle(img, 
                          (text_x - 5, text_y - text_height - 5), 
                          (text_x + text_width + 5, text_y + baseline + 5), 
                          (255, 255, 255), 
                          cv2.FILLED)

            # Put the text label on the image
            cv2.putText(img, point_name, (text_x, text_y), font, font_scale, (0, 0, 0), thickness, line_type)

    # Project coordinate axes onto the image
    axis_length = 0.2  # length of the axes
    axis_points = np.float32([
        [0, 0, 0],  # Origin
        [axis_length, 0, 0],  # X-axis
        [0, axis_length, 0],  # Y-axis
        [0, 0, axis_length],   # Z-axis
        [0.00, 17.0, 0.00]
    ])

    # Project 3D points to image plane
    imgpts, jac = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
    print(imgpts)

    # Convert projected points to integer tuples
    origin = tuple(imgpts[0].ravel().astype(int))
    x_axis = tuple(imgpts[1].ravel().astype(int))
    y_axis = tuple(imgpts[2].ravel().astype(int))
    z_axis = tuple(imgpts[3].ravel().astype(int))

    # Draw the axes on the image
    cv2.line(img, origin, x_axis, (0, 0, 255), 3)  # X-axis in Red
    cv2.line(img, origin, y_axis, (0, 255, 0), 3)  # Y-axis in Green
    cv2.line(img, origin, z_axis, (255, 0, 0), 3)  # Z-axis in Blue

    custom_point = tuple(imgpts[4].ravel().astype(int))
    cv2.circle(img, custom_point, radius=10, color=(0, 255, 255), thickness=-1)

    # Label the axes at the end or intersection with image boundary
    def get_boundary_intersection(origin, end, img_width, img_height):
        """
        Calculate the intersection point of the line from origin to end with the image boundary.

        Parameters:
            origin (tuple): (x, y) of the origin.
            end (tuple): (x, y) of the axis end.
            img_width (int): Width of the image.
            img_height (int): Height of the image.

        Returns:
            tuple: (x, y) coordinates of the intersection point.
        """
        x0, y0 = origin
        x1, y1 = end

        # Define image boundaries
        boundaries = [
            ((0, 0), (img_width - 1, 0)),           # Top
            ((img_width - 1, 0), (img_width - 1, img_height - 1)),  # Right
            ((img_width - 1, img_height - 1), (0, img_height - 1)),  # Bottom
            ((0, img_height - 1), (0, 0))            # Left
        ]

        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

        def intersect(A, B, C, D):
            return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

        for boundary in boundaries:
            C, D = boundary
            # Check if lines intersect
            if intersect(origin, end, C, D):
                # Calculate intersection point
                # Line AB represented as a1x + b1y = c1
                a1 = y1 - y0
                b1 = x0 - x1
                c1 = a1 * x0 + b1 * y0

                # Line CD represented as a2x + b2y = c2
                a2 = D[1] - C[1]
                b2 = C[0] - D[0]
                c2 = a2 * C[0] + b2 * C[1]

                determinant = a1 * b2 - a2 * b1
                if determinant == 0:
                    continue  # Lines are parallel
                else:
                    x = int((b2 * c1 - b1 * c2) / determinant)
                    y = int((a1 * c2 - a2 * c1) / determinant)
                    return (x, y)
        # If no intersection, return the end point
        return end

    img_height, img_width = img.shape[:2]

    # Determine label positions
    label_x = get_boundary_intersection(origin, x_axis, img_width, img_height)
    label_y = get_boundary_intersection(origin, y_axis, img_width, img_height)
    label_z = get_boundary_intersection(origin, z_axis, img_width, img_height)

    # Define label offsets to slightly move the label inside the image
    label_offset = 10  # Pixels

    def adjust_label_position(label, intersection, axis_end):
        x, y = intersection
        # Move the label slightly towards the origin to avoid being exactly on the boundary
        if axis_end[0] > origin[0]:
            x -= label_offset
        else:
            x += label_offset
        if axis_end[1] > origin[1]:
            y -= label_offset
        else:
            y += label_offset
        # Ensure the label stays within the image boundaries
        x = max(0, min(x, img_width - 20))
        y = max(20, min(y, img_height - 10))
        return (x, y)

    label_x_pos = adjust_label_position('X', label_x, x_axis)
    label_y_pos = adjust_label_position('Y', label_y, y_axis)
    label_z_pos = adjust_label_position('Z', label_z, z_axis)

    # Put the labels on the image
    cv2.putText(img, 'x', label_x_pos, font, font_scale, (0, 0, 255), thickness, line_type)
    cv2.putText(img, 'y', label_y_pos, font, font_scale, (0, 255, 0), thickness, line_type)
    cv2.putText(img, 'z', label_z_pos, font, font_scale, (255, 0, 0), thickness, line_type)

    # Save the annotated image to disk
    output_dir = config.get('output_image_dir', '/app')
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
    output_filename = 'annotated_image.jpg'  # You can change the filename if desired
    output_path = os.path.join(output_dir, output_filename)

    # Save the annotated image to disk
    if cv2.imwrite(output_path, img):
        print(f"Annotated image saved to {output_path}")
    else:
        raise IOError(f"Failed to save image to {output_path}")

def calibrate_camera(all_keypoints: dict, config: dict, img_shape: tuple):
    """
    Performs camera calibration using the detected keypoints and their corresponding 3D coordinates.

    Parameters:
        all_keypoints (dict): Dictionary of detected keypoints in image with their names.
        config (dict): Configuration dictionary containing the real world coordinates of keypoints.
        img_shape (tuple): Shape of the image (height, width, channels).

    Returns:
        rvec (np.ndarray): Rotation vector.
        tvec (np.ndarray): Translation vector.
        camera_matrix (np.ndarray): Camera intrinsic matrix.
        dist_coeffs (np.ndarray): Distortion coefficients.
    """
    # Extract 3D object points and 2D image points
    obj_points = []  # 3D points in world coordinate
    img_points = []  # 2D points in image plane

    keypoints_3d = config.get('keypoints', {})

    for category, points in all_keypoints.items():
        for point_name, (x, y) in points.items():
            if point_name in keypoints_3d:
                obj_points.append(keypoints_3d[point_name])  # [x, y, z]
                img_points.append([x, y])  # [u, v]

    # Check that we have enough points
    if len(obj_points) < 4:
        raise ValueError("Not enough keypoints for calibration. At least 4 points are required.")

    # Convert to numpy arrays
    obj_points = np.array(obj_points, dtype=np.float32)
    img_points = np.array(img_points, dtype=np.float32)

    # Assume no distortion
    dist_coeffs = np.zeros((5, 1))  # Typically 5 distortion coefficients

    # Initialize camera matrix with approximate values
    fx = fy = 0.5 * img_shape[1]  # Approximate focal length
    center = (img_shape[1] / 2, img_shape[0] / 2)
    camera_matrix = np.array([[fx, 0, center[0]],
                              [0, fy, center[1]],
                              [0, 0, 1]], dtype='double')

    # Use cv2.solvePnP to estimate the rotation and translation vectors
    success, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)

    if not success:
        raise ValueError("Camera calibration failed.")

    return rvec, tvec, camera_matrix, dist_coeffs

def detect_cones(img: np.ndarray, config: dict, mode: str) -> dict:
    cone_detector = torch.hub.load('ultralytics/yolov5', 'custom', path=config['conedetector_weight_dir'], force_reload=False)
    img_resized = cv2.resize(img, (config['conedetector_resolution'], config['conedetector_resolution']))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    cone_positions = cone_detector(img_rgb).pandas().xyxy[0]
    points = []
    for index, row in cone_positions.iterrows():
        x = int((row['xmin']*img.shape[1]/config['conedetector_resolution'] + row['xmax']*img.shape[1]/config['conedetector_resolution']) / 2)
        y = int(row['ymax']*img.shape[0]/config['conedetector_resolution'])
        points.append((x, y))
    assert len(points) == 2
    points.sort(key=lambda point: point[0])
    point_names = ['cl1', 'cl2'] if mode == 'right' else ['cr2', 'cr1']
    return dict(zip(point_names, points))

def process_goal_extremities(goal_extremities: dict, img_shape: tuple) -> dict:
    ret = {}

    line_horizontal = goal_extremities['line_horizontal']
    line_vertical_left = goal_extremities['line_vertical_left']
    line_vertical_right = goal_extremities['line_vertical_right']

    al, ar = line_horizontal[1], line_horizontal[0]
    l1, l2 = line_vertical_left[1], line_vertical_left[0]
    r1, r2 = line_vertical_right[1], line_vertical_right[0]
    if line_horizontal[0]['x'] < line_horizontal[1]['x']:
        al, ar = line_horizontal[0], line_horizontal[1]
    if line_vertical_left[0]['y'] > line_vertical_left[1]['y']:
        l1, l2 = line_vertical_left[0], line_vertical_left[1]
    if line_vertical_right[0]['y'] > line_vertical_right[1]['y']:
        r1, r2 = line_vertical_right[0], line_vertical_right[1]

    ret['gl2'] = (int((al['x'] + l2['x'])*img_shape[1]/2), int((al['y'] + l2['y'])*img_shape[0]/2)) # gl2
    ret['gr2'] = (int((ar['x'] + r2['x'])*img_shape[1]/2), int((ar['y'] + r2['y'])*img_shape[0]/2)) # gr2
    ret['gl1'] = (int((l1['x'])*img_shape[1]), int((l1['y'])*img_shape[0])) # gl1
    ret['gr1'] = (int((r1['x'])*img_shape[1]), int((r1['y'])*img_shape[0])) # gr1

    return ret

def detect_keypoints(img: np.ndarray = None):
    config = json.load(open('/app/config.json'))
    keypoint_config = json.load(open(os.path.join(config['line_segmentation_dir'], 'keypoint/config.json')))
    goal_segnet = SegmentationModel(config=keypoint_config, img_resolution=keypoint_config['img_resolution'])
    if img is None:
        img = cv2.imread(config['img_dir'])
    goal_extremities = find_extremities(calib_net=goal_segnet, image=img)
    goal_keypoints = process_goal_extremities(goal_extremities=goal_extremities, img_shape=img.shape)
    cone_positions = detect_cones(img, config, 'right')
    all_keypoints = {
        'cones': cone_positions,
        'goal': goal_keypoints
    }
    # Perform camera calibration
    rvec, tvec, camera_matrix, dist_coeffs = calibrate_camera(all_keypoints, config, img.shape)
    # Draw keypoints and projected lines
    draw_points(img, all_keypoints, config, rvec, tvec, camera_matrix, dist_coeffs)

if __name__ == '__main__':
    detect_keypoints()
