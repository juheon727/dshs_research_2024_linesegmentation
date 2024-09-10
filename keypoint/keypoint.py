import argparse
import copy
import json
import os.path
import random
from collections import deque
from pathlib import Path
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2 as cv
import numpy as np
import torch
import torch.backends.cudnn
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from unet import UNet

def generate_class_synthesis(semantic_mask, radius, mask_colors, erosion=True):
    """
    This function selects for each class present in the semantic mask, a set of circles that cover most of the semantic
    class blobs.
    :param semantic_mask: a image containing the segmentation predictions
    :param radius: circle radius
    :return: a dictionary which associates with each class detected a list of points ( the circles centers)
    """
    buckets = dict()
    if erosion:
        kernel = np.ones((2, 2), np.uint8)
        semantic_mask = cv.erode(semantic_mask, kernel, iterations=1)
    for k, (class_name, _) in enumerate(mask_colors.items()):
        if class_name == 'background':
            continue
        mask = semantic_mask == k
        if mask.sum() > 0:
            disk_list = synthesize_mask(mask, radius)
            if len(disk_list):
                buckets[class_name] = disk_list

    return buckets


def join_points(point_list, maxdist):
    """
    Given a list of points that were extracted from the blobs belonging to a same semantic class, this function creates
    polylines by linking close points together if their distance is below the maxdist threshold.
    :param point_list: List of points of the same line class
    :param maxdist: minimal distance between two polylines.
    :return: a list of polylines
    """
    polylines = []

    if not len(point_list):
        return polylines
    head = point_list[0]
    tail = point_list[0]
    polyline = deque()
    polyline.append(point_list[0])
    remaining_points = copy.deepcopy(point_list[1:])

    while len(remaining_points) > 0:
        min_dist_tail = 1000
        min_dist_head = 1000
        best_head = -1
        best_tail = -1
        for j, point in enumerate(remaining_points):
            dist_tail = np.sqrt(np.sum(np.square(point - tail)))
            dist_head = np.sqrt(np.sum(np.square(point - head)))
            if dist_tail < min_dist_tail:
                min_dist_tail = dist_tail
                best_tail = j
            if dist_head < min_dist_head:
                min_dist_head = dist_head
                best_head = j

        if min_dist_head <= min_dist_tail and min_dist_head < maxdist:
            polyline.appendleft(remaining_points[best_head])
            head = polyline[0]
            remaining_points.pop(best_head)
        elif min_dist_tail < min_dist_head and min_dist_tail < maxdist:
            polyline.append(remaining_points[best_tail])
            tail = polyline[-1]
            remaining_points.pop(best_tail)
        else:
            polylines.append(list(polyline.copy()))
            head = remaining_points[0]
            tail = remaining_points[0]
            polyline = deque()
            polyline.append(head)
            remaining_points.pop(0)
    polylines.append(list(polyline))
    return polylines


def get_line_extremities(buckets, maxdist, width, height):
    """
    Given the dictionary {lines_class: points}, finds plausible extremities of each line, i.e the extremities
    of the longest polyline that can be built on the class blobs,  and normalize its coordinates
    by the image size.
    :param buckets: The dictionary associating line classes to the set of circle centers that covers best the class
    prediction blobs in the segmentation mask
    :param maxdist: the maximal distance between two circle centers belonging to the same blob (heuristic)
    :param width: image width
    :param height: image height
    :return: a dictionary associating to each class its extremities
    """
    extremities = dict()
    for class_name, disks_list in buckets.items():
        polyline_list = join_points(disks_list, maxdist)
        max_len = 0
        longest_polyline = []
        for polyline in polyline_list:
            if len(polyline) > max_len:
                max_len = len(polyline)
                longest_polyline = polyline
        extremities[class_name] = [
            {'x': longest_polyline[0][1] / width, 'y': longest_polyline[0][0] / height},
            {'x': longest_polyline[-1][1] / width, 'y': longest_polyline[-1][0] / height}
        ]
    return extremities


def get_support_center(mask, start, disk_radius, min_support=0.1):
    """
    Returns the barycenter of the True pixels under the area of the mask delimited by the circle of center start and
    radius of disk_radius pixels.
    :param mask: Boolean mask
    :param start: A point located on a true pixel of the mask
    :param disk_radius: the radius of the circles
    :param min_support: proportion of the area under the circle area that should be True in order to get enough support
    :return: A boolean indicating if there is enough support in the circle area, the barycenter of the True pixels under
     the circle
    """
    x = int(start[0])
    y = int(start[1])
    support_pixels = 1
    result = [x, y]
    xstart = x - disk_radius
    if xstart < 0:
        xstart = 0
    xend = x + disk_radius
    if xend > mask.shape[0]:
        xend = mask.shape[0] - 1

    ystart = y - disk_radius
    if ystart < 0:
        ystart = 0
    yend = y + disk_radius
    if yend > mask.shape[1]:
        yend = mask.shape[1] - 1

    for i in range(xstart, xend + 1):
        for j in range(ystart, yend + 1):
            dist = np.sqrt(np.square(x - i) + np.square(y - j))
            if dist < disk_radius and mask[i, j] > 0:
                support_pixels += 1
                result[0] += i
                result[1] += j
    support = True
    if support_pixels < min_support * np.square(disk_radius) * np.pi:
        support = False

    result = np.array(result)
    result = np.true_divide(result, support_pixels)

    return support, result


def synthesize_mask(semantic_mask, disk_radius):
    """
    Fits circles on the True pixels of the mask and returns those which have enough support : meaning that the
    proportion of the area of the circle covering True pixels is higher that a certain threshold in order to avoid
    fitting circles on alone pixels.
    :param semantic_mask: boolean mask
    :param disk_radius: radius of the circles
    :return: a list of disk centers, that have enough support
    """
    mask = semantic_mask.copy().astype(np.uint8)
    points = np.transpose(np.nonzero(mask))
    disks = []
    while len(points):

        start = random.choice(points)
        dist = 10.
        success = True
        while dist > 1.:
            enough_support, center = get_support_center(mask, start, disk_radius)
            if not enough_support:
                bad_point = np.round(center).astype(np.int32)
                cv.circle(mask, (bad_point[1], bad_point[0]), disk_radius, (0), -1)
                success = False
            dist = np.sqrt(np.sum(np.square(center - start)))
            start = center
        if success:
            disks.append(np.round(start).astype(np.int32))
            cv.circle(mask, (disks[-1][1], disks[-1][0]), disk_radius, 0, -1)
        points = np.transpose(np.nonzero(mask))

    return disks
    
class SegmentationModel:
    def __init__(self, img_resolution=448):
        self.model = UNet()
        self.model.load_state_dict(torch.load(os.path.join(config['unet_dir'], 'weights/unet_best.pt'), weights_only=True))
        self.img_resolution = img_resolution
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def inference(self, image):
        img = cv.resize(image, (self.img_resolution, self.img_resolution), interpolation=cv.INTER_LINEAR)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = np.asarray(img, np.float32) / 255.
        img = (img - img.mean()) / img.std()
        img = img.transpose((2, 0, 1))
        with torch.no_grad():
            img = torch.from_numpy(img).to(self.device).unsqueeze(0)
            output = self.model(img)[0].cpu().numpy()
            output = output.transpose(1, 2, 0)
            output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
        return output

def visualize_keypoints_on_image(image, extremities, color=(0, 255, 0)):
    # Draw keypoints on the image
    color_bgr = (color[2], color[1], color[0])
    for class_name, points in extremities.items():
        for point in points:
            x = int(point['x'] * image.shape[1])
            y = int(point['y'] * image.shape[0])
            cv.circle(image, (x, y), 5, color_bgr, -1)
    return image

def overlay_mask_on_image(image, mask, mask_colors, img_resolution):
    # Convert mask to RGB using mask_colors and overlay on the original image
    r_image = cv.resize(image, (img_resolution, img_resolution), cv.INTER_LINEAR)
    overlay = np.zeros((img_resolution, img_resolution, 3), dtype=np.uint8)
    for k, (class_name, color) in enumerate(mask_colors.items()):
        if class_name == 'background':
            continue
        overlay[mask == k] = color
    overlay = cv.cvtColor(overlay, cv.COLOR_RGB2BGR)
    overlay_image = 0.3*np.array(r_image, dtype=np.uint8) + 0.7*np.array(overlay, dtype=np.uint8)
    return overlay_image

if __name__ == "__main__":
    config = json.load(open('/app/keypoint/config.json'))
    mask_colors = json.load(open(os.path.join(config['data_dir'], 'mask_colors.json')))

    lines_palette = []
    for line_class, color in mask_colors.items():
        lines_palette.extend(color)

    calib_net = SegmentationModel(img_resolution=config['img_resolution'])

    dataset_dir = os.path.join(config['data_dir'], config['task'])
    print(dataset_dir)
    if not os.path.exists(dataset_dir):
        print("Invalid dataset path !")
        exit(-1)

    with tqdm(range(len(os.listdir(os.path.join(dataset_dir, 'images'))))) as t:
        for i in t:
            output_prediction_folder = config['pred_dir']
            mask_folder = config['mask_dir']
            if not os.path.exists(output_prediction_folder):
                os.makedirs(output_prediction_folder)
            
            frame_path = os.path.join(dataset_dir, f'images/{i:04d}.jpg')

            image = cv.imread(frame_path)
            semlines = calib_net.inference(image)
            
            # Overlay mask on the original image
            overlay_image = overlay_mask_on_image(image.copy(), semlines, mask_colors, config['img_resolution'])

            if config['save_masks']:
                mask = Image.fromarray(semlines.astype(np.uint8)).convert('P')
                mask.putpalette(lines_palette)
                mask_file = os.path.join(mask_folder, f'mask_{i:04d}.png')
                mask.save(mask_file)
            
            skeletons = generate_class_synthesis(semlines, 6, mask_colors, erosion=True)
            extremities = get_line_extremities(skeletons, 40, config['img_resolution'], config['img_resolution'])

            # Visualize keypoints on the overlay image
            result_image = visualize_keypoints_on_image(overlay_image, extremities, color=config['keypoint_color'])

            # Save the result image with keypoints visualized
            result_image_file = os.path.join(output_prediction_folder, f"overlay_{i:04d}.jpg")
            cv.imwrite(result_image_file, result_image)

            # Save the extremities prediction as JSON
            prediction_file = os.path.join(output_prediction_folder, f"keypoints_{i:04d}.json")
            with open(prediction_file, "w") as f:
                json.dump(extremities, f, indent=4)