import os

import numpy as np
import random
import cv2
from tqdm import tqdm

def deal_hand_and_flop(base_path, num_cards_probs=[0.35, 0.40, 0.15, 0.10]):
    """
    Select the hand and flop card images based on the given rules.
    - base_path: Base path for card images.
    - num_cards_probs: Probabilities for selecting the number of cards [2, 5, 6, 7].
    Returns:
        hand_cards (list): List of paths for the hand cards.
        flop_cards (list): List of paths for the flop cards.
    """
    # Define card classes and number of cards to draw
    classes = [ '2C', '2D', '2H', '2S',
                '3C', '3D', '3H', '3S',
                '4C', '4D', '4H', '4S',
                '5C', '5D', '5H', '5S', 
                '6C', '6D', '6H', '6S', 
                '7C', '7D', '7H', '7S', 
                '8C', '8D', '8H', '8S', 
                '9C', '9D', '9H', '9S',
                'TC', 'TD', 'TH', 'TS', 
                'AC', 'AD', 'AH', 'AS', 
                'JC', 'JD', 'JH', 'JS',
                'KC', 'KD', 'KH', 'KS', 
                'QC', 'QD', 'QH', 'QS']
    num_cards_options = [2, 5, 6, 7]

    # Step 1: Randomly choose a deck
    deck = random.choice(["first", "second"])
    suffix = "_" if deck == "second" else ""

    #Constructing the bounding box for the yolo format (x_center, y_center, width, height)
    if deck == "first":
        b_boxes_yolo = [[66, 145, 100, 230], [625, 911, 100, 230], [345, 528, 691, 1056]]
        #b_boxes_yolo = [66, 145, 100, 230]
    else:
        b_boxes_yolo = [[45, 90, 90, 180], [455, 636, 90, 180], [250, 363, 500, 726]]
        #b_boxes_yolo = [45, 90, 90, 180]
    

    # Step 2: Shuffle and draw a random number of cards
    random.shuffle(classes)
    num_cards = random.choices(num_cards_options, weights=num_cards_probs, k=1)[0]
    selected_classes = classes[:num_cards]

    hand_labels = [[card, b_boxes_yolo] for card in selected_classes[:2]]
    flop_labels = [[card, b_boxes_yolo] for card in selected_classes[2:]]

    # Step 3: Generate file paths for the selected cards
    hand_paths = [os.path.join(base_path, f"{suffix}{card}.png") for card in selected_classes[:2]]
    flop_paths = [os.path.join(base_path, f"{suffix}{card}.png") for card in selected_classes[2:]]

    return hand_paths, flop_paths, hand_labels, flop_labels

# YOLO support functions
def yolo_notation2cv2_notation(yolo_notation):
    x, y, w, h = yolo_notation
    #Upper left
    x1 = int(x - w / 2)
    y1 = int(y - h / 2)
    #Upper right
    x2 = x1 + w
    y2 = y1
    #Lower right
    x3 = x2
    y3 = y1 + h
    #Lower left
    x4 = x1
    y4 = y3

    return np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

def cv2_notation2yolo_notation(cv2_notation):
    x1, y1 = cv2_notation[0]
    x3, y3 = cv2_notation[2]
    
    #Width and height
    w = x3 - x1
    h = y3 - y1

    #Center
    x = x1 + w / 2
    y = y1 + h / 2
    
    return x, y, w, h

def cv2_modified2yolo_notation(cv2_modified):
    # Extract the x and y coordinates from the points
    x_coords = cv2_modified[:, 0]
    y_coords = cv2_modified[:, 1]

    # Calculate the center
    x_center = sum(x_coords) / 4
    y_center = sum(y_coords) / 4

    # Calculate the width and height
    width = max(x_coords) - min(x_coords)
    height = max(y_coords) - min(y_coords)

    return x_center, y_center, width, height



def rotate_image(image, b_boxes_yolo, angle):
    """
    Rotate the image around its center.
    
    Parameters:
    - image: Input image as a NumPy array (H x W x 3).
    - b_box_yolo: Bounding box in YOLO format (x_center, y_center, width, height).
    - angle: Angle in degrees.
    
    Returns:
    - rotated_image: The rotated image.
    - rotated_bb_yolo: The rotated bounding box in YOLO format.
    """

    h, w = image.shape[:2]
    new_w = int(w * abs(np.cos(np.deg2rad(angle))) + h * abs(np.sin(np.deg2rad(angle))))
    new_h = int(h * abs(np.cos(np.deg2rad(angle))) + w * abs(np.sin(np.deg2rad(angle))))
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    # Rotate the bounding box
    def rotate_bounding_box(b_box_yolo):
        bb_coords = yolo_notation2cv2_notation(b_box_yolo)
        bb_coords = np.hstack([bb_coords, np.ones((bb_coords.shape[0], 1), dtype=bb_coords.dtype)])
        rotated_points = bb_coords @ M.T
        return rotated_points
    
    edge_points = [rotate_bounding_box(b_box_yolo) for b_box_yolo in b_boxes_yolo]
    b_boxes_yolo = [cv2_modified2yolo_notation(edge) for edge in edge_points]    

    rotated_image = cv2.warpAffine(image, M, (new_w, new_h))
    return rotated_image, b_boxes_yolo, edge_points


def generate_canvas(cards, overlap_factor, max_w):
    canvas_h = max([card.shape[0] for card in cards])
    canvas_w = sum([card.shape[1] for card in cards]) - int(max_w * overlap_factor) 
    canvas = np.ones((canvas_h, canvas_w, 4), dtype=np.uint8) * 255
    canvas[..., 3] = 0
    return canvas


def dynamic_canvas_adjustment(canvas, cards, overlap_factor, positions, max_w):
    for i in range(1, len(cards)):
        last_x, last_y = positions[-1]
        overlap_x = cards[i - 1].shape[1] - int(overlap_factor * max_w)
        positions.append((last_x + overlap_x, last_y))

    canvas_h, canvas_w = canvas.shape[:2]
    min_x = min([pos[0] for pos in positions])
    if min_x < 0:
        # Expand the canvas width by abs(min_x) on the left
        new_canvas_w = canvas_w + abs(min_x)
        new_canvas = np.ones((canvas_h, new_canvas_w, 4), dtype=np.uint8) * 255

        # Shift all positions to the right by abs(min_x)
        positions = [(pos[0] + abs(min_x), pos[1]) for pos in positions]

        # Copy old canvas into the new one
        new_canvas[:, abs(min_x):, :] = canvas
        canvas = new_canvas
    
    return canvas, positions


def expand_canvas(canvas, h, w, x, y):
    if x + w > canvas.shape[1]:
        # Expand the canvas width
        new_canvas_w = x + w
        new_canvas = np.ones((canvas.shape[0], new_canvas_w, 4), dtype=np.uint8) * 255
        new_canvas[:, :canvas.shape[1], :] = canvas
        canvas = new_canvas
    if y + h > canvas.shape[0]:
        # Expand the canvas height
        new_canvas_h = y + h
        new_canvas = np.ones((new_canvas_h, canvas.shape[1], 4), dtype=np.uint8) * 255
        new_canvas[:canvas.shape[0], :, :] = canvas
        canvas = new_canvas
    return canvas

def is_visible(large_rect, small_rect):
    """
    Check if a smaller rectangle is fully outside the larger rectangle.
    
    Parameters:
    large_rect (numpy.ndarray): 4x2 array representing the larger rectangle's vertices
    small_rect (numpy.ndarray): 4x2 array representing the smaller rectangle's vertices
    
    Returns:
    bool: True if the small rectangle is fully outside the large rectangle, 
          False if any part of the small rectangle is inside the large rectangle
    """
    def is_point_inside_rect(point, rect):
        # Ensure the point and rectangle are numpy arrays
        point = np.asarray(point)
        
        # Number of intersections with rectangle edges
        intersections = 0
        
        for i in range(4):
            j = (i + 1) % 4
            # Check if the point is between the y-coordinates of the edge
            if ((rect[i][1] > point[1]) != (rect[j][1] > point[1])):
                # Calculate x-coordinate of the edge intersection with a horizontal line
                x_intersect = rect[i][0] + (point[1] - rect[i][1]) * \
                    (rect[j][0] - rect[i][0]) / (rect[j][1] - rect[i][1])
                
                # If the point's x is to the left of the intersection, count an intersection
                if point[0] < x_intersect:
                    intersections += 1
        
        # If number of intersections is odd, point is inside the rectangle
        return intersections % 2 == 1

    # Check if any point of the small rectangle is inside the large rectangle
    for point in small_rect:
        if is_point_inside_rect(point, large_rect):
            return False
    
    # Check if any point of the large rectangle is inside the small rectangle
    for point in large_rect:
        if is_point_inside_rect(point, small_rect):
            return False
    
    return True


# Define the function to merge the cards
def transform_merge(card_paths, card_labels):
    # Extract the cards
    cards = [cv2.cvtColor(cv2.imread(card), cv2.COLOR_BGR2RGBA) for card in card_paths]
    # Genearate the angles for rotation and percentage of overlap
    if len(cards) == 2:
        angles = [np.random.normal(0, 20) for _ in range(len(cards))]
        overlap_factor = np.clip(np.random.normal(0.4, 0.2), -0.1, .7)
    else:
        angles = [np.random.normal(0, 5) for _ in range(len(cards))]
        overlap_factor = np.clip(np.random.normal(0, 0.2), -0.2, 0.2)

    # Apply shading and Gaussian blur
    shading_coeff = np.clip(np.random.normal(1, 0.3), 0.3, 2)
    for i in range(len(cards)):
        cards[i][..., :3] = np.clip(cards[i][..., :3] * shading_coeff, 0, 255).astype(np.uint8)
        cards[i] = cv2.GaussianBlur(cards[i], (5, 5), 0)

    # Rotate the cards
    rotated_cards = []
    modified_labels = []
    edges = []
    for card, label, angle in zip(cards, card_labels, angles):
        rotated_card, rotated_bboxes, edge_points = rotate_image(card, label[1], angle)
        rotated_cards.append(rotated_card)
        modified_labels.append([label[0], rotated_bboxes])
        edges.append(edge_points)
    

    # Define the canvas to fit the cards
    max_w = max([rotated_card.shape[1] for rotated_card in rotated_cards])
    canvas = generate_canvas(rotated_cards, overlap_factor, max_w)

    positions = [(0, 0)]
    canvas, positions = dynamic_canvas_adjustment(canvas, rotated_cards, overlap_factor, positions, max_w)

    # Adjust the bounding boxes and edge points
    for j in range(1, len(positions)):
        for i in range(len(modified_labels[j][1])):
            modified_labels[j][1][i] = list(modified_labels[j][1][i])
            modified_labels[j][1][i][0] += positions[j][0]
            modified_labels[j][1][i] = tuple(modified_labels[j][1][i])
            edges[j][i][:, 0] += positions[j][0]

    rev = False
    # Randomly reverse the order of the cards
    if np.random.rand() > 0.5: 
        rev = True
        rotated_cards = list(reversed(rotated_cards))
        modified_labels = list(reversed(modified_labels))
        edges = list(reversed(edges))
        positions = list(reversed(positions))

    for i in range(1, len(edges)):
        # Conditions:
        invisible_to_first = not is_visible(edges[i][-1], edges[i - 1][0])
        invisible_to_second = not is_visible(edges[i][-1], edges[i - 1][1])

        if invisible_to_first and invisible_to_second:
            # Both are invisible:
            # Adjust overlap and re-run layout (as before)
            #print("Warning: Both bounding boxes are invisible for card:", modified_labels[i-1][0])
            #print("Cards were reversed:", rev)
            overlap_factor = -0.2

            if not rev:
                for j in range(1, len(positions)):
                    for k in range(len(modified_labels[j][1])):
                        x, y, w, h = modified_labels[j][1][k]
                        modified_labels[j][1][k] = (x - positions[j][0], y, w, h)
                        edges[j][i][:, 0] -= positions[j][0]
            else:
                for j in range(0, len(positions)-1):
                    for k in range(len(modified_labels[j][1])):
                        x, y, w, h = modified_labels[j][1][k]
                        modified_labels[j][1][k] = (x - positions[j][0], y, w, h)
                        edges[j][i][:, 0] -= positions[j][0]
                    
            # Re-generate the canvas and positions
            canvas = generate_canvas(rotated_cards, overlap_factor, max_w)
            positions = [(0, 0)]
            canvas, positions = dynamic_canvas_adjustment(canvas, rotated_cards, overlap_factor, positions, max_w)

            # Re-adjust bounding boxes and edges after re-positioning
            for j in range(1, len(positions)):
                for k in range(len(modified_labels[j][1])):
                    x, y, w, h = modified_labels[j][1][k]
                    modified_labels[j][1][k] = (x + positions[j][0], y, w, h)
                    edges[j][i][:, 0] += positions[j][0]
            
        else:
            # If only one is invisible:
            # If invisible_to_first is True and invisible_to_second is False, remove index 0
            # If invisible_to_second is True and invisible_to_first is False, remove index 1
            if invisible_to_first and not invisible_to_second:
                # Remove the bounding box at index 0 from the previous card
                #print("Removing upper left bounding box for card:", modified_labels[i-1][0])
                if len(modified_labels[i-1][1]) > 0:
                    del modified_labels[i-1][1][0]
                    del edges[i-1][0]

            elif invisible_to_second and not invisible_to_first:
                # Remove the bounding box at index 1 from the previous card (if it exists)
                #print("Removing lower right bounding box for card:", modified_labels[i-1][0])
                if len(modified_labels[i-1][1]) > 1:
                    del modified_labels[i-1][1][1]
                    del edges[i-1][1]
            
        

    
    # Merge the cards onto the canvas
    for card, pos in zip(rotated_cards, positions):
        card_h, card_w = card.shape[:2]
        x, y = pos

        # Check if the card exceeds the canvas boundaries and adjust
        canvas = expand_canvas(canvas, card_h, card_w, x, y)

        alpha = card[:, :, 3] / 255.0
        for c in range(3):
            canvas[y:y+card_h, x:x+card_w, c] = (
                card[:, :, c] * alpha +
                canvas[y:y+card_h, x:x+card_w, c] * (1.0 - alpha)
            )
        # Update alpha channel
        canvas[y:y+card_h, x:x+card_w, 3] = (
            alpha * 255 + canvas[y:y+card_h, x:x+card_w, 3] * (1.0 - alpha)
        )

    canvas_h, canvas_w = canvas.shape[:2]

    if len(cards) == 2:
        if random.random() > 0.2:
            max_y = 0
            for card_edges in edges:
                if len(card_edges) <= 2:
                    # edge_points is an array with shape (4, 2) for rectangle corners: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                    rect_max_y = np.max(card_edges[0][:, 1])
                    if rect_max_y > max_y:
                        max_y = rect_max_y
            

            cut_factor = np.clip(np.random.normal(4, 3), 3, None)
            canvas_h = canvas.shape[0]

            proposed_cut_height = int(canvas_h / cut_factor)

            # Ensure we do not cut into the minimal critical rectangle area.
            max_allowed_cut = canvas_h - int(max_y)
            if proposed_cut_height > max_allowed_cut:
                proposed_cut_height = max_allowed_cut

            canvas = canvas[:-proposed_cut_height]
        
        # Overlay a random transparent object
        # if random.random() > 0.5:
        #     canvas = overlay_random_transparent_object(canvas, object_folder)

        modified_labels.append(['hand', [[canvas_w//2, canvas_h//2, canvas_w, canvas_h]]])
    else:
        modified_labels.append(['flop', [[canvas_w//2, canvas_h//2, canvas_w, canvas_h]]])

    
    return canvas, modified_labels


def rotate_image_3d_centered(image, image_labels, pitch, yaw, f=2000):
    """
    Rotate the image in 3D around its center and keep the center of the image stable.
    The output canvas is adjusted so the entire rotated image is visible.
    
    Parameters:
    - image: Input image as a NumPy array (H x W x 3).
    - pitch: Pitch angle in degrees (rotation around x-axis).
    - yaw:   Yaw angle in degrees (rotation around y-axis).
    - f:     Focal length for the projection (in pixels).

    Returns:
    - warped_image: The rotated image, with the center in roughly the same position 
                    and no cutoff.
    """
    h, w = image.shape[:2]

    # Image center
    cx, cy = w / 2.0, h / 2.0

    # Convert angles to radians
    pitch_rad = np.deg2rad(pitch)
    yaw_rad = np.deg2rad(yaw)

    # Rotation matrices
    R_pitch = np.array([
        [1,              0,               0           ],
        [0,  np.cos(pitch_rad), -np.sin(pitch_rad)],
        [0,  np.sin(pitch_rad),  np.cos(pitch_rad)]
    ], dtype=np.float32)

    R_yaw = np.array([
        [ np.cos(yaw_rad), 0, np.sin(yaw_rad)],
        [ 0,               1,              0],
        [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]
    ], dtype=np.float32)

    # Combined rotation
    R = R_yaw @ R_pitch

    # Original corners
    original_corners_2d = np.array([[0,0],
                                    [w,0],
                                    [w,h],
                                    [0,h]], dtype=np.float32)

    corners_3d = np.array([
        [0, 0, 0],
        [w, 0, 0],
        [w, h, 0],
        [0, h, 0]
    ], dtype=np.float32)

    # Center the corners around the image center
    corners_centered = corners_3d - [cx, cy, 0]

    # Rotate corners
    rotated_corners = corners_centered @ R.T

    # Project corners
    Z = rotated_corners[:, 2] + f
    X = rotated_corners[:, 0]
    Y = rotated_corners[:, 1]

    Z[Z == 0] = 0.001
    projected_corners = np.zeros((4, 2), dtype=np.float32)
    projected_corners[:, 0] = (f * X / Z) + cx
    projected_corners[:, 1] = (f * Y / Z) + cy

    # Project the original center to see where it lands
    center_3d = np.array([[cx, cy, 0]], dtype=np.float32)
    center_centered = center_3d - [cx, cy, 0]
    center_rotated = center_centered @ R.T
    Zc = center_rotated[0, 2] + f
    Xc = center_rotated[0, 0]
    Yc = center_rotated[0, 1]
    proj_cx = (f * Xc / Zc) + cx
    proj_cy = (f * Yc / Zc) + cy

    # Compute homography
    H, _ = cv2.findHomography(original_corners_2d, projected_corners)

    # We want the projected center to remain at (cx, cy)
    # So we translate by (cx - proj_cx, cy - proj_cy)
    T_center = np.array([
        [1, 0, cx - proj_cx],
        [0, 1, cy - proj_cy],
        [0, 0, 1]
    ], dtype=np.float32)

    H_adjusted = T_center @ H

    # Apply H_adjusted to corners to find the final bounding box
    ones = np.ones((4,1), dtype=np.float32)
    corners_h = np.hstack([original_corners_2d, ones])
    transformed_corners = (H_adjusted @ corners_h.T).T
    transformed_corners = transformed_corners[:, :2] / transformed_corners[:, 2:3]

    # Compute bounding box
    all_x = transformed_corners[:, 0]
    all_y = transformed_corners[:, 1]
    min_x, max_x = np.min(all_x), np.max(all_x)
    min_y, max_y = np.min(all_y), np.max(all_y)

    new_w = int(np.ceil(max_x - min_x))
    new_h = int(np.ceil(max_y - min_y))

    # We also need to shift so that the bounding box starts at (0,0)
    # We'll add another translation to the homography
    T_box = np.array([
        [1, 0, -min_x],
        [0, 1, -min_y],
        [0, 0, 1]
    ], dtype=np.float32)

    H_final = T_box @ H_adjusted

    # Warp with final homography
    warped_image = cv2.warpPerspective(image, H_final, (new_w, new_h))

    # Warp the bounding box
    def warp_bounding_box(b_box_yolo):
        points = yolo_notation2cv2_notation(b_box_yolo)
        points_h = np.hstack([points, np.ones((points.shape[0], 1), dtype=points.dtype)])
        transformed_points_h = (H_final @ points_h.T).T

        transformed_points = transformed_points_h[:, :2] / transformed_points_h[:, 2:3]
        return cv2_modified2yolo_notation(transformed_points)

    for i, c in enumerate(image_labels):
        image_labels[i] = [c[0], [warp_bounding_box(b_box_yolo) for b_box_yolo in c[1]]]

    return warped_image, image_labels


def scale_bounding_boxes(labels, scale_x, scale_y):
    for item in labels:
        item[1] = [(x * scale_x, y * scale_y, w * scale_x, h * scale_y) for x, y, w, h in item[1]]
    
    return labels

def translate_coordinates(labels, shift_x, shift_y):
    for item in labels:
        item[1] = [ (x + shift_x, y + shift_y, w, h) for x, y, w, h in item[1]]
    
    return labels



# Final function that merges the cards with the background
def merge_cards_background(cards_folder, background_path):

    background = cv2.cvtColor(cv2.imread(background_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
    hand, flop, hand_labels, flop_labels = deal_hand_and_flop(cards_folder)

    hand, trans_hand_labels = transform_merge(hand, hand_labels)
    pitch_hand = np.random.normal(50, 10)
    yaw_hand = np.random.randint(-45, 45)

    hand, final_hand_labels = rotate_image_3d_centered(hand, trans_hand_labels, pitch=pitch_hand, yaw=yaw_hand, f=2500)

    h_hand_new = int(np.round(hand.shape[0] /5))
    w_hand_new = int(np.round(hand.shape[1] /5))
    resized_hand = cv2.resize(hand,(w_hand_new, h_hand_new), interpolation=cv2.INTER_AREA)
    x_new = int((yaw_hand+45)/90 * (background.shape[1]-w_hand_new)) 
    y_new = np.random.randint(background.shape[0]//2, background.shape[0]-h_hand_new)

    resized_hand_labels = scale_bounding_boxes(final_hand_labels, 1/5, 1/5)
    shifted_hand_labels = translate_coordinates(resized_hand_labels, x_new, y_new)
    

    alpha = resized_hand[:, :, 3] / 255.0
    for c in range(3):
        background[y_new:y_new+h_hand_new, x_new:x_new+w_hand_new, c] = (
            resized_hand[:, :, c] * alpha +
            background[y_new:y_new+h_hand_new, x_new:x_new+w_hand_new, c] * (1.0 - alpha)
        )

    if len(flop) > 0:
        flop, trans_flop_labels = transform_merge(flop, flop_labels)
        yaw_flop = np.random.normal(0, 3)
        pitch_flop = np.random.randint(-75, -50)

        flop, final_flop_labels = rotate_image_3d_centered(flop, trans_flop_labels, pitch=pitch_flop, yaw=yaw_flop, f=2500)

        h_new_flop = int(np.round(flop.shape[0] / 8))
        w_new_flop = int(np.round(flop.shape[1] / 8))
        if background.shape[1] - w_new_flop <= 0:
            w_new_flop = int(np.round(flop.shape[1] // 10))
        resized_flop = cv2.resize(flop, (w_new_flop, h_new_flop), interpolation=cv2.INTER_AREA)

        x_new_flop = np.random.randint(0, background.shape[1]-w_new_flop)

        # Calculate y_new based on pitch_flop
        min_y_flop = 0
        max_y_flop = background.shape[0] // 2 - h_new_flop
        pitch_normalized_flop = (-pitch_flop - 50) / 30  # Normalize pitch to [0, 1] for range [-80, -50]
        y_new_flop = int(min_y_flop + pitch_normalized_flop * (max_y_flop - min_y_flop))

        resized_flop_labels = scale_bounding_boxes(final_flop_labels, 1/8, 1/8)
        shifted_flop_labels = translate_coordinates(resized_flop_labels, x_new_flop, y_new_flop)

        # Alpha blending for flop
        alpha_flop = resized_flop[:, :, 3] / 255.0
        for c in range(3):
            background[y_new_flop:y_new_flop+h_new_flop, x_new_flop:x_new_flop+w_new_flop, c] = (
                resized_flop[:, :, c] * alpha_flop +
                background[y_new_flop:y_new_flop+h_new_flop, x_new_flop:x_new_flop+w_new_flop, c] * (1.0 - alpha_flop)
            )

        if not shifted_flop_labels:
            shifted_flop_labels = []

    return background, shifted_hand_labels, shifted_flop_labels


class_map = {'2C': 0, '2D': 1, '2H': 2, '2S': 3,
            '3C': 4, '3D': 5, '3H': 6, '3S': 7,
            '4C': 8, '4D': 9, '4H': 10, '4S': 11, 
            '5C': 12, '5D': 13, '5H': 14, '5S': 15, 
            '6C': 16, '6D': 17, '6H': 18, '6S': 19, 
            '7C': 20, '7D': 21, '7H': 22, '7S': 23, 
            '8C': 24, '8D': 25, '8H': 26, '8S': 27,
            '9C': 28, '9D': 29, '9H': 30, '9S': 31,
            'AC': 32, 'AD': 33, 'AH': 34, 'AS': 35, 
            'JC': 36, 'JD': 37, 'JH': 38, 'JS': 39, 
            'KC': 40, 'KD': 41, 'KH': 42, 'KS': 43, 
            'QC': 44, 'QD': 45, 'QH': 46, 'QS': 47,
            'TC': 48, 'TD': 49, 'TH': 50, 'TS': 51,
            'hand': 52, 'flop': 53}

# Define the function to generate the synthetic data
def generate_synthetic_data(num_samples:int, cards_folder:str, background_folder:str, output_folder:str, class_map:dict)->None:

    image_dir = os.path.join(output_folder, "images")
    labels_lr_dir = os.path.join(output_folder, "labels_lr")
    labels_full_dir = os.path.join(output_folder, "labels_full")

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(labels_lr_dir, exist_ok=True)
    os.makedirs(labels_full_dir, exist_ok=True)

    background_files = [f for f in os.listdir(background_folder) if not f.startswith('.') and os.path.isfile(os.path.join(background_folder, f))]

    for i in tqdm(range(num_samples), desc=f"Generating {output_folder}"):
        try:

            background_file = np.random.choice(background_files)
            background_file = os.path.join(background_folder, background_file)
            final_image, final_hand_labels, final_flop_labels = merge_cards_background(cards_folder, background_file)
            image_path = os.path.join(image_dir, f"image_{i:05d}.png")
            cv2.imwrite(image_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))

            label_lr_path = os.path.join(labels_lr_dir, f"image_{i:05d}.txt")
            label_full_path = os.path.join(labels_full_dir, f"image_{i:05d}.txt")
            with open(label_lr_path, "w") as f:
                for c in final_hand_labels:
                    if len(c[1]) > 2:
                        f.write(f"{class_map[c[0]]} {c[1][0][0]/640} {c[1][0][1]/640} {c[1][0][2]/640} {c[1][0][3]/640}\n")
                        f.write(f"{class_map[c[0]]} {c[1][1][0]/640} {c[1][1][1]/640} {c[1][1][2]/640} {c[1][1][3]/640}\n")
                    else:
                        f.write(f"{class_map[c[0]]} {c[1][0][0]/640} {c[1][0][1]/640} {c[1][0][2]/640} {c[1][0][3]/640}\n")

                if len(final_flop_labels) > 0:
                    for c in final_flop_labels:
                        if len(c[1]) > 2:
                            f.write(f"{class_map[c[0]]} {c[1][0][0]/640} {c[1][0][1]/640} {c[1][0][2]/640} {c[1][0][3]/640}\n")
                            f.write(f"{class_map[c[0]]} {c[1][1][0]/640} {c[1][1][1]/640} {c[1][1][2]/640} {c[1][1][3]/640}\n")
                        else:
                            f.write(f"{class_map[c[0]]} {c[1][0][0]/640} {c[1][0][1]/640} {c[1][0][2]/640} {c[1][0][3]/640}\n")

            with open(label_full_path, "w") as f:
                for c in final_hand_labels:
                    f.write(f"{class_map[c[0]]} {c[1][-1][0]/640} {c[1][-1][1]/640} {c[1][-1][2]/640} {c[1][-1][3]/640}\n")
                
                if len(final_flop_labels) > 0:
                    for c in final_flop_labels:
                        f.write(f"{class_map[c[0]]} {c[1][-1][0]/640} {c[1][-1][1]/640} {c[1][-1][2]/640} {c[1][-1][3]/640}\n")


        except:
            continue

generate_synthetic_data(15_000, './data/benchmark', './data/background', './synthetic_dataset_v3/train', class_map)
generate_synthetic_data(2_500, './data/benchmark', './data/background', './synthetic_dataset_v3/test', class_map)
generate_synthetic_data(2_500, './data/benchmark', './data/background', './synthetic_dataset_v3/val', class_map)
