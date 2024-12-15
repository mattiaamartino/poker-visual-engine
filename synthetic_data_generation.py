import os

import numpy as np
import random
import cv2
from tqdm import tqdm

def deal_hand_and_flop(base_path:str, num_cards_probs:list=[0.35, 0.40, 0.15, 0.10])->tuple:
    """
    Select the hand and flop card images based on the given rules.
    - base_path: Base path for card images.
    - num_cards_probs: Probabilities for selecting the number of cards [2, 5, 6, 7].
    Returns:
        hand_cards (list): List of paths for the hand cards.
        flop_cards (list): List of paths for the flop cards.
        hand_classes (list): List of classes for the hand cards.
        flop_classes (list): List of classes for the flop cards.
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

    deck = random.choice(["first", "second"])
    suffix = "_" if deck == "second" else ""

    random.shuffle(classes)
    num_cards = random.choices(num_cards_options, weights=num_cards_probs, k=1)[0]
    selected_classes = classes[:num_cards]

    # Step 3: Generate file paths for the selected cards
    hand_paths = [os.path.join(base_path, f"{suffix}{card}.png") for card in selected_classes[:2]]
    flop_paths = [os.path.join(base_path, f"{suffix}{card}.png") for card in selected_classes[2:]]

    return hand_paths, flop_paths, selected_classes[:2], selected_classes[2:]


def rotate_image(image:np.array, angle:float)->np.array:
    """
    Rotate the image around its center.
    Parameters:
    - image: Input image as a NumPy array (H x W x 3).
    - angle: Angle in degrees.
    
    Returns:
    - rotated_image: The rotated image.
    """

    h, w = image.shape[:2]
    new_w = int(w * abs(np.cos(np.deg2rad(angle))) + h * abs(np.sin(np.deg2rad(angle))))
    new_h = int(h * abs(np.cos(np.deg2rad(angle))) + w * abs(np.sin(np.deg2rad(angle))))
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    rotated_image = cv2.warpAffine(image, M, (new_w, new_h), borderMode=cv2.BORDER_CONSTANT,borderValue=(0, 0, 0, 0))
    return rotated_image


def generate_canvas(cards:list, overlap_factor:float, max_w:int)->np.array:
    canvas_h = max([card.shape[0] for card in cards])
    canvas_w = sum([card.shape[1] for card in cards]) - int(max_w * overlap_factor) 
    canvas = np.ones((canvas_h, canvas_w, 4), dtype=np.uint8) * 255
    canvas[..., 3] = 0
    return canvas


def dynamic_canvas_adjustment(canvas:np.array, cards:list, overlap_factor:float, positions:list, max_w:int)->tuple:
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


def expand_canvas(canvas:np.array, h:int, w:int, x:int, y:int):
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

def overlay_random_transparent_object(base_image:np.ndarray, object_folder:str) -> np.ndarray:
    """
    Overlays a randomly selected transparent image from `object_folder` onto `base_image`.
    Scales the object (hand) so that its height is 110% of the base image's height,
    then places it randomly in the lower 60% of the base image, allowing partial cropping.
    
    Parameters:
        base_image (np.ndarray): The base RGBA image (H x W x 4).
        object_folder (str): Path to the folder containing transparent RGBA images.
    
    Returns:
        np.ndarray: The modified RGBA image after overlay.
    """
    
    # Get a list of all files in the object folder
    object_files = [f for f in os.listdir(object_folder) if not f.startswith('.') and os.path.isfile(os.path.join(object_folder, f))]
    if not object_files:
        raise ValueError("No object files found in the given folder.")
    
    # Randomly select an object image file
    obj_file = np.random.choice(object_files)
    obj_path = os.path.join(object_folder, obj_file)
    
    # Load the object image with alpha channel
    obj_img = cv2.cvtColor(cv2.imread(obj_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGBA) 

    # Randomly mirror the object image with 50% probability
    if np.random.random() < 0.5:
        obj_img = cv2.flip(obj_img, 1)  # Flip horizontally

    # Get dimensions of base and object
    base_h, base_w = base_image.shape[:2]
    obj_h, obj_w = obj_img.shape[:2]
    
    # Scale the object so that its height is 110% of the base image height
    desired_obj_h = int(1.1 * base_h)
    scale_factor = desired_obj_h / obj_h
    desired_obj_w = int(obj_w * scale_factor)
    obj_img = cv2.resize(obj_img, (desired_obj_w, desired_obj_h), interpolation=cv2.INTER_AREA)
    obj_h, obj_w = obj_img.shape[:2]

    # We want to place the object in the lower 60% of the card
    # That means y should be from 0.4 * base_h to base_h
    y_min = int(0.1 * base_h)
    y_max = base_h  # potentially placing it so that part goes below the card
    
    top_left_y = np.random.randint(y_min, y_max)
    
    # For x, we can allow some part to fall out of the image as well.
    # Let's choose x from a range that might allow partial clipping on either side.
    # For example, from -obj_w//2 to base_w (allow half the width to go off-screen on the left)
    x_min = -obj_w // 2
    x_max = base_w
    top_left_x = np.random.randint(x_min, x_max)

    # Compute the overlapping area between the object and the base image
    # Visible region in the base image
    visible_x_start = max(0, top_left_x)
    visible_y_start = max(0, top_left_y)
    visible_x_end = min(base_w, top_left_x + obj_w)
    visible_y_end = min(base_h, top_left_y + obj_h)

    # If there's no overlap, just return the base image as is
    if visible_x_end <= visible_x_start or visible_y_end <= visible_y_start:
        return base_image

    # Corresponding region in the object
    obj_x_start = visible_x_start - top_left_x
    obj_y_start = visible_y_start - top_left_y
    obj_x_end = obj_x_start + (visible_x_end - visible_x_start)
    obj_y_end = obj_y_start + (visible_y_end - visible_y_start)

    # Extract the relevant portion of the object and the alpha channel
    obj_crop = obj_img[obj_y_start:obj_y_end, obj_x_start:obj_x_end, :]
    alpha_obj = obj_crop[:, :, 3] / 255.0

    # Extract the corresponding base area
    base_crop = base_image[visible_y_start:visible_y_end, visible_x_start:visible_x_end, :]
    alpha_base = base_crop[:, :, 3] / 255.0

    # Compute the combined alpha
    combined_alpha = alpha_obj + alpha_base * (1 - alpha_obj)

    # For each color channel: R, G, B
    for c in range(3):
        base_crop[:, :, c] = (
            obj_crop[:, :, c] * alpha_obj +
            base_crop[:, :, c] * alpha_base * (1 - alpha_obj)
        ) / np.maximum(combined_alpha, 1e-6)

    # Update the alpha channel
    base_crop[:, :, 3] = (combined_alpha * 255).astype(np.uint8)

    # Put the blended region back into the base image
    base_image[visible_y_start:visible_y_end, visible_x_start:visible_x_end, :] = base_crop

    return base_image


# Define the function to merge the cards
def transform_merge(card_paths:list, object_folder:str)->np.array:
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
    rotated_cards = [rotate_image(card, angle) for card, angle in zip(cards, angles)]

    # Define the canvas to fit the cards
    max_w = max([rotated_card.shape[1] for rotated_card in rotated_cards])
    canvas = generate_canvas(rotated_cards, overlap_factor, max_w)

    positions = [(0, 0)]
    canvas, positions = dynamic_canvas_adjustment(canvas, rotated_cards, overlap_factor, positions, max_w)

    # Randomly reverse the order of the cards
    if np.random.rand() > 0.5: 
        rotated_cards = reversed(rotated_cards)
        positions = reversed(positions)

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

    if len(cards) == 2:
        if np.random.random() > 0.5:
            # Cut the cards with a random factor
            cut_factor = np.clip(np.random.normal(4, 3), 3, None)
            canvas_h = canvas.shape[0]
            canvas = canvas[:-int(canvas_h/cut_factor)]
        if np.random.random() > 0.5:
            canvas = overlay_random_transparent_object(canvas, object_folder)

    return canvas

def rotate_image_3d_centered(image:np.array, pitch:float, yaw:float, f:int=2500)->np.array:
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
        [1,            0      ,         0         ],
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

    return warped_image

# Final function that merges the cards with the background
def merge_cards_background(cards_folder:str, background_path:str, objects_folder:str)->np.array:
    """
    Merges card images with a background image to create a synthetic poker scene.
    Args:
        cards_folder (str): Path to the folder containing card images.
        background_path (str): Path to the background image file.
        objects_folder (str): Path to the folder containing additional objects for transformation.
    Returns:
        np.array: The resulting image with cards merged onto the background.
    The function performs the following steps:
    1. Reads and converts the background image.
    2. Deals a hand and flop of cards.
    3. Transforms and merges the hand of cards onto the background.
    4. Optionally transforms and merges the flop of cards onto the background if present.
    5. Uses alpha blending to combine the card images with the background.
    """

    background = cv2.cvtColor(cv2.imread(background_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
    hand, flop, _, _ = deal_hand_and_flop(cards_folder)

    hand = transform_merge(hand, objects_folder)
    pitch_hand = np.random.normal(50, 10)
    yaw_hand = np.random.randint(-45, 45)

    hand = rotate_image_3d_centered(hand, pitch=pitch_hand, yaw=yaw_hand, f=2500)

    h_hand_new = hand.shape[0] //5
    w_hand_new = hand.shape[1] //5
    resized_hand = cv2.resize(hand,(w_hand_new, h_hand_new), interpolation=cv2.INTER_AREA)
    x_new = int((yaw_hand+45)/90 * (background.shape[1]-w_hand_new)) 
    y_new = np.random.randint(background.shape[0]//2, background.shape[0]-h_hand_new)

    alpha = resized_hand[:, :, 3] / 255.0
    for c in range(3):
        background[y_new:y_new+h_hand_new, x_new:x_new+w_hand_new, c] = (
            resized_hand[:, :, c] * alpha +
            background[y_new:y_new+h_hand_new, x_new:x_new+w_hand_new, c] * (1.0 - alpha)
        )

    if len(flop) > 0:
        flop = transform_merge(flop, objects_folder)
        yaw_flop = np.random.normal(0, 3)
        pitch_flop = np.random.randint(-75, -50)

        flop = rotate_image_3d_centered(flop, pitch=pitch_flop, yaw=yaw_flop, f=2500)

        h_new_flop = flop.shape[0] // 8
        w_new_flop = flop.shape[1] // 8
        if background.shape[1] - w_new_flop <= 0:
            w_new_flop = flop.shape[1] // 10
        resized_flop = cv2.resize(flop, (w_new_flop, h_new_flop), interpolation=cv2.INTER_AREA)

        x_new_flop = np.random.randint(0, background.shape[1]-w_new_flop)

        # Calculate y_new based on pitch_flop
        min_y_flop = 0
        max_y_flop = background.shape[0] // 2 - h_new_flop
        pitch_normalized_flop = (-pitch_flop - 50) / 30  # Normalize pitch to [0, 1] for range [-80, -50]
        y_new_flop = int(min_y_flop + pitch_normalized_flop * (max_y_flop - min_y_flop))

        # Alpha blending for flop
        alpha_flop = resized_flop[:, :, 3] / 255.0
        for c in range(3):
            background[y_new_flop:y_new_flop+h_new_flop, x_new_flop:x_new_flop+w_new_flop, c] = (
                resized_flop[:, :, c] * alpha_flop +
                background[y_new_flop:y_new_flop+h_new_flop, x_new_flop:x_new_flop+w_new_flop, c] * (1.0 - alpha_flop)
            )

    return background


# Define the function to generate the synthetic data
def generate_synthetic_data(num_samples:int, cards_folder:str, background_folder:str, objects_folder:str, output_folder:str)->None:

    image_dir = os.path.join(output_folder, "images")
    os.makedirs(image_dir, exist_ok=True)

    background_files = [f for f in os.listdir(background_folder) if not f.startswith('.') and os.path.isfile(os.path.join(background_folder, f))]

    for i in tqdm(range(num_samples)):

        background_file = np.random.choice(background_files)
        background_file = os.path.join(background_folder, background_file)
        final_image = merge_cards_background(cards_folder, background_file, objects_folder)

        image_path = os.path.join(image_dir, f"image_{i:05d}.png")
        cv2.imwrite(image_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))

generate_synthetic_data(10, '../shared_data/benchmark', '../background', '../shared_data/objects', '../synthetic_data_trial')
