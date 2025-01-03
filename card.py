import os
import random
import numpy as np
import cv2

class Card:
    def __init__(self, path: str, bboxes: np.array=None):
        """
        Loads a card image (RGBA) from path and keeps track of YOLO bounding boxes.
        
        :param path: Path to the card .png file
        :param bboxes: A list of bounding boxes in YOLO format [class, [x, y, w, h]]
        """
        self.image = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGBA)
        self.b_boxes = bboxes if bboxes else []


    def rotate(self, angle: float):
        """
        Rotates the card image by angle degrees.
        
        :param angle: The angle to rotate the image by
        """
        rotated_image, rotated_bboxes, _ = self._rotate_image(self.image, self.b_boxes, angle)
        self.image = rotated_image
        self.b_boxes = rotated_bboxes


    def shade(self, mean: float=1, std: float=0.3, kernel_size: tuple=(5, 5)):
        """
        Apply shading and Gaussian blur to the image.
        This method adjusts the shading of the image by applying a random 
        shading coefficient drawn from a normal distribution with the specified 
        mean and standard deviation. The shading coefficient is clipped to be 
        within the range [0.3, 2]. The method then applies a Gaussian blur to 
        the image using the specified kernel size.
        Parameters:
        mean (float): The mean of the normal distribution for the shading coefficient. Default is 1.
        std (float): The standard deviation of the normal distribution for the shading coefficient. Default is 0.3.
        kernel_size (tuple): The size of the kernel to be used for the Gaussian blur. Default is (5, 5).
        Returns:
        None
        """
        shading_coeff = np.clip(np.random.normal(mean, std), 0.3, 2)
        self.image[..., :3] = np.clip(self.image[..., :3] * shading_coeff, 0, 255).astype(np.uint8)
        self.image = cv2.GaussianBlur(self.image, kernel_size, 0)


    @staticmethod
    def _yolo_to_cv2(yolo_notation: tuple) -> np.ndarray:
        """
        Convert YOLO format bounding box notation to OpenCV format.
        YOLO format represents bounding boxes with the center coordinates (x, y),
        width (w), and height (h). This function converts these to the four corner
        points required by OpenCV.
        Parameters:
        yolo_notation (tuple): A tuple containing four elements (x, y, w, h) where
                               x and y are the center coordinates, and w and h are
                               the width and height of the bounding box.
        Returns:
        np.ndarray: A 2D numpy array of shape (4, 2) containing the coordinates of
                    the four corners of the bounding box in OpenCV format.
        """
        x, y, w, h = yolo_notation
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = x1 + w
        y2 = y1
        x3 = x2
        y3 = y1 + h
        x4 = x1
        y4 = y3

        return np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.float32)
    
    
    @staticmethod
    def _cv2_to_yolo(cv2_coords: np.ndarray) -> tuple:
        """
        Convert OpenCV bounding box coordinates to YOLO format.
        Parameters:
        cv2_coords (numpy.ndarray): A 2D array of shape (N, 2) where N is the number of points,
                                    and each point is represented by (x, y) coordinates.
        Returns:
        tuple: A tuple containing (x_center, y_center, width, height) of the bounding box in YOLO format.
        """
        x_coords = cv2_coords[:, 0]
        y_coords = cv2_coords[:, 1]

        w = np.max(x_coords) - np.min(x_coords)
        h = np.max(y_coords) - np.min(y_coords)
        x_center = np.min(x_coords) + w / 2
        y_center = np.min(y_coords) + h / 2

        return (x_center, y_center, w, h)
    

    @classmethod
    def _rotate_image(cls, image: np.ndarray, bboxes: list, angle: float) -> tuple:
        """
        Rotates an image and its corresponding bounding boxes by a given angle.
        Args:
            cls: The class instance.
            image (np.ndarray): The image to be rotated.
            bboxes (list): List of bounding boxes in YOLO format.
            angle (float): The angle by which to rotate the image and bounding boxes.
        Returns:
            tuple: A tuple containing:
                - rotated_image (np.ndarray): The rotated image.
                - new_bboxes_yolo (list): List of new bounding boxes in YOLO format after rotation.
                - edge_points (list): List of edge points for each bounding box after rotation.
        """
        h, w = image.shape[:2]
        # Compute new bounding image size after rotation
        new_w = int(w * abs(np.cos(np.deg2rad(angle))) + h * abs(np.sin(np.deg2rad(angle))))
        new_h = int(h * abs(np.cos(np.deg2rad(angle))) + w * abs(np.sin(np.deg2rad(angle))))
        center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        # Adjust translation to keep the result centered
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2

        # Rotate bounding boxes
        def rotate_bbox(yolo_box):
            coords = cls._yolo_to_cv2(yolo_box)
            coords_ones = np.hstack([coords, np.ones((coords.shape[0], 1), dtype=coords.dtype)])
            rotated_coords = coords_ones @ M.T
            return rotated_coords

        edge_points = []
        new_bboxes_yolo = []
        for box in bboxes:
            edge_pts = rotate_bbox(box)
            edge_points.append(edge_pts)
            new_bboxes_yolo.append(cls._cv2_to_yolo(edge_pts))

        # Rotate the image
        rotated_image = cv2.warpAffine(image, M, (new_w, new_h))

        return rotated_image, new_bboxes_yolo, edge_points
    

















































import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_card(card):
    """
    Plot the card image and its YOLO bounding boxes.

    Parameters:
        card (Card): An instance of the Card class.
    """
    # Ensure the card has an image
    if card.image is None:
        print("No image available for the card.")
        return
    
    # Convert the RGBA image to RGB for display
    img = card.image[..., :3]

    # Plot the image
    fig, ax = plt.subplots(1, figsize=(6, 8))
    ax.imshow(img)
    
    # Overlay bounding boxes
    for bbox in card.b_boxes:
        # Convert YOLO format to OpenCV format
        cv2_box = card._yolo_to_cv2(bbox)
        rect = patches.Polygon(cv2_box, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(
            cv2_box[0][0], cv2_box[0][1] - 10,
            f"Class: {bbox[0]}",
            color="red", fontsize=8, backgroundcolor="white"
        )

    ax.axis('off')
    plt.show()

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

hand, flop, hand_labels, flop_labels = deal_hand_and_flop("../shared_data/benchmark")
card = Card(hand[0], hand_labels[0][1])
card.shade()
card.rotate(45)
plot_card(card)