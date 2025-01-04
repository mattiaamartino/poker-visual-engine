# Poker Visual Engine
Authors: Mattia Martino, Alessandro Pranzo

## Description
In this work we propose a deep learning framework to detect poker hands from a first-person per-
spective, focusing on classifying hole cards and community cards. Due to the absence of suitable
datasets, we generated a synthetic dataset using geometrical transformations, scanned card decks,
and the MS COCO dataset for realistic backgrounds. This dataset simulates various gameplay sce-
narios and includes automatically generated bounding boxes for card identification.
We fine-tuned the YOLOv11 model, a state-of-the-art object detection framework, on this dataset to
perform robust card detection and classification. The model demonstrates strong performance within
the synthetic dataset, successfully identifying both individual cards and their roles in gameplay.
However, its generalization to real-world images is limited, suggesting the need for mixed datasets
combining synthetic and manually annotated real-life images.
Our findings highlight the potential of synthetic data in training deep learning models for poker card
recognition while emphasizing areas for future improvement, including enhanced dataset diversity
and advanced warping techniques to better simulate real-world conditions.
