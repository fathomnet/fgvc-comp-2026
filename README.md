# FathomNetCLEF2026 @ LifeCLEF & CVPR-FGVC

Positive-Unlabeled Object Detection in Marine Images

# Overview

Underwater imaging opens a unique window into the deep sea. It is one of the few ways scientists can directly observe marine animals and habitats, creating visual records that are essential for understanding our changing ocean. But the ocean generates more data than humans can handle and computer vision is vital to process the thousands of hours of video collected by researchers around the world. Compared to terrestrial datasets, marine imagery presents many unique challenges for computer vision. 

The biggest problem? Incomplete labels. Marine biologists are experts in specific animals so they only tag what they know in each photo. This means the same image might have carefully labeled jellyfish but completely ignore the dozen other creatures swimming by. Just because something isn't labeled doesn't mean it isn't there. And that's a serious problem for anyone trying to teach a computer to recognize ocean life.
Description

See our kaggle competition page [here](https://www.kaggle.com/competitions/fathomnet-2026).

# Description
The FathomNet 2026 challenge addresses this real-world constraint by focusing on positive-unlabeled learning for object detection. Developing robust methods in this setting will reduce reliance on expert annotations, accelerate dataset curation, and broaden the usability of large-scale archives like the [FathomNet Database](https://database.fathomnet.org/fathomnet/#/). Ultimately, improving detection under positive-unlabeled regimes will support scalable biodiversity monitoring, enhance conservation strategies, and enable marine researchers to extract ecological insights from decades of imagery.

# Evaluation

Participants are tasked with developing object detection systems that can operate effectively in a positive-unlabeled learning regime, identifying both the presence and location of marine organisms across diverse habitats. The evaluation metric is the mean Average Precision (mAP), applied to the fully labeled evaluation set. For more information on mAP see [here](https://learnopencv.com/mean-average-precision-map-object-detection-model-evaluation-metric/). The evaluation notebook can be found [here](https://www.kaggle.com/code/lauravchrobak/map50-95). To succeed, participants are expected to explore approaches that leverage unlabeled positives, including positive-unlabeled learning, semi-supervised, and self-supervised techniques. The challenge not only serves to benchmark approaches in positive-unlabeled detection but also aims to produce practical methods that can generalize to large-scale, heterogeneous marine datasets, directly supporting ongoing conservation and ecological research.
# Submission Format

Submission format should resemble the following:

| annotation_id | image_id | category_id | bbox_x | bbox_y | bbox_width | bbox_height | score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 28 | 476.0 | 790.0 | 77.0 | 149.0 | 0.3 |

Submissions must be provided as a CSV file where each row corresponds to a single predicted object detection. The file must contain the following columns:
 annotation_id, image_id, category_id, bbox_x, bbox_y, bbox_width, bbox_height, score.
The annotation_id is a unique (but otherwise arbitrary) identifier for each prediction and is used only as a row index. The image_id must exactly match the image_id of the corresponding image in the COCO-formatted test set. The category_id must match the category IDs used in both the training and test datasets.
Bounding boxes are specified using the COCO object detection format: bbox_x and bbox_y give the pixel coordinates of the top-left corner of the box, while bbox_width and bbox_height specify the width and height of the box in pixels.
The score column represents the model’s confidence for the predicted detection. Scores must be floating-point values between 0 and 1 (inclusive). Predictions may be provided in any order.

### Competition Information

This competition is held jointly as part of:
The [LifeCLEF 2026 Lab](https://www.imageclef.org/LifeCLEF2026) of the CLEF 2026 conference, and of the [FGVC13 Workshop](https://sites.google.com/view/fgvc13/home) organized in conjunction with the CVPR 2026 conference. Being part of scientific research, the participants are encouraged to participate in both events. In particular, only participants who submit a working note paper to LifeCLEF (see below) will be part of the officially published ranking used for scientific communication.

This competition is part of the Fine-Grained Visual Categorization FGVC13 workshop at the Computer Vision and Pattern Recognition (CVPR) Conference 2026 in Denver, Colorado. The task results will be presented at the workshop, and the contribution of the winning team(s) will be highlighted. Attending the workshop is not required to participate in the competition. We highly encourage teams to share a public repository with their submission to aid the community in learning from one another’s approach. There is no cash prize for this competition. PLEASE NOTE: CVPR frequently sells out early, and we cannot guarantee CVPR registration after the competition's end. If you are interested in attending CVPR, please plan ahead.

You can see a list of all FGVC13 competitions [here](https://sites.google.com/view/fgvc13/competitions?authuser=0).

LifeCLEF 2026 at CLEF 2026 LifeCLEF lab is part of the Conference and Labs of the Evaluation Forum (CLEF). CLEF consists of independent peer-reviewed workshops on a broad range of challenges in multilingual and multimodal information access evaluation and benchmarking activities in various labs designed to test different aspects of mono and cross-language Information retrieval systems. CLEF 2026 will take place in Jena, Germany, on September 21-24, 2026. You can find more details on the CLEF 2026 [website](https://clef2026.clef-initiative.eu/).

Please feel free to open an issue on the competition git repo if you have a question, comment, or problem. We will also respond to threads opened on the competition discussion board here on Kaggle.

### Acknowledgments

The images for this competition have been generously provided by MBARI, the Schmidt Ocean Institute, Dawn Wright, NOAA, Ocean Networks Canada, Joost Daniels and others. Annotations were made possible through the support of FathomNet and experts from the MBARI Video Lab. Special acknowledgment is given to Marine Biologist Linda Kuhnz for her invaluable expertise in generating a substantial portion of the ground truth dataset.

# Citation
Kevin Barnard and Laura Chrobak. FathomNetCLEF2026 @ LifeCLEF & CVPR-FGVC. https://kaggle.com/competitions/fathomnet-2026, Unpublished. Kaggle.


# Data
The dataset is derived from the [FathomNet Database](https://database.fathomnet.org/fathomnet/#/), which contains more than 400k images and 1M expert-curated bounding boxes, as of Feb 2026, across thousands of marine taxonomic, geologic, and equipment classes. The dataset is formatted to adhere to the [COCO Object Detection](https://cocodataset.org/#format-data) standard. For the challenge, participants will be provided with a training set of 6,463 images with both labeled and unlabeled positive instances. The test set, by contrast, is composed of 1,425 fully annotated images. To ensure broad accessibility, the entire dataset is less than 50 GB in size.


The FathomNet Database contains a wide spectrum of class labels. To make this challenge more manageable, we have mapped the labels as they appear in the FathomNet Database to 32 consolidated groupings. These groupings were assembled by our expert team for the [FathomVerse](https://www.fathomverse.game/) game and reflect a combination of morphological and taxonomic categories. A full list of these categories can be found below.
- amphipod
- anemone
- barnacle
- benthic worm
- bivalve
- black coral
- bony fish
- brittle star
- calycophoran siphonophore
- chiton
- crab
- feather star
- hydroid
- isopod
- jelly
- larvacean
- octopus
- physonect siphonophore
- pyrosome
- sea cucumber
- sea fan
- sea pen
vsea slug
- sea snail
- sea squirt
- sea star
- shrimp
- soft coral
- sponge
- squat lobster
- stony coral
- urchin

## Data Format
The datasets are formatted to adhere to the COCO Object Detection standard. Every training image contains at least one annotation corresponding to 32 distinct categories.
We are not able to provide images as a single downloadable archive due to FathomNet Database's [Terms of Use](about:blank). Images should be downloaded using the indicated coco_url field in each of the annotation files. Participants can either use the provided download.py Python script or write their own. The download script will save one folder containing the images. The image folder contains files with names specified by the file_name field in dataset.json (typically UUID.extension format like 000b8e39-7240-49fd-9f50-713edcb28544.png). The id field is used for COCO format references between images and annotations. 
Files
dataset_train.json - the training images, annotations, and categories in COCO formatted json
dataset_test.json - the evaluation images in COCO formatted json
sample_submission.csv - a sample submission file in the correct format
download.py - python script to download imagery from FathomNet

## Files
* **`dataset_train.json`** \- the training images, annotations, and categories in COCO formatted json  
* **`dataset_test.json`** \- the evaluation images in COCO formatted json  
* **`download.py`** \- python script to download imagery from FathomNet


## Terms of Use
By downloading and using this dataset you are agreeing to FathomNet's [data use](https://www.fathomnet.org/datause) policy. In particular: The images are licensed under one of three Creative Commons licenses: CC0, CC-BY, CC-BY-NC, and CC-BY-NC-ND.  The annotations are licensed under a Creative Commons license CC0, CC-BY or CC-BY-NC. Notwithstanding the foregoing, all of the images may be used for training and development of machine learning algorithms for commercial, academic, and government purposes. Images and annotations are provided by the copyright holders and contributors "as is" and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. Please acknowledge FathomNet and MBARI when using images for publication or communication purposes regarding this competition. For all other uses of the images, users should contact the original copyright holder. For more details please see the FathomNet Database [Terms of Use](https://www.fathomnet.org/terms).

## Download

This directory contains the code and requirements for downloading the training and test datasets used for FathomNetCLEF 2026.

### Requirements

- Python 3.10+
- Packages listed in `requirements.txt` (available via PyPI)

### Installation

```bash
pip install -r requirements.txt
```

### Usage

```bash
python download.py [-h] [-o OUTPUT_DIR] [--min-workers MIN_WORKERS] [--max-workers MAX_WORKERS] [--initial-workers INITIAL_WORKERS] dataset_path
```

The script will autoscale the number of workers based on server failures.
