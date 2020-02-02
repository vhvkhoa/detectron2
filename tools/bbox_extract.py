
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import multiprocessing as mp
import os
import cv2
import json
import glob

from tqdm import tqdm
import torch

from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from bbox_extractor import BboxExtractor

# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument("input_dir", help="A list of space separated input videos")
    parser.add_argument(
        "output_dir",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--captured_class_ids",
        nargs='+',
        help="Class ids list to be caputred, ignore those that are not in this list."
    )
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--sampling-rate",
        dest="sampling_rate",
        type=int,
        default=8,
        help="Target FPS to extract bboxes",
    )
    parser.add_argument(
        "--target-fps",
        dest="target_fps",
        type=int,
        default=30,
        help="Target FPS to extract bboxes",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    bbox_extractor = BboxExtractor(cfg, sampling_rate=args.sampling_rate, target_fps=args.target_fps)

    for video_path in tqdm(glob.glob(os.path.join(args.input_dir, '*'))):
        video = cv2.VideoCapture(video_path)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(video_path)

        video_bboxes = []
        for frame_preds in bbox_extractor.run_on_video(video, frames_per_second):
            frame_bboxes = []

            boxes = frame_preds.pred_boxes.tensor.tolist() if frame_preds.has("pred_boxes") else None
            scores = frame_preds.scores.tolist() if frame_preds.has("scores") else None
            classes = frame_preds.pred_classes.tolist() if frame_preds.has("pred_classes") else None

            for box, score, class_id in zip(boxes, scores, classes):
                if (args.captured_class_ids is not None) and (class_id not in args.captured_class_ids):
                    continue
                frame_bboxes.append({'box': box, 'score': score, 'class_id': class_id})

            video_bboxes.append(frame_bboxes)

        video.release()

        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir)
        with open(os.path.join(args.output_dir, os.path.splitext(basename)[0] + '.json'), 'w') as f:
            print(len(video_bboxes))
            json.dump(video_bboxes, f)
