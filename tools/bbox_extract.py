
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import multiprocessing as mp
import os
import cv2
import json
import glob

from tqdm import tqdm

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
    parser.add_argument("input_dir", help="A directory contains videos to extract bboxes from.")
    parser.add_argument(
        "output_dir",
        help="A directory to save bboxes coordinations."
    )
    parser.add_argument(
        "--input_list",
        default="",
        type=str,
        help="List os input videos to extract bboxes from (Optional)."
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
        "--target-frames",
        dest="target_frames",
        type=int,
        default=100,
        help="Target frames to extract",
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

    if os.path.isfile(args.input_list):
        path_to_videos = []
        with open(args.input_list, 'r') as f:
            video_list = json.load(f)
            for video_name in video_list:
                video_path = os.path.join(args.input_dir, video_name)
                if os.path.isfile(video_path):
                    path_to_videos.append(video_path)

    else:
        path_to_videos = glob.glob(os.path.join(args.input_dir, '*'))

    for video_path in tqdm(path_to_videos):
        basename = os.path.basename(video_path)
        output_path = os.path.join(args.output_dir, os.path.splitext(basename)[0] + '.json')
        if os.path.exists(output_path):
            continue

        video = cv2.VideoCapture(video_path)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        video_bboxes = []
        for idx, frame_preds in bbox_extractor.run_on_video(video, num_frames, frames_per_second):
            frame_bboxes = []

            boxes = frame_preds.pred_boxes.tensor.tolist() if frame_preds.has("pred_boxes") else None
            scores = frame_preds.scores.tolist() if frame_preds.has("scores") else None
            classes = frame_preds.pred_classes.tolist() if frame_preds.has("pred_classes") else None

            for box, score, class_id in zip(boxes, scores, classes):
                if (args.captured_class_ids is not None) and (class_id not in args.captured_class_ids):
                    continue
                frame_bboxes.append({'box': box, 'score': score, 'class_id': class_id})

            video_bboxes.append({'pts': idx / frames_per_second, 'frame_bboxes': frame_bboxes})

        video.release()

        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir)
        with open(output_path, 'w') as f:
            json.dump({
                'num_frames': num_frames,
                'secs_per_frame': 1. / frames_per_second,
                'video_bboxes': video_bboxes
            }, f)

        if len(video_bboxes) != args.target_frames:
            print('Warning! Number of frames processed does not equal to 100.')
