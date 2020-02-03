import os
import json
import argparse
from glob import glob

import cv2
import numpy as np

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.colormap import random_color
from detectron2.utils.visualizer import Visualizer

import pycocotools.mask as mask_util

class _DetectedInstance:
    """
    Used to store data about detected objects in video frame,
    in order to transfer color to objects in the future frames.

    Attributes:
        label (int):
        bbox (tuple[float]):
        mask_rle (dict):
        color (tuple[float]): RGB colors in range (0, 1)
        ttl (int): time-to-live for the instance. For example, if ttl=2,
            the instance color can be transferred to objects in the next two frames.
    """

    __slots__ = ["label", "bbox", "mask_rle", "color", "ttl"]

    def __init__(self, label, bbox, mask_rle, color, ttl):
        self.label = label
        self.bbox = bbox
        self.mask_rle = mask_rle
        self.color = color
        self.ttl = ttl


def _create_text_labels(classes, scores, class_names):
    """
    Args:
        classes (list[int] or None):
        scores (list[float] or None):
        class_names (list[str] or None):

    Returns:
        list[str] or None
    """
    labels = None
    if classes is not None and class_names is not None and len(class_names) > 1:
        labels = [class_names[i] for i in classes]
    if scores is not None:
        if labels is None:
            labels = ["{:.0f}%".format(s * 100) for s in scores]
        else:
            labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]
    return labels


def _assign_colors(instances, old_instances):
    """
    Naive tracking heuristics to assign same color to the same instance,
    will update the internal state of tracked instances.

    Returns:
        list[tuple[float]]: list of colors.
    """

    # Compute iou with either boxes or masks:
    is_crowd = np.zeros((len(instances),), dtype=np.bool)
    if instances[0].bbox is None:
        assert instances[0].mask_rle is not None
        # use mask iou only when box iou is None
        # because box seems good enough
        rles_old = [x.mask_rle for x in old_instances]
        rles_new = [x.mask_rle for x in instances]
        ious = mask_util.iou(rles_old, rles_new, is_crowd)
        threshold = 0.5
    else:
        boxes_old = [x.bbox for x in old_instances]
        boxes_new = [x.bbox for x in instances]
        ious = mask_util.iou(boxes_old, boxes_new, is_crowd)
        threshold = 0.6
    if len(ious) == 0:
        ious = np.zeros((len(old_instances), len(instances)), dtype="float32")

    # Only allow matching instances of the same label:
    for old_idx, old in enumerate(old_instances):
        for new_idx, new in enumerate(instances):
            if old.label != new.label:
                ious[old_idx, new_idx] = 0

    matched_new_per_old = np.asarray(ious).argmax(axis=1)
    max_iou_per_old = np.asarray(ious).max(axis=1)

    # Try to find match for each old instance:
    extra_instances = []
    for idx, inst in enumerate(old_instances):
        if max_iou_per_old[idx] > threshold:
            newidx = matched_new_per_old[idx]
            if instances[newidx].color is None:
                instances[newidx].color = inst.color
                continue
        # If an old instance does not match any new instances,
        # keep it for the next frame in case it is just missed by the detector
        inst.ttl -= 1
        if inst.ttl > 0:
            extra_instances.append(inst)

    # Assign random color to newly-detected instances:
    for inst in instances:
        if inst.color is None:
            inst.color = random_color(rgb=True, maximum=1)
    old_instances = instances[:] + extra_instances
    return [d.color for d in instances], old_instances


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "bbox_dir",
        help="directory containing bounding boxes json files. Must have same tree structure as video_input.")
    parser.add_argument(
        "video_dir",
        help="directory containing videos corresponding to bounding boxes json files. Must have same tree structure as bbox_input.")
    parser.add_argument(
        "output_dir",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--captured_class_ids",
        nargs='+',
        help="Class ids list to be caputred, ignore those that are not in this list, must be at the same length as captured_class_names."
    )
    parser.add_argument(
        "--captured_class_names",
        nargs='+',
        help="Class names list corresponding with captured class ids, must be at the same length as captured_class_ids."
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    cfg = setup_cfg(args)
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused")
    print(cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused")

    alpha = 0.5

    for root, _, file_names in os.walk(args.bbox_dir):
        for file_name in file_names:
            print('Processing video %s.' % file_name)
            with open(os.path.join(root, file_name), 'r') as f:
                video_bboxes = json.load(f)['bboxes']

            video_path = glob(os.path.join(args.video_dir, os.path.relpath(root, start=args.bbox_dir), os.path.splitext(file_name)[0] + '.*'))
            assert len(video_path) == 1, 'There exists more than one video path.'

            video_input = cv2.VideoCapture(video_path[0])
            width = int(video_input.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video_input.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frames_per_second = video_input.get(cv2.CAP_PROP_FPS)
            num_frames = int(video_input.get(cv2.CAP_PROP_FRAME_COUNT))
            secs_per_frame = 1. / frames_per_second

            video_output_path = os.path.join(args.output_dir, os.path.relpath(video_path[0], start=args.video_dir))
            if not os.path.isdir(os.path.dirname(video_output_path)):
                os.makedirs(os.path.dirname(video_output_path))

            video_output = cv2.VideoWriter(
                filename=video_output_path + '.mp4',
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )

            bbox_idx, old_instances = 0, []
            for frame_idx in tqdm(range(num_frames)):
                frame_idx_secs = frame_idx * secs_per_frame
                _, frame = video_input.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                frame_visualizer = Visualizer(frame, metadata)

                if bbox_idx < len(video_bboxes) - 1:
                    start, end = video_bboxes[bbox_idx]['idx_secs'], video_bboxes[bbox_idx + 1]['idx_secs']
                    if frame_idx_secs >= (start + end) / 2:
                        bbox_idx += 1

                if len(video_bboxes[bbox_idx]['bboxes']) > 0:
                    boxes, classes, scores = [], [], []
                    for frame_bbox in video_bboxes[bbox_idx]['bboxes']:
                        boxes.append(frame_bbox['box'])
                        classes.append(frame_bbox['class_id'])
                        scores.append(frame_bbox['score'])

                    detected = [
                        _DetectedInstance(bbox['class_id'], bbox['box'], mask_rle=None, color=None, ttl=8)
                        for bbox in video_bboxes[bbox_idx]['bboxes']
                    ]
                    colors, old_isntances = _assign_colors(detected, old_instances)

                    labels = _create_text_labels(classes, scores, metadata.get("thing_classes", None))

                    keep_ids = []
                    for i, label in enumerate(labels):
                        if label.split() == 'person':
                            keep_ids.append(i)

                    labels = [labels[i] for i in keep_ids]
                    boxes = [boxes[i] for i in keep_ids]

                    frame_visualizer.overlay_instances(
                        boxes=boxes,  # boxes are a bit distracting
                        labels=labels,
                        assigned_colors=colors,
                        alpha=alpha,
                    )

                vis_frame = cv2.cvtColor(frame_visualizer.output.get_image(), cv2.COLOR_RGB2BGR)
                video_output.write(vis_frame)

            video_input.release()
            video_output.release()
