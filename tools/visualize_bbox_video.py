import os
import json
import argparse
from glob import glob

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog


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
            with open(os.path.join(root, file_name), 'r') as f:
                video_bboxes = json.load(f)['bboxes']

            video_path = glob(os.path.join(args.video_dir, os.path.relpath(root, start=args.bbox_dir), os.path.splitext(file_name)[0] + '.*'))
            print(video_path)
            assert len(video_path) == 0, 'There exists more than one video path.'

            video = cv2.VideoCapture(video_path[0])
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frames_per_second = video.get(cv2.CAP_PROP_FPS)
            num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            secs_per_frame = 1. / frames_per_second

            video_output_path = os.path.join(args.output_dir, os.path.relpath(video_path[0], start=args.video_dir))
            if not os.path.isdir(os.path.dirname(video_output_path)):
                os.makedirs(os.path.dirname(video_output_path))

            video_output = cv2.VideoWriter(
                filename=video_output_path,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*"x264"),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )

            video_visualizer = VideoVisualizer(metadata, self.instance_mode)

            bbox_idx = 0
            for frame_idx in range(num_frames):
                frame_idx_secs = frame_idx * secs_per_frame
                _, frame = video.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                if bbox_idx < len(video_bboxes):
                    start, end = video_bboxes[bbox_idx]['idx_secs'], video_bboxes[bbox_idx + 1]['idx_secs']
                    if frame_idx_secs >= (start + end) / 2:
                        bbox_idx += 1

                bboxes, classes, scores = [], [], []
                for frame_bbox in video_bboxes[bbox_idx]['bboxes']:
                    bboxes.append(frame_bbox['box'])
                    classes.append(frame_bbox['class_id'])
                    scores.append(frame_bbox['score'])

                detected = [
                    _DetectedInstance(bbox['class_id'], bbox['box'], mask_rle=None, color=None, ttl=8)
                    for bbox in video_bboxes[bbox_idx]['bboxes']
                ]
                colors = self._assign_colors(detected)

                labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))

                keep_ids = []
                for i, label in enumerate(labels):
                    if label.split() == 'person':
                        keep_ids.append(i)
                
                labels = [labels[i] for i in keep_ids]
                bboxes= [bboxes[i] for i in keep_ids]
                
                frame_visualizer = Visualizer(frame, metadata)
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
