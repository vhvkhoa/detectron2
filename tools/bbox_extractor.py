
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import atexit
import bisect
import multiprocessing as mp
from collections import deque
import torch

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor


class BboxExtractor(object):
    def __init__(self, cfg, sampling_rate=8, target_fps=30, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.target_fps = target_fps
        self.sampling_rate = sampling_rate

        print('Construct bounding box extractor with sampling rate: {0} on fps: {1}'.format(self.sampling_rate, self.target_fps))

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)

    def _frame_from_video(self, video, num_frames):
        for frame_idx in range(num_frames):
            success, frame = video.read()
            if success:
                yield frame
            else:
                yield None

    def run_on_video(self, video, num_frames, fps):
        """
        Visualizes predictions on frames of the input video.

        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.

        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        target_sampling_rate = self.sampling_rate * fps / self.target_fps
        try:
            sampling_pts = torch.arange(
                target_sampling_rate / 2,
                num_frames + 1 - target_sampling_rate / 2,
                target_sampling_rate).tolist()
        except RuntimeError:
            print(
                'Cannot make sampling list.\n\tVideo length: %f frames.\n\tTarget_sampling_rate: %f frames.'
                % (num_frames, target_sampling_rate))
            sampling_pts = []

        if target_sampling_rate * len(sampling_pts) < num_frames:
            sampling_pts.append((target_sampling_rate + num_frames) / 2)

        frame_gen = self._frame_from_video(video, num_frames)
        if self.parallel:
            buffer_size = self.predictor.default_buffer_size

            frame_data = deque()

            sampling_idx, fails_count = 0, 0
            for idx, frame in enumerate(frame_gen):
                if frame is None:
                    fails_count += 1
                    continue

                if sampling_idx < len(sampling_pts) and idx >= sampling_pts[sampling_idx]:
                    sampling_idx += 1
                    frame_data.append((idx, frame))
                    self.predictor.put(frame)

                    if idx >= buffer_size:
                        idx, frame = frame_data.popleft()
                        yield idx, self.predictor.get()['instances'].to(self.cpu_device)

            if fails_count != 0:
                print('Failed to read %d frames.' % fails_count)

            while len(frame_data):
                idx, frame = frame_data.popleft()
                yield idx, self.predictor.get()['instances'].to(self.cpu_device)
        else:
            sampling_idx, fails_count = 0, 0
            for idx, frame in enumerate(frame_gen):
                if frame is None:
                    fails_count += 1

                if sampling_idx < len(sampling_pts) and idx >= sampling_pts[sampling_idx]:
                    sampling_idx += 1
                    instances = self.predictor(frame)['instances'].to(self.cpu_device)
                    if frame is None:
                        print(frame)
                        print(instances)
                    yield idx, instances

            if fails_count != 0:
                print('Failed to read %d frames.' % fails_count)



class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5
