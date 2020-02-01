import os.path as osp
import glob
from subprocess import Popen, PIPE
import argparse
import tqdm

OPTS = 'MODEL.WEIGHTS model_final_f6e8b1.pkl'
DEFAULT_CMD = 'python tools/bbox_extract.py {0} {1}  --confidence-threshold {2} --config-file {3} --opts {4}'
def main(args):
    video_path_list = glob.glob(osp.join(args.input_dir, '*'))
    processes = [{
        'video_name': osp.basename(video_path_list[i]),
        'command': Popen(DEFAULT_CMD.format(video_path_list[i], args.output_dir, args.thresh, args.cfg, OPTS), stdout=PIPE, stderr=PIPE),
        'finished': False
    } for i in range(min(args.num_workers, len(video_path_list)))]

    pbar = tqdm.tqdm(total=len(processes))
    current_video_idx = len(processes)
    pbar.update(current_video_idx)

    while current_video_idx < len(video_path_list):
        for i, process in enumerate(processes):
            if process['finished'] == True:
                cmd = DEFAULT_CMD.format(video_path, args.output_dir, args.thresh, args.cfg, OPTS)
                processes[i] = {
                    'video_name': osp.basename(video_path_list[current_video_idx]),
                    'command': Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True),
                    'finished': False
                }
                current_video_idx += 1
                pbar.update(1)

        for i, process in enumerate(processes):
            if process['command'].poll() is not None:
                if not osp.exists(osp.join(args.output, osp.splitext(process['video_name'])[0] + '.json')):
                    tqdm.write('some error happened at video %s.' % process['video_name'])
                processes[i]['finished'] = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument("input_dir", help="A list of space separated input videos")
    parser.add_argument(
        "output_dir",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--config-file",
        dest='cfg',
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--confidence-threshold",
        dest='thresh',
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument('--num_workers', default=8, type=int, help='Number of threads working simultaneously')

    main(parser.parse_args())
