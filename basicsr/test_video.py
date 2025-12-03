import logging

import numpy as np
import torch
from os import path as osp
import os
import multiprocessing as mp

import sys

repo_path = "/backup/github/LKMN/basicsr"   # adjust to correct path
#sys.path.insert(0, repo_path)

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils.options import dict2str, parse_options
import cv2
import subprocess


import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset
from tqdm import tqdm

import multiprocessing
from multiprocessing import shared_memory
import pickle
import time
import shutil
import tempfile

import torch.nn as nn

class SRWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.net_g = model.net_g
        self.net_g.eval()

    def forward(self, x):
        return self.net_g(x).clamp(0, 1)


def encode_video(video_array, output_path, fps, num_chunks=8):
    """
    Encode a NumPy array (N x H x W x C, RGB24) to MP4 using parallel ffmpeg processes.
    """
    N = video_array.shape[0]
    chunk_size = (N + num_chunks - 1) // num_chunks  # ceiling division

    tempdir = tempfile.mkdtemp()
    chunk_files = [os.path.join(tempdir, f"chunk_{i}.mp4") for i in range(num_chunks)]

    # Launch parallel encoders
    procs = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, N)
        if start >= end:
            break
        frames = video_array[start:end]
        p = multiprocessing.Process(
            target=save_video_ffmpeg_communicate,
            args=(frames, chunk_files[i], fps)
        )
        procs.append(p)
        p.start()

    for p in procs:
        p.join()

    # Write concat list
    list_file = os.path.join(tempdir, "file_list.txt")
    with open(list_file, "w") as f:
        for fpath in chunk_files:
            if os.path.exists(fpath):
                f.write(f"file '{fpath}'\n")

    # Concatenate into final output
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", list_file,
        "-c", "copy",
        output_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Cleanup
    shutil.rmtree(tempdir)
    print(f"✅ Video saved to {output_path}")

def save_video_ffmpeg_communicate(frames_np, filepath, fps, codec="libx264"):
    """
    Save all frames to a video using ffmpeg via stdin in one shot using communicate().

    Args:
        frames: list or np.ndarray of shape (N,H,W,3), BGR, uint8 or float [0,255]
        filepath: output file
        fps: frame rate
        codec: 'libx264' or 'libx265'
    """
    N, H, W, C = frames_np.shape
    assert C == 3, "Frames must be RGB"
    start = time.perf_counter()
    # FFmpeg command
    crf = "13"

    cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}",
        "-r", str(fps),
        "-i", "-",  # input from stdin
        "-an",
        "-c:v", codec,
        "-crf", crf,
        "-preset", 'fast',
        filepath
    ]

    '''
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}",
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-c:v", "hevc_nvenc",  # instead of h264_nvenc
        "-preset", "p6",
        "-rc", "vbr_hq",
        "-cq", "0",
        "-b:v", "200M",
        "-maxrate", "250M",
        "-bufsize", "500M",
        filepath
    ]
    '''
    # Launch process
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Send all frames at once and wait for completion
    out, err = proc.communicate(input=frames_np.tobytes())
    end = time.perf_counter()
    print(f'ffmpeg encoding command takes {end - start : .2f}s')
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg encoding failed:\n{err.decode()}")


gt_video = '/data/dataset/pexel/videos_SD24K/1c677a10-709f-496a-8eee-92e524ee723d_2_2_trim.mp4'

def inference_time_check(opt):
    device = 'cuda'
    lkmn_model = build_model(opt)
    model = SRWrapper(lkmn_model).to(device)

    batch_size = 2
    n_tries = 20
    x = torch.rand(batch_size, 3, 1200, 2200).to(device)  # image size(1,3,320,192) iDetection
    if opt['half']:
        model = model.half()
        x = x.half()

    model.eval()

    scripted_model = torch.jit.script(model)
    scripted_model.save("superres_x2.pt")

    model = torch.compile(model)

    with torch.no_grad():
        torch_out_np = model(x).cpu().numpy()

    # pytorch model timing
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(x)

    # Timing
    start = time.perf_counter()
    for _ in tqdm(range(n_tries)):
        with torch.no_grad():
            _ = model(x)
    torch.cuda.synchronize()
    end = time.perf_counter()
    print("PyTorch avg time:", (end - start) / n_tries / batch_size, "s")

#@profile
def run_ddp_line_perf(world_size, opt, shm_name, shm_shape, shm_dtype, per_gpu_batch_size, scale, hr_shm_name):
    rank = 0

    start_time = time.perf_counter()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "41453"  # any free port
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)

    #dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # Build model
    lkmn_model = build_model(opt)
    model = SRWrapper(lkmn_model).to(rank)
    #model = torch.compile(model)

    #model = DDP(model, device_ids=[rank], output_device=rank)
    model.eval()

    # Reconstruct shared NumPy array (no copy!)
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    lq_frames_rgb_np = np.ndarray(shm_shape, dtype=shm_dtype, buffer=existing_shm.buf)

    n_per_chunk = lq_frames_rgb_np.shape[0] // world_size + 1
    rank_start = n_per_chunk * rank
    rank_end = min(rank_start + n_per_chunk, lq_frames_rgb_np.shape[0])
    input_np = lq_frames_rgb_np[rank_start:rank_end,...]

    input_tensor = torch.from_numpy(input_np)
    input_tensor = input_tensor.to(rank, non_blocking=True)
    input_tensor = input_tensor / 255.
    input_tensor = input_tensor.permute(0, 3, 1, 2)
    if opt['half']:
        input_tensor = input_tensor.half()
    N, c, h, w = input_tensor.shape

    hr_shm = shared_memory.SharedMemory(name=hr_shm_name)
    hr_frames_rgb_np = np.ndarray([lq_frames_rgb_np.shape[0], scale * h, scale * w, c], dtype=shm_dtype, buffer=hr_shm.buf)

    end_time = time.perf_counter()
    print(f'to preparing model input takes {end_time - start_time: .2f}s')

    start_time = time.perf_counter()
    copy_stream = torch.cuda.Stream(rank)
    results = []

    with torch.no_grad():
        for batch_start_idx in range(0, N, per_gpu_batch_size):
            batch_end_idx = min(batch_start_idx + per_gpu_batch_size, N)

            buf_tensor = model(input_tensor[batch_start_idx:batch_end_idx])
            cpu_buf = torch.empty(
                (batch_end_idx - batch_start_idx, scale * h, scale * w, c),
                dtype=torch.uint8,
                device="cpu",
                pin_memory=True,
            )
            torch.cuda.synchronize()

            # 3. Launch async copy in separate stream
            with torch.cuda.stream(copy_stream):
                tmp = (buf_tensor.permute(0, 2, 3, 1) * 255).to(torch.uint8)
                cpu_buf.copy_(tmp, non_blocking=True)

            # 5. Store for later NumPy conversion
            results.append((batch_start_idx, batch_end_idx, cpu_buf))

    # 6. Synchronize once at the end
    torch.cuda.synchronize()

    # 7. Convert safely to NumPy
    for start, end, cpu_buf in results:
        hr_frames_rgb_np[start:end] = cpu_buf.numpy()
    end_time = time.perf_counter()
    print(f'running model takes {end_time - start_time: .2f}s')

    #dist.destroy_process_group()

def run_ddp(rank, world_size, opt, shm_name, shm_shape, shm_dtype, per_gpu_batch_size, scale, hr_shm_name):
    start_time = time.perf_counter()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "41453"  # any free port
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # Build model
    lkmn_model = build_model(opt)
    model = SRWrapper(lkmn_model).to(rank)
    #model = torch.compile(model)

    model = DDP(model, device_ids=[rank], output_device=rank)
    model.eval()

    # Reconstruct shared NumPy array (no copy!)
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    lq_frames_rgb_np = np.ndarray(shm_shape, dtype=shm_dtype, buffer=existing_shm.buf)

    n_per_chunk = lq_frames_rgb_np.shape[0] // world_size + 1
    rank_start = n_per_chunk * rank
    rank_end = min(rank_start + n_per_chunk, lq_frames_rgb_np.shape[0])
    input_np = lq_frames_rgb_np[rank_start:rank_end,...]

    input_tensor = torch.from_numpy(input_np)
    input_tensor = input_tensor.to(rank, non_blocking=True)
    input_tensor = input_tensor / 255.
    input_tensor = input_tensor.permute(0, 3, 1, 2)
    if opt['half']:
        input_tensor = input_tensor.half()
    N, c, h, w = input_tensor.shape

    hr_shm = shared_memory.SharedMemory(name=hr_shm_name)
    hr_frames_rgb_np = np.ndarray([lq_frames_rgb_np.shape[0], scale * h, scale * w, c], dtype=shm_dtype, buffer=hr_shm.buf)

    end_time = time.perf_counter()
    print(f'to preparing model input takes {end_time - start_time: .2f}s')

    start_time = time.perf_counter()
    copy_stream = torch.cuda.Stream(rank)
    results = []

    with torch.no_grad():
        for batch_start_idx in range(0, N, per_gpu_batch_size):
            batch_end_idx = min(batch_start_idx + per_gpu_batch_size, N)

            buf_tensor = model(input_tensor[batch_start_idx:batch_end_idx])
            cpu_buf = torch.empty(
                (batch_end_idx - batch_start_idx, scale * h, scale * w, c),
                dtype=torch.uint8,
                device="cpu",
                pin_memory=True,
            )
            torch.cuda.synchronize()

            # 3. Launch async copy in separate stream
            with torch.cuda.stream(copy_stream):
                tmp = (buf_tensor.permute(0, 2, 3, 1) * 255).to(torch.uint8)
                cpu_buf.copy_(tmp, non_blocking=True)

            # 5. Store for later NumPy conversion
            results.append((batch_start_idx, batch_end_idx, cpu_buf))

    # 6. Synchronize once at the end
    torch.cuda.synchronize()

    # 7. Convert safely to NumPy
    for start, end, cpu_buf in results:
        hr_frames_rgb_np[rank_start + start: rank_start + end] = cpu_buf.numpy()
    end_time = time.perf_counter()
    print(f'running model takes {end_time - start_time: .2f}s')

    dist.destroy_process_group()

def test_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, args = parse_options(root_path, is_train=False)

    #inference_time_check(opt)

    #torch.backends.cudnn.benchmark = True

    start = time.perf_counter()
    cap = cv2.VideoCapture(args.input_LR_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Get width and height
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Get number of frames
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    lq_frames_rgb_np_shape = (frame_count, height, width, 3)
    shm = shared_memory.SharedMemory(create=True, size=np.prod(lq_frames_rgb_np_shape) * np.dtype(np.uint8).itemsize)
    shared_array = np.ndarray(lq_frames_rgb_np_shape, dtype=np.uint8, buffer=shm.buf)

    hr_frames_rgb_np_shape = (frame_count, height * args.scale, width * args.scale, 3)
    hr_shm = shared_memory.SharedMemory(create=True, size=np.prod(hr_frames_rgb_np_shape) * np.dtype(np.uint8).itemsize)
    hr_shared_array = np.ndarray(hr_frames_rgb_np_shape, dtype=np.uint8, buffer=hr_shm.buf)

    success, frame = cap.read()
    cur_frame_idx = 0
    while success:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        shared_array[cur_frame_idx] = frame_rgb
        success, frame = cap.read()
        cur_frame_idx += 1

    # Release the video capture object
    cap.release()
    if args.scale == 2:
        with open('opt_x2.pkl', 'wb') as f:
            pickle.dump(opt, f)

        opt['path']['pretrain_network_g'] = '/data/github/LKMN/models/net_g_x2.pth'
    if args.scale == 4:
        with open('opt_x4.pkl', 'wb') as f:
            pickle.dump(opt, f)

        opt['path']['pretrain_network_g'] = '/data/github/LKMN/models/net_g_x4.pth'

    end = time.perf_counter()
    print(f'{shared_array.shape} video reading takes {end - start: .2f}s')

    start = time.perf_counter()
    print(f'starting model inference...')

    world_size = torch.cuda.device_count()
    if world_size <= 1:
        raise RuntimeError("Need multiple GPUs for this code")
    print(f'using {world_size} GPUs')

    mp.spawn(run_ddp,
             args=(world_size, opt, shm.name, shared_array.shape, shared_array.dtype, 6, args.scale, hr_shm.name), # 6
             nprocs=world_size,
             join=True)

    # run_ddp_line_perf(world_size, opt, shm.name, shared_array.shape, shared_array.dtype, 6, args.scale, hr_shm.name)
    end = time.perf_counter()

    print(f'model inferencing takes {end - start: .2f}s')

    start_time = time.perf_counter()

    encode_video(hr_shared_array, args.output_HR_video, fps, 4)
    print(f'processed {hr_shared_array.shape} frames')

    file_size_bytes = os.path.getsize(args.output_HR_video)  # size in bytes
    file_size_mb = file_size_bytes / (1024 * 1024)  # convert to MB
    print(f"File size: {file_size_mb:.2f} MB")

    end_time = time.perf_counter()
    print(f'video encoding takes {end_time - start_time: .2f}s')

    try:
        shm.close()
    except:
        pass
    try:
        shm.unlink()
    except:
        pass

if __name__ == '__main__':
    start = time.perf_counter()
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
    end = time.perf_counter()
    print(f'total processing takes {end - start: .2f}s')

'''
-opt /data/github/LKMN/options/test/test_LKMN_x4.yml --input_LR_video /data/dataset/pexel/pexel_chunk/1494286_1_downscale.mp4 --output_HR_video demo.mp4 --scale 4
-opt /data/github/LKMN/options/test/test_LKMN_x2.yml --input_LR_video /data/dataset/pexel/pexel_chunk/29477440_0_downscale.mp4 --output_HR_video demo.mp4 --scale 2
-opt /data/github/LKMN/options/test/test_LKMN_x2.yml --input_LR_video /data/dataset/pexel/pexel_chunk/1494286_1_downscale.mp4 --output_HR_video demo.mp4 --scale 2
'''
