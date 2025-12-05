# lkmn_upscaler.py

import os
import cv2
import time
import shutil
import numpy as np
import subprocess
import multiprocessing as mp
from multiprocessing import shared_memory

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from basicsr.models import build_model
from basicsr.utils.options import yaml_load



# ----------------------------------------------
# Load LKMN Options
# ----------------------------------------------
def load_lkmn_options(yaml_path, scale):
    """
    Loads LKMN YAML and returns an inference-only config dictionary.
    Removes all training, dataset, and experiment logic.
    """

    opt = yaml_load(yaml_path)

    # enforce inference mode
    opt["is_train"] = False
    opt["dist"] = False
    opt["rank"] = 0
    opt["world_size"] = 1

    # Inject scale
    opt["scale"] = scale

    # fix paths (basicSR expects absolute paths)
    if "path" in opt:
        for k, v in opt["path"].items():
            if v is not None:
                if isinstance(v, str):
                    opt["path"][k] = os.path.abspath(v)
                else:
                    opt["path"][k] = v

    # ensure options exist
    if "manual_seed" not in opt:
        opt["manual_seed"] = 42

    return opt

# ----------------------------------------------
# Model Wrapper
# ----------------------------------------------
class SRWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.net_g = model.net_g
        self.net_g.eval()

    def forward(self, x):
        return self.net_g(x).clamp(0, 1)


# ----------------------------------------------
# FFmpeg Encoding
# ----------------------------------------------
def encode_video_parallel(rgb_array, output_path, fps, chunks=8):

    N = rgb_array.shape[0]
    chunk_size = (N + chunks - 1) // chunks

    temp_dir = os.path.dirname(output_path) + "/enc_temp"
    os.makedirs(temp_dir, exist_ok=True)

    chunk_files = []

    procs = []
    for i in range(chunks):
        start = i * chunk_size
        end = min((i+1) * chunk_size, N)
        if start >= end:
            continue

        sub_arr = rgb_array[start:end]
        out_path = f"{temp_dir}/chunk_{i}.mp4"
        chunk_files.append(out_path)

        p = mp.Process(
            target=encode_chunk,
            args=(sub_arr, out_path, fps)
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    # concat
    list_file = f"{temp_dir}/list.txt"
    with open(list_file, "w") as f:
        for p in chunk_files:
            f.write(f"file '{p}'\n")

    subprocess.run([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", list_file,
        "-c", "copy",
        output_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    shutil.rmtree(temp_dir)


def encode_chunk(frames, output_path, fps):

    N, H, W, _ = frames.shape

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}",
        "-r", str(fps),
        "-i", "-",
        "-vcodec", "libx264",
        "-preset", "fast",
        "-crf", "13",
        "-loglevel", "quiet",
        output_path,
    ]

    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    p.stdin.write(frames.tobytes())
    p.stdin.close()
    p.wait()


# ----------------------------------------------
# Worker for multi-GPU inference
# ----------------------------------------------
def gpu_worker(rank, world_size, opt, shm_name, lq_shape, dtype, scale, batch_size, hr_shm_name):

    torch.cuda.set_device(rank)

    # init dist
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # build model
    lkmn = build_model(opt)
    model = SRWrapper(lkmn).to(rank).eval()
    model = DDP(model, device_ids=[rank], output_device=rank)

    # shared memory → numpy array
    shm = shared_memory.SharedMemory(name=shm_name)
    frames_np = np.ndarray(lq_shape, dtype=dtype, buffer=shm.buf)

    # hr shared memory
    hr_shm = shared_memory.SharedMemory(name=hr_shm_name)
    H = lq_shape[1]
    W = lq_shape[2]
    hr_shape = (lq_shape[0], H*scale, W*scale, 3)
    hr_frames = np.ndarray(hr_shape, dtype=np.uint8, buffer=hr_shm.buf)

    # split work
    total = lq_shape[0]
    per_rank = (total + world_size - 1) // world_size
    start = rank * per_rank
    end = min(start + per_rank, total)
    if start >= end:
        return

    # load frames
    inp = torch.from_numpy(frames_np[start:end]).cuda(rank).float() / 255.0
    inp = inp.permute(0, 3, 1, 2)

    # inference loop
    with torch.no_grad():
        out_all = []
        for i in range(0, inp.shape[0], batch_size):
            batch = inp[i:i+batch_size]
            out = model(batch).clamp(0, 1)
            out = (out.permute(0, 2, 3, 1) * 255).byte().cpu().numpy()
            out_all.append(out)

        out_all = np.concatenate(out_all, axis=0)
        hr_frames[start:end] = out_all

    dist.destroy_process_group()


# ----------------------------------------------
# Worker for CPU inference
# ----------------------------------------------
def cpu_worker(rank, world_size, opt, shm_name, shm_shape, shm_dtype, per_cpu_batch_size, scale, hr_shm_name):

    print(f"[CPU Worker {rank}] Started")

    # Load model on CPU
    opt['num_gpu'] = 0
    model = build_model(opt)
    model = SRWrapper(model)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # CPU cannot use FP16 → ensure FP32
    if device.type == "cpu":
        model = model.float()

    
    model.eval()

    # Access shared memory inputs
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    lq_frames = np.ndarray(shm_shape, dtype=shm_dtype, buffer=existing_shm.buf)

    # Determine slice for this worker
    n = shm_shape[0]
    chunk = (n + world_size - 1) // world_size
    start = rank * chunk
    end = min(start + chunk, n)

    # Prepare output shared memory
    hr_shm = shared_memory.SharedMemory(name=hr_shm_name)
    out_h = shm_shape[1] * scale
    out_w = shm_shape[2] * scale
    out_frames = np.ndarray((n, out_h, out_w, 3), dtype=np.uint8, buffer=hr_shm.buf)

    # Run inference (CPU)
    for idx in range(start, end):
        frame = lq_frames[idx]

        x = torch.from_numpy(frame).float().to(device).permute(2,0,1).unsqueeze(0) / 255.0

        with torch.no_grad():
            sr = model(x)[0].permute(1,2,0).numpy()
            sr = np.clip(sr * 255, 0, 255).astype(np.uint8)

        out_frames[idx] = sr

    print(f"[CPU Worker {rank}] Finished frames {start}..{end}")


# ----------------------------------------------
# ⚡ Main Upscale Function (used by CPU)
# ----------------------------------------------
def test_pipeline_cpu(input_video_path, output_video_path, scale=4):

    # load video
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    ok, frame = cap.read()
    while ok:
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ok, frame = cap.read()
    cap.release()

    frames_np = np.array(frames, dtype=np.uint8)
    N, H, W, _ = frames_np.shape

    # load config
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "options/test"))
    opt = load_lkmn_options(os.path.join(root, f"test_LKMN_x{scale}.yml"), scale)

    # update weights
    opt['path']['pretrain_network_g'] = os.path.abspath(
        f"./models/net_g_x{scale}.pth"
    )

    # shared memory for LQ + HR
    shm = shared_memory.SharedMemory(create=True, size=frames_np.nbytes)
    shm_np = np.ndarray(frames_np.shape, dtype=np.uint8, buffer=shm.buf)
    shm_np[:] = frames_np[:]

    hr_shape = (N, H*scale, W*scale, 3)
    hr_bytes = np.prod(hr_shape)
    hr_shm = shared_memory.SharedMemory(create=True, size=hr_bytes)

    # spawn processes
    world_size = 1
    cpu_worker(
        rank=0,
        world_size=world_size,
        opt=opt,
        shm_name=shm.name,
        shm_shape=frames_np.shape,
        shm_dtype=frames_np.dtype,
        per_cpu_batch_size=1,
        scale=scale,
        hr_shm_name=hr_shm.name,
    )

    # load hr output
    out_np = np.ndarray(hr_shape, dtype=np.uint8, buffer=hr_shm.buf)

    # encode
    # Save output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (W * scale, H * scale))

    for frame in out_np:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr)
    writer.release()

    shm.close()
    shm.unlink()
    hr_shm.close()
    hr_shm.unlink()

    return output_video_path

# ----------------------------------------------
# ⚡ Main Upscale Function (used by Vast.ai)
# ----------------------------------------------
def upscale_lkmn(input_video_path, output_video_path, scale=4, batch_size=6):

    # load video
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    ok, frame = cap.read()
    while ok:
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ok, frame = cap.read()
    cap.release()

    frames_np = np.array(frames, dtype=np.uint8)
    N, H, W, _ = frames_np.shape

    # load config
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "options/test"))
    opt = load_lkmn_options(os.path.join(root, f"test_LKMN_x{scale}.yml"), scale)

    # update weights
    opt['path']['pretrain_network_g'] = os.path.abspath(
        f"./models/net_g_x{scale}.pth"
    )

    # shared memory for LQ + HR
    shm = shared_memory.SharedMemory(create=True, size=frames_np.nbytes)
    shm_np = np.ndarray(frames_np.shape, dtype=np.uint8, buffer=shm.buf)
    shm_np[:] = frames_np[:]

    hr_shape = (N, H*scale, W*scale, 3)
    hr_bytes = np.prod(hr_shape)
    hr_shm = shared_memory.SharedMemory(create=True, size=hr_bytes)

    # spawn processes
    world_size = torch.cuda.device_count()
    mp.spawn(
        gpu_worker,
        args=(
            world_size, opt,
            shm.name, frames_np.shape, frames_np.dtype,
            scale, batch_size,
            hr_shm.name
        ),
        nprocs=world_size,
        join=True
    )

    # load hr output
    out_np = np.ndarray(hr_shape, dtype=np.uint8, buffer=hr_shm.buf)

    # encode
    encode_video_parallel(out_np, output_video_path, fps)

    shm.close()
    shm.unlink()
    hr_shm.close()
    hr_shm.unlink()

    return output_video_path
