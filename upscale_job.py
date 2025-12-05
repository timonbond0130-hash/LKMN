# upscale_job.py
import argparse
import os
import subprocess
import tempfile
import boto3
import requests
from lkmn_upscaler import upscale_lkmn, test_pipeline_cpu

def download_to_file(url: str, dst_path: str):
    if url.startswith("s3://"):
        # S3 → local
        s3 = boto3.client("s3")
        _, path = url.split("s3://", 1)
        bucket, key = path.split("/", 1)
        s3.download_file(bucket, key, dst_path)
    else:
        # HTTP/HTTPS → local
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(dst_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

def upload_to_s3(src_path: str, s3_uri: str):
    assert s3_uri.startswith("s3://")
    s3 = boto3.client("s3")
    _, path = s3_uri.split("s3://", 1)
    bucket, key = path.split("/", 1)
    s3.upload_file(src_path, bucket, key)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-url", required=True, help="HTTP or s3:// URL of input video")
    parser.add_argument("--output-s3", required=True, help="s3://bucket/key for result")
    parser.add_argument("--scale", type=int, default=4, choices=[2, 4])
    args = parser.parse_args()

    work_dir = os.path.abspath("./work")
    os.makedirs(work_dir, exist_ok=True)

    input_path = os.path.join(work_dir, "input.mp4")
    output_path = os.path.join(work_dir, "output_4k.mp4")

    print(f"⬇️  Downloading {args.input_url} → {input_path}")
    # download_to_file(args.input_url, input_path)

    # TODO: call your LKMN multi-GPU pipeline here.
    # For now we just leave a placeholder:
    print("⚙️  Running LKMN upscaling (placeholder)...")
    # You will replace this with your real function, like:
    upscale_lkmn(input_path, output_path, scale=args.scale)

    # Temporary placeholder: just copy input to output
    subprocess.run(["cp", input_path, output_path], check=True)

    print(f"⬆️  Uploading {output_path} → {args.output_s3}")
    upload_to_s3(output_path, args.output_s3)
    print("✅ Done.")

if __name__ == "__main__":
    main()
