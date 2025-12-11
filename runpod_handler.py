import os
from upscale_job import download_to_file, upload_to_s3
from lkmn_upscaler import upscale_lkmn

def handler(event):
    input_url = event["input_url"]
    output_s3 = event["output_s3"]
    scale = int(event.get("scale", 4))

    work_dir = "/workspace/work"
    os.makedirs(work_dir, exist_ok=True)

    input_path = f"{work_dir}/input.mp4"
    output_path = f"{work_dir}/output.mp4"

    print(f"⬇️  Downloading {input_url} → {input_path}")
    download_to_file(input_url, input_path)

    print(f"⚙️ Running LKMN Upscaling (scale={scale})")
    upscale_lkmn(input_path, output_path, scale)

    print(f"⬆️ Uploading → {output_s3}")
    upload_to_s3(output_path, output_s3)

    return {
        "status": "success",
        "output": output_s3
    }
