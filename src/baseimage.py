import modal

BASE_IMAGE = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel")
    .env(
        {
            "PYTHONUNBUFFERED": "1",
            "SAM2_BUILD_ALLOW_ERRORS": "0",
            "CUDA_HOME": "/usr/local/cuda",
            "UV_SYSTEM_PYTHON": "1",
            "TORCH_CUDA_ARCH_LIST": "7.0 7.2 7.5 8.0 8.6 8.9 9.0",
            "PYTHONPATH": "/sam2modalwebapp/.venv/lib/python3.11/site-packages:$PYTHONPATH",
        },
    )
    .apt_install(
        "ffmpeg",
        "libavutil-dev",
        "libavcodec-dev",
        "libavformat-dev",
        "libswscale-dev",
        "pkg-config",
        "build-essential",
        "g++-11",
        "gcc-11",
        "libffi-dev",
        "curl",
        "ca-certificates",
        "wget",
        "git",
    )
    .workdir("/sam2modalwebapp")
    .run_commands("git clone -b Modal https://github.com/ORNLxUTK/sam2.git")
    .workdir("/sam2modalwebapp/sam2")
    .run_commands("pip install -e '.[dev]'")
    .workdir("/sam2modalwebapp/sam2/checkpoints")
    .run_commands("sh download_ckpts.sh")
    .workdir("/sam2modalwebapp")
    .add_local_dir("sam2/", remote_path="/sam2modalwebapp/sam2/")
)
