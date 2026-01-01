import modal

sam2_image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel")
    .env(
        {
            "PYTHONUNBUFFERED": "1",
            "SAM2_BUILD_ALLOW_ERRORS": "0",
            "CUDA_HOME": "/usr/local/cuda",
            "UV_SYSTEM_PYTHON": "1",
            "TORCH_CUDA_ARCH_LIST": "7.0 7.2 7.5 8.0 8.6 8.9 9.0",
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
        "git",
    )
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .workdir("/sam2modalwebapp")
    .run_commands("git clone -b Modal https://github.com/ORNLxUTK/sam2.git")
    .add_local_file(
        "pyproject.toml", remote_path="/sam2modalwebapp/pyproject.toml", copy=True
    )
    .run_commands("$HOME/.local/bin/uv sync -v")
    .add_local_dir("sam2/training", remote_path="/sam2modalwebapp/sam2/training")
)

app = modal.App(name="sam2modalwebapp")


@app.function(image=sam2_image)
def listfiles():
    import os

    for dirpath, dirnames, filesnames in os.walk("/sam2modalwebapp/sam2"):
        print(f"Directory: {dirpath}")
        for filename in filesnames:
            print(f"  {dirpath}/{filename}")
