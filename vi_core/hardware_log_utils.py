# Get GPU name if it exists
import subprocess


def get_gpu_name():
    try:
        gpu_info = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], encoding='utf-8')
        gpu_name = gpu_info.strip()
        return gpu_name
    except subprocess.CalledProcessError:
        # If 'nvidia-smi' command fails, it means no GPU is available
        return None
        