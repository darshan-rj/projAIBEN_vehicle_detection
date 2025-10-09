import onnxruntime as ort
import numpy as np
import time
from PIL import Image
from torchvision import transforms
import yaml

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def preprocess(image_path, input_size):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor()
    ])
    tensor = transform(image).unsqueeze(0).numpy()
    return tensor

def benchmark_onnx():
    cfg = load_config()
    session = ort.InferenceSession(cfg['model']['onnx_output'])
    input_name = session.get_inputs()[0].name

    image_path = f"{cfg['dataset']['root']}/samples/CAM_FRONT/n015-2018-11-21-19-38-26+0800__CAM_FRONT__1542800854412460.jpg"
    input_tensor = preprocess(image_path, cfg['model']['input_size'])

    times = []
    for _ in range(50):
        start = time.time()
        outputs = session.run(None, {input_name: input_tensor})
        times.append(time.time() - start)

    print(f"ONNX Inference: Avg = {np.mean(times)*1000:.2f} ms, Min = {np.min(times)*1000:.2f} ms, Max = {np.max(times)*1000:.2f} ms")

if __name__ == "__main__":
    benchmark_onnx()