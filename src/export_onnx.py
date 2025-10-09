import os
import torch
import yaml
from effdet import get_efficientdet_config, EfficientDet
from effdet.efficientdet import HeadNet


def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_model(variant, num_classes, input_size, ckpt_path):
    config = get_efficientdet_config(variant)
    config.num_classes = num_classes
    config.image_size = (input_size, input_size)
    config.max_det_per_image = 1000
    config.soft_nms = False
    config.use_nms = False  # ✅ disables NMS for ONNX compatibility

    net = EfficientDet(config, pretrained_backbone=False)
    net.class_net = HeadNet(config, num_outputs=num_classes)
    model = net
    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


class ONNXWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


def export_to_onnx():
    cfg = load_config()
    variant = cfg['model'].get('variant', 'tf_efficientdet_d0')
    input_size = cfg['model']['input_size']
    num_classes = cfg['model']['num_classes']
    ckpt_path = os.path.join(cfg['model']['checkpoint_dir'], "best_model.pth")
    output_path = cfg['model']['onnx_output']

    model = create_model(variant, num_classes, input_size, ckpt_path)
    wrapped_model = ONNXWrapper(model)
    dummy_input = torch.randn(1, 3, input_size, input_size)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    torch.onnx.export(
        wrapped_model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['class_out', 'box_out'],
        opset_version=12
    )
    print(f"✅ ONNX model exported to {output_path}")


if __name__ == "__main__":
    export_to_onnx()