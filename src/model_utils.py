# src/model_utils.py

from effdet.bench import DetBenchTrain

def wrap_model(model, config):
    """
    Wrap EfficientDet model with DetBenchTrain for both training and evaluation.
    """
    return DetBenchTrain(model, config)