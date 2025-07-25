import os
import sys

directory = os.path.dirname(__file__)
os.path.join(directory, "lightweight-human-pose-estimation.pytorch")
sys.path.append("lightweight-human-pose-estimation.pytorch")

import torch
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.load_state import load_state

def main():
    checkpoint_path = os.path.join(directory, "checkpoint_iter_370000.pth")
    output_name = os.path.join(directory, "human-pose-estimation.onnx")

    net = PoseEstimationWithMobileNet()
    model = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    load_state(net, model)

    dummy_input = torch.randn(1, 3, 256, 456)
    input_names = ['data']
    output_names = [
        'stage_0_output_1_heatmaps',
        'stage_0_output_0_pafs',
        'stage_1_output_1_heatmaps',
        'stage_1_output_0_pafs']
    torch.onnx.export(
        net, dummy_input, output_name, verbose=True, input_names=input_names, output_names=output_names)
    print("[success] export onnx format {0}".format(output_name))

if __name__ == '__main__':
    main()
