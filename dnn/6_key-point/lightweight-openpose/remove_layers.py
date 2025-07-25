import os
import onnx
import json
from google.protobuf.json_format import MessageToJson
from google.protobuf.json_format import Parse

def main():
    directory = os.path.dirname(__file__)
    output_name = os.path.join(directory, "human-pose-estimation.onnx")
    onnx_model = onnx.load(output_name)
    json_str = MessageToJson(onnx_model)
    json_obj = json.loads(json_str)

    outputs = json_obj["graph"]["output"]
    remove_layers = ["stage_0_output_1_heatmaps", "stage_0_output_0_pafs", "stage_1_output_0_pafs"]
    json_obj["graph"]["output"] = [output for output in outputs if output["name"] not in remove_layers]

    nodes = json_obj["graph"]["node"]
    remove_nodes = ["Conv_114", "Relu_115", "Conv_116"]
    json_obj["graph"]["node"] = [node for node in nodes if node["name"] not in remove_nodes]

    json_str = json.dumps(json_obj)
    onnx_model = Parse(json_str, onnx.ModelProto())
    onnx.save(onnx_model, output_name)
    print("[success] remove unnecessary layers {0}".format(output_name))

if __name__ == '__main__':
    main()
