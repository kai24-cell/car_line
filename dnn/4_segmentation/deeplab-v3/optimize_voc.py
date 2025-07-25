# original https://github.com/opencv/opencv/issues/17362#issuecomment-634407860
import os
import tensorflow as tf
from tensorflow.tools.graph_transforms import TransformGraph

def get_graphdef(path):
    tf.compat.v1.reset_default_graph
    with tf.python.ops.Graph().as_default():
        with tf.io.gfile.GFile(path, 'rb') as f:
            graphdef = tf.compat.v1.GraphDef()
            graphdef.ParseFromString(f.read())
            return graphdef

def main():
    directory = os.path.dirname(__file__)
    in_graph = os.path.join(directory, "deeplabv3_mnv2_pascal_trainval/frozen_inference_graph.pb")
    out_graph = os.path.join(directory, "optimized_graph_voc.pb")
    inputs = ["sub_7"]
    outputs = ["ResizeBilinear_3"]
    transforms = [
        "remove_nodes(op=Identity)",
        "merge_duplicate_nodes",
        "strip_unused_nodes",
        "fold_constants(ignore_errors=true)",
        "fold_batch_norms",
        "fold_old_batch_norms"
    ]
    graphdef = get_graphdef(in_graph)
    optimized_graphdef = TransformGraph(graphdef, inputs, outputs, transforms)
    tf.io.write_graph(optimized_graphdef, logdir=directory, as_text=False, name=out_graph)
    print("[success] transform optimize graph {0}".format(out_graph))

if __name__ == '__main__':
    main()