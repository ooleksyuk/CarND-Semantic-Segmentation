import subprocess

def run_command(cmd):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    out, err = p.communicate()
    result = out.split(b'\n')
    for lin in result:
        if not lin.startswith(b'#'):
            print(lin)

run_command("summarize_graph --in_graph=./runs/normal/model.pb")
run_command("benchmark_model "
            "--input_layer='image_input:0' "
            "--graph=./runs/normal/model.pb "
            "--show_flops=true "
            "--show_summary=true "
            "--input_layer=image_input")

run_command('python -m tensorflow.python.tools.freeze_graph '
            '--input_graph=./runs/normal/model.pb '
            '--input_meta_graph=./runs/normal/model.meta '
            '--input_binary=true '
            '--input_checkpoint=./runs/normal/model '
            '--output_graph=./runs/freeze/model.pb '
            '--input_names=image_input '
            '--output_node_names=my_logits')
run_command("summarize_graph --in_graph=./runs/freeze/model.pb")
run_command("benchmark_model "
            "--input_layer='image_input:0' "
            "--graph=./runs/freeze/model.pb "
            "--show_flops=true "
            "--show_summary=true "
            "--input_layer=image_input")

run_command('python -m tensorflow.python.tools.optimize_for_inference '
            '--input=./runs/freeze/model.pb '
            '--output=./runs/optimized/model.pb '
            '--frozen_graph=True '
            '--input_names=image_input '
            '--output_names=my_logits')
run_command("summarize_graph --in_graph=./runs/optimized/model.pb")

run_command("transform_graph "
            "--in_graph=./runs/freeze/model.pb "
            "--out_graph=./runs/eight_bit/model.pb "
            "--inputs=image_input "
            "--outputs=my_logits "
            "--transforms=' add_default_attributes remove_nodes(op=Identity, op=CheckNumerics) fold_constants(ignore_errors=true) fold_batch_norms fold_old_batch_norms fuse_resize_and_conv quantize_weights quantize_nodes strip_unused_nodes sort_by_execution_order'")
run_command("summarize_graph --in_graph=./runs/eight_bit/model.pb")

import re
from google.protobuf import text_format
from tensorflow.core.framework import graph_pb2

def convert_graph_to_dot(input_graph, output_dot, is_input_graph_binary):
    graph = graph_pb2.GraphDef()
    with open(input_graph, "rb") as fh:
        if is_input_graph_binary:
            graph.ParseFromString(fh.read())
        else:
            text_format.Merge(fh.read(), graph)
    with open(output_dot, "wt") as fh:
        print("digraph graphname {", file=fh)
        for node in graph.node:
            output_name = node.name
            print("  \"" + output_name + "\" [label=\"" + node.op + "\"];", file=fh)
            for input_full_name in node.input:
                parts = input_full_name.split(":")
                input_name = re.sub(r"^\^", "", parts[0])
                print("  \"" + input_name + "\" -> \"" + output_name + "\";", file=fh)
        print("}", file=fh)
        print("Created dot file '%s' for graph '%s'." % (output_dot, input_graph))

normal_input_graph = './runs/normal/model.pb'
normal_output_dot = './runs/normal/graph.dot'
convert_graph_to_dot(input_graph=normal_input_graph, output_dot=normal_output_dot, is_input_graph_binary=True)

freeze_input_graph = './runs/freeze/model.pb'
freeze_output_dot = './runs/freeze/graph.dot'
convert_graph_to_dot(input_graph=freeze_input_graph, output_dot=freeze_output_dot, is_input_graph_binary=True)

optimized_input_graph = './runs/optimized/model.pb'
optimized_output_dot = './runs/optimized/graph.dot'
convert_graph_to_dot(input_graph=optimized_input_graph, output_dot=optimized_output_dot, is_input_graph_binary=True)

eight_bit_input_graph = './runs/eight_bit/model.pb'
eight_bit_output_dot = './runs/eight_bit/graph.dot'
convert_graph_to_dot(input_graph=eight_bit_input_graph, output_dot=eight_bit_output_dot, is_input_graph_binary=True)

normal_dot_to_png = "dot -O -T png " + normal_output_dot
print(normal_dot_to_png)
run_command(normal_dot_to_png) # + " -o " + normal_pb_output_dot + ".png > /tmp/a.out"
print("normal pb graph png created")

freeze_dot_to_png = "dot -O -T png " + freeze_output_dot
print(freeze_dot_to_png)
run_command(freeze_dot_to_png) # + " -o " + freeze_output_dot + ".png > /tmp/a.out"
print("freeze graph png created")

optimized_dot_to_png = "dot -O -T png " + optimized_output_dot
print(optimized_dot_to_png)
run_command(optimized_dot_to_png) # + " -o " + optimized_output_dot + ".png > /tmp/a.out"
print("optimized graph png created")

eight_bit_dot_to_png = "dot -O -T png " + eight_bit_output_dot
print(eight_bit_dot_to_png)
run_command(eight_bit_dot_to_png) # + " -o " + eight_bit_output_dot + ".png > /tmp/a.out"
print("eight bit graph png created")