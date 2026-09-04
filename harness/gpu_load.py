"""Stimulus: a square-wave accelerator load. Builds a synthetic compute-heavy
ONNX graph (chained MatMul+Relu) and runs it in bursts on the chosen execution
provider, emitting an event log (unix_ns, state) for correlation.

Track A: device="dml" -> Radeon 880M iGPU (DirectML).
Track B: device="ryzenai" (once the AMD Ryzen AI SDK / VitisAI EP is installed)
         -> XDNA 2 NPU.

--sham keeps the schedule but skips compute: the null control.
"""
import os
import csv
import time
import argparse
import numpy as np


def build_model(path, dim=1024, depth=8):
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    nodes, inits = [], []
    helper.make_tensor_value_info("X", TensorProto.FLOAT, [dim, dim])
    prev = "X"
    for i in range(depth):
        w = (np.random.randn(dim, dim).astype(np.float32) * 0.02)
        inits.append(numpy_helper.from_array(w, name=f"W{i}"))
        nodes.append(helper.make_node("MatMul", [prev, f"W{i}"], [f"M{i}"]))
        nodes.append(helper.make_node("Relu", [f"M{i}"], [f"R{i}"]))
        prev = f"R{i}"
    inp = helper.make_tensor_value_info("X", TensorProto.FLOAT, [dim, dim])
    out = helper.make_tensor_value_info(prev, TensorProto.FLOAT, [dim, dim])
    graph = helper.make_graph(nodes, "load", [inp], [out], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, path)
    return "X", prev


PROVIDERS = {
    "dml": ["DmlExecutionProvider", "CPUExecutionProvider"],
    "cpu": ["CPUExecutionProvider"],
    "ryzenai": ["VitisAIExecutionProvider", "CPUExecutionProvider"],
}


def make_session(model_path, device):
    import onnxruntime as ort
    want = PROVIDERS.get(device, ["CPUExecutionProvider"])
    avail = ort.get_available_providers()
    use = [p for p in want if p in avail] or ["CPUExecutionProvider"]
    sess = ort.InferenceSession(model_path, providers=use)
    return sess, sess.get_providers()


def run(sess, inp_name, dim, cycles, on_s, off_s, out_csv, sham=False):
    X = np.random.rand(dim, dim).astype(np.float32)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["unix_ns", "state"])
        for _ in range(cycles):
            w.writerow([time.time_ns(), 1])
            t_end = time.time() + on_s
            while time.time() < t_end:
                if sham:
                    time.sleep(0.001)
                else:
                    sess.run(None, {inp_name: X})
            w.writerow([time.time_ns(), 0])
            time.sleep(off_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="dml", choices=list(PROVIDERS))
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--cycles", type=int, default=20)
    ap.add_argument("--on", type=float, default=15)
    ap.add_argument("--off", type=float, default=15)
    ap.add_argument("--sham", action="store_true")
    ap.add_argument("--model", default="_load.onnx")
    ap.add_argument("--out", default="events.csv")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        build_model(args.model, args.dim, args.depth)
    inp_name, _ = None, None
    import onnxruntime as ort  # noqa
    sess, providers = make_session(args.model, args.device)
    inp_name = sess.get_inputs()[0].name
    print(f"[gpu_load] providers actually used: {providers}")
    run(sess, inp_name, args.dim, args.cycles, args.on, args.off,
        args.out, args.sham)
    print(f"[gpu_load] wrote {args.out}")


if __name__ == "__main__":
    main()
