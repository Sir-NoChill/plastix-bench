#!/usr/bin/env python3
"""Generate the shared workload for benchmark 10 (engineered sparse large NN).

Produces `topology.bin`: a deep (~1000-layer), sparse, *irregular* DAG plus a
streaming-regression data stream. All three implementations (plastix / cpp /
pytorch) load this same file so the workload is identical and comparable.

Wire format (little-endian):
  magic "ESLN" (4 bytes) | u32 version=1
  u32 n_in | u32 n_units | u32 n_edges | u32 n_steps | u32 output_id
  layer[n_units]            : u32 each (layer index of every unit)
  edges[n_edges]            : (u32 src, u32 dst), src.layer < dst.layer
  data[n_steps]             : (f32 x[n_in], f32 target)

Units 0..n_in-1 are inputs (layer 0). The output unit is `output_id`. Hidden
units sit in layers 1..n_layers-1 with IRREGULAR fan-in: each draws FANIN
sources from *any* earlier layer (skip connections), so connectivity is sparse
and not layer-adjacent. The teacher target is y = tanh(w_teacher · x_t).

Growth/shrink is NOT baked into the file — it is parametric (see README): every
GROW_EVERY steps each impl adds GROW_UNITS units (FANIN incoming edges from
earlier units + one edge to the output) and prunes PRUNE_EDGES edges, capped at
MAX_UNITS, using the shared LCG in the README so the schedule matches.
"""
import argparse, struct
import numpy as np


def build(n_in, n_layers, units_per_layer, fanin, n_steps, seed):
    rng = np.random.default_rng(seed)
    # Layer assignment: inputs in layer 0; hidden layers 1..n_layers-1; output last.
    layers = [0] * n_in
    for L in range(1, n_layers):
        layers += [L] * units_per_layer
    output_id = len(layers)
    layers.append(n_layers)  # output sits one past the last hidden layer
    layers = np.array(layers, dtype=np.uint32)
    n_units = len(layers)

    # Index of units by layer, for drawing earlier-layer sources.
    by_layer = {}
    for u, L in enumerate(layers):
        by_layer.setdefault(int(L), []).append(u)
    earlier = []  # units strictly before layer L (cumulative)
    cum = []
    for L in range(0, n_layers + 1):
        earlier.append(list(cum))
        cum += by_layer.get(L, [])

    edges = []
    for u in range(n_in, n_units):
        L = int(layers[u])
        pool = earlier[L]
        if not pool:
            continue
        k = min(fanin, len(pool))
        srcs = rng.choice(len(pool), size=k, replace=False)
        for si in srcs:
            edges.append((int(pool[si]), u))
    # Ensure the output is reachable: give it extra fan-in from scattered units.
    out_extra = rng.choice(output_id, size=min(8 * fanin, output_id), replace=False)
    for s in out_extra:
        edges.append((int(s), output_id))
    edges = np.array(edges, dtype=np.uint32)

    # Teacher + data stream.
    w_teacher = rng.standard_normal(n_in).astype(np.float32) / np.sqrt(n_in)
    X = rng.standard_normal((n_steps, n_in)).astype(np.float32)
    Y = np.tanh(X @ w_teacher).astype(np.float32)
    return layers, edges, output_id, X, Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="10_engineered_sparse_large_nn/topology.bin")
    ap.add_argument("--n-in", type=int, default=32)
    ap.add_argument("--n-layers", type=int, default=1000)
    ap.add_argument("--units-per-layer", type=int, default=4)
    ap.add_argument("--fanin", type=int, default=4)
    ap.add_argument("--n-steps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    layers, edges, output_id, X, Y = build(
        a.n_in, a.n_layers, a.units_per_layer, a.fanin, a.n_steps, a.seed)
    with open(a.out, "wb") as f:
        f.write(b"ESLN")
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<IIIII", a.n_in, len(layers), len(edges),
                            a.n_steps, output_id))
        f.write(layers.tobytes())
        f.write(edges.tobytes())
        rec = np.empty((a.n_steps, a.n_in + 1), dtype=np.float32)
        rec[:, :a.n_in] = X
        rec[:, a.n_in] = Y
        f.write(rec.tobytes())
    print(f"[gen] units={len(layers)} edges={len(edges)} layers={a.n_layers} "
          f"steps={a.n_steps} output_id={output_id} -> {a.out}")


if __name__ == "__main__":
    main()
