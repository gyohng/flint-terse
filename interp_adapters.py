"""Interpolate two LoRA adapters exactly: W + (1-t)*dA + t*dB, stored as one rank-(rA+rB) adapter
(factors concatenated; mixing weights folded into lora_b). Optionally refit to a lower rank by SVD.

  python3 interp_adapters.py adapters/spark-final adapters/spark-rb5 --t 0.5 --output adapters/mix-05
  python3 interp_adapters.py adapters/mix-05 --refit 32 --output adapters/mix-05-r32
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def factors(path):
    cfg = json.loads((Path(path) / "adapter_config.json").read_text())
    w = load_file(Path(path) / "adapters.safetensors")
    scale = cfg["lora_parameters"]["scale"]
    # MLX layout: lora_a [in, r], lora_b [r, out]; delta^T = scale * a @ b
    return cfg, {k.rsplit(".", 1)[0]: (w[k.rsplit(".", 1)[0] + ".lora_a"], w[k.rsplit(".", 1)[0] + ".lora_b"] * scale)
                 for k in w if k.endswith(".lora_a")}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("a")
    p.add_argument("b", nargs="?")
    p.add_argument("--t", type=float, default=0.5)
    p.add_argument("--refit", type=int, help="SVD-refit the (single) input adapter to this rank")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    cfg, fa = factors(args.a)
    out = {}
    if args.refit:
        for name, (a, b) in fa.items():
            d = (a.double().cuda() @ b.double().cuda())  # [in, out]
            u, s, vh = torch.linalg.svd(d, full_matrices=False)
            r = args.refit
            out[name + ".lora_a"] = (u[:, :r] * s[:r].sqrt()).float().cpu().contiguous()
            out[name + ".lora_b"] = (s[:r].sqrt()[:, None] * vh[:r]).float().cpu().contiguous()
        rank = args.refit
    else:
        _, fb = factors(args.b)
        for name in fa:
            a1, b1 = fa[name]
            a2, b2 = fb[name]
            out[name + ".lora_a"] = torch.cat((a1, a2), dim=1).contiguous()
            out[name + ".lora_b"] = torch.cat(((1 - args.t) * b1, args.t * b2), dim=0).contiguous()
        rank = next(iter(out.values())).shape[1]
    o = Path(args.output)
    o.mkdir(parents=True, exist_ok=False)
    save_file(out, o / "adapters.safetensors")
    cfg["lora_parameters"].update(rank=rank, scale=1.0)
    cfg["source"] = (f"SVD refit of {args.a} to rank {rank}" if args.refit else
                     f"exact interpolation (1-{args.t})*{args.a} + {args.t}*{args.b}")
    (o / "adapter_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"{o}: rank {rank}, {len(out) // 2} projections")


if __name__ == "__main__":
    main()
