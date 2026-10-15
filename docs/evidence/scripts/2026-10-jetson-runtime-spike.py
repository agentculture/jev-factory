"""t1 Jetson runtime spike: one forward pass of Qwen3.5-2B's text torso, then torchao
weight-only int8 / int4 on its linears. Prints one JSON line per step; never raises."""

import glob
import json
import os
import platform
import sys
import time
import traceback

BASE = sys.argv[1] if len(sys.argv) > 1 else "/data/pin-cache/Qwen__Qwen3.5-2B-Base"
SEQ = int(os.environ.get("SPIKE_SEQ", "512"))


def emit(step, **kw):
    print(json.dumps({"step": step, **kw}), flush=True)


def meminfo():
    out = {}
    with open("/proc/meminfo") as fh:
        for line in fh:
            k, v = line.split(":", 1)
            if k in ("MemTotal", "MemAvailable", "SwapFree"):
                out[k] = int(v.split()[0]) // 1024
    return out


def facts():
    import torch

    rel = open("/etc/nv_tegra_release").read().splitlines()[0] if os.path.exists(
        "/etc/nv_tegra_release") else None
    import transformers

    try:
        import torchao

        ao = torchao.__version__
    except Exception as exc:  # noqa: BLE001
        ao = f"import failed: {exc}"
    cuda = torch.cuda.is_available()
    emit("facts", python=platform.python_version(), torch=torch.__version__,
         torch_cuda=torch.version.cuda, transformers=transformers.__version__, torchao=ao,
         cuda_available=cuda,
         device=torch.cuda.get_device_name(0) if cuda else None,
         capability=list(torch.cuda.get_device_capability(0)) if cuda else None,
         arch_list=torch.cuda.get_arch_list() if cuda else None,
         nv_tegra_release=rel, mem_mib=meminfo())


def load_text_torso(torch):
    from safetensors.torch import load_file
    from transformers import AutoConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m

    from safetensors import safe_open

    del load_file
    cfg = AutoConfig.from_pretrained(BASE).text_config
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("cuda"):
        model = m.Qwen3_5TextModel(cfg)
    torch.set_default_dtype(torch.float32)
    params = dict(model.named_parameters())
    prefix = "model.language_model."
    seen = set()
    unexpected = []
    with torch.no_grad():
        for f in sorted(glob.glob(os.path.join(BASE, "*.safetensors"))):
            with safe_open(f, framework="pt", device="cpu") as fh:
                for k in fh.keys():
                    if not k.startswith(prefix):
                        continue
                    name = k[len(prefix):]
                    if name not in params:
                        unexpected.append(name)
                        continue
                    params[name].copy_(fh.get_tensor(k).to(params[name].dtype))
                    seen.add(name)
    missing = [n for n in params if n not in seen]
    model.eval()
    return model, cfg, sorted(missing), sorted(unexpected)


def forward(torch, model, ids, label):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    out = None
    with torch.inference_mode():
        for _ in range(3):
            t0 = time.perf_counter()
            out = model(input_ids=ids).last_hidden_state
            torch.cuda.synchronize()
            times.append(round((time.perf_counter() - t0) * 1000, 1))
    emit(label, ok=True, ms=times, peak_alloc_mib=torch.cuda.max_memory_allocated() // 2**20,
         mem_mib=meminfo())
    return out.float()


def try_quant(torch, make_model, ids, ref, kind):
    try:
        from torchao.quantization import quantize_

        import torchao.quantization as q

        model = make_model()
        if kind == "int8":
            conf = q.Int8WeightOnlyConfig()
        else:
            conf = q.Int4WeightOnlyConfig(group_size=128)
        is_lin = lambda mod, fqn: isinstance(mod, torch.nn.Linear) and "in_proj_a" not in fqn \
            and "in_proj_b" not in fqn  # noqa: E731
        quantize_(model, conf, filter_fn=is_lin)
        out = forward(torch, model, ids, f"forward_{kind}")
        cos = torch.nn.functional.cosine_similarity(out[0, -1], ref[0, -1], dim=0).item()
        emit(f"quant_{kind}", ok=True, last_token_cosine_vs_bf16=round(cos, 5))
        del model
        torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        emit(f"quant_{kind}", ok=False, error=f"{type(exc).__name__}: {exc}"[:600],
             tb=traceback.format_exc()[-1200:])


def main():
    try:
        facts()
    except Exception as exc:  # noqa: BLE001
        emit("facts", ok=False, error=str(exc))
        return
    import torch

    try:
        model, cfg, missing, unexpected = load_text_torso(torch)
        emit("load_bf16", ok=True, missing=missing[:10], unexpected=unexpected[:10],
             layer_types=sorted(set(cfg.layer_types)), mem_mib=meminfo())
        try:
            import fla  # noqa: F401

            fast = True
        except Exception:  # noqa: BLE001
            fast = False
        emit("deltanet_path", fla_importable=fast)
        ids = torch.randint(0, cfg.vocab_size, (1, SEQ), device="cuda")
        ref = forward(torch, model, ids, "forward_bf16")
        del model
        torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        emit("load_bf16", ok=False, error=f"{type(exc).__name__}: {exc}"[:600],
             tb=traceback.format_exc()[-1500:])
        return
    make = lambda: load_text_torso(torch)[0]  # noqa: E731
    for kind in ("int8", "int4"):
        try_quant(torch, make, ids, ref, kind)


if __name__ == "__main__":
    main()
