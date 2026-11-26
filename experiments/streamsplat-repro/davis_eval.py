"""G8 helper, run inside the StreamSplat image by run_G8_davis-val-authors-metric.sh.

  keycheck                                   : load the released weights into the authors' training SplatModel on
                                               the CPU and check the strict=False load (no unexpected key; missing
                                               keys only in the DINO condition encoder and the LPIPS / tracker
                                               modules, which the checkpoint never holds).
  depth [--bf16] <frames-root> <depth-root> <seq>...
                                             : davis_bear.py depth for each sequence not yet marked .done;
                                               --bf16 (G16) runs the depth model under bf16 autocast.
  eval [--bf16] <frames-root> <depth-root> <jsonl> <split> <seq>...
                                             : every 6-frame window i..i+5 of each sequence, sampled and loaded as
                                               datasets/provider_davis.py does unshuffled (output_frames = 6 gives
                                               interval 1), through SplatModel.forward; records the authors' own
                                               metrics['psnr_given'] (the two keyframes) and ['psnr_novel'] (the
                                               middle 4), both 256x256 bilinear as compute_losses makes them.
                                               --bf16 (G12, C2 of the G10 plan) runs the forward under bf16
                                               autocast; the renderer and compute_losses stay fp32
                                               (model/splat_model.py disables autocast around them).
  metrics <jsonl> <json>                     : the mean over windows per split, and per-sequence mean, min, max.
  compare-depth <root> <reference-root> <json> <seq>...
                                             : (G16) every depth map of root against the reference's, in 0..255
                                               levels: per map the mean and max absolute difference and the
                                               share of equal pixels, then the mean over maps and the overall max.
"""
import json, sys
from pathlib import Path


def load_model(device):
    import torch
    from safetensors.torch import load_file
    from configs.options_inference import Options  # pred_keys / fix_keys must match the released heads
    from model.splat_model import SplatModel  # the training model: its forward computes the paper's metrics
    torch.set_float32_matmul_precision("high")
    opt = Options()
    opt.input_frames, opt.output_frames, opt.epoch, opt.compile = 1, 6, 0, False
    assert (opt.image_height, opt.image_width) == (288, 512), (opt.image_height, opt.image_width)
    model = SplatModel(opt).to(device)
    weights = {key.replace("_orig_mod.", ""): value for key, value in load_file("checkpoints/streamsplat.safetensors", device=device).items()}
    missing, unexpected = model.load_state_dict(weights, strict=False)
    allowed = ("condition_encoder", "lpips_loss", "tracker_prior")
    assert not unexpected and not [key for key in missing if not any(part in key for part in allowed)], (missing, unexpected)
    print("missing key groups:", sorted({next(part for part in allowed if part in key) for key in missing}))
    return model.eval()


def keycheck():
    import torch
    real_tensor = torch.tensor
    torch.tensor = lambda *args, device=None, **kwargs: real_tensor(*args, **kwargs)  # SplatModel's background colour
    load_model("cpu")
    print("KEYCHECK_OK")


def depth(frames_root, depth_root, sequences, bf16=False):
    from davis_bear import depth as depth_one
    for sequence in sequences:
        done = Path(depth_root) / sequence / ".done"
        if not done.exists():
            depth_one(str(Path(frames_root) / sequence), str(Path(depth_root) / sequence), bf16)
            done.touch()


def evaluate(frames_root, depth_root, jsonl, split, sequences, bf16=False):
    import torch, torchvision.transforms as tf
    from PIL import Image
    model = load_model("cuda")
    image = tf.Compose([tf.Resize((288, 512)), tf.ToTensor()])  # DAVISDataset.transform
    nearest = tf.Compose([tf.Resize((288, 512), interpolation=tf.InterpolationMode.NEAREST), tf.ToTensor()])
    with open(jsonl, "a") as out, torch.no_grad():
        for sequence in sequences:
            frames = sorted(Path(frames_root, sequence).glob("*.jpg"), key=lambda path: float(path.stem))
            for start in range(len(frames) - 5):
                window = frames[start:start + 6]
                depth_path = lambda path: Path(depth_root, sequence, f"{path.stem}_pred.png")
                # the first frame's depth is resized nearest, the others bilinear, as __getitem__ does
                depths = [nearest(Image.open(depth_path(window[0])))] + [image(Image.open(depth_path(path))) for path in window[1:]]
                data = {"frames": torch.stack([image(Image.open(path).convert("RGB")) for path in window])[None].cuda(),
                        "depths": torch.stack(depths)[None].cuda(),
                        "timestamps": torch.arange(6, dtype=torch.float32)[None].cuda(), "supv_masks": None}
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                    metrics = model(data)["metrics"]
                out.write(json.dumps({"split": split, "seq": sequence, "start": start,
                                      "psnr_given": metrics["psnr_given"].item(), "psnr_novel": metrics["psnr_novel"].item()}) + "\n")
            print(f"{split} {sequence}: {len(frames) - 5} windows", flush=True)


def compare_depth(root, reference_root, json_path, sequences):
    import numpy as np
    from PIL import Image
    maps = []
    for sequence in sequences:
        for path in sorted(Path(reference_root, sequence).glob("*_pred.png")):
            ours = np.asarray(Image.open(Path(root, sequence, path.name)), dtype=np.int16)
            gap = np.abs(ours - np.asarray(Image.open(path), dtype=np.int16))
            maps.append({"seq": sequence, "map": path.name, "mean_abs_levels": float(gap.mean()),
                         "max_abs_levels": int(gap.max()), "equal_share": float((gap == 0).mean())})
    result = {"maps": len(maps), "mean_abs_levels": float(np.mean([entry["mean_abs_levels"] for entry in maps])),
              "max_abs_levels": max(entry["max_abs_levels"] for entry in maps),
              "equal_share_mean": float(np.mean([entry["equal_share"] for entry in maps])), "per_map": maps}
    Path(json_path).write_text(json.dumps(result, indent=1))
    print(json.dumps({key: value for key, value in result.items() if key != "per_map"}))


def summarise(jsonl, metrics_path):
    import numpy as np
    rows = [json.loads(line) for line in open(jsonl)]
    result = {}
    for split in sorted({row["split"] for row in rows}):
        part = [row for row in rows if row["split"] == split]
        sequences = {}
        for sequence in sorted({row["seq"] for row in part}):
            values = {key: [row[key] for row in part if row["seq"] == sequence] for key in ("psnr_given", "psnr_novel")}
            sequences[sequence] = {"windows": len(values["psnr_given"]),
                                   **{f"{key}_{stat}": float(fn(vals)) for key, vals in values.items()
                                      for stat, fn in (("mean", np.mean), ("min", np.min), ("max", np.max))}}
        result[split] = {"windows": len(part), "sequences": len(sequences),
                         "psnr_given_mean": float(np.mean([row["psnr_given"] for row in part])),
                         "psnr_novel_mean": float(np.mean([row["psnr_novel"] for row in part])),
                         "per_sequence": sequences}
    Path(metrics_path).write_text(json.dumps(result, indent=1))
    for split, entry in result.items():
        print(f"{split}: psnr_given {entry['psnr_given_mean']:.2f} dB, psnr_novel {entry['psnr_novel_mean']:.2f} dB "
              f"over {entry['windows']} windows in {entry['sequences']} sequences")


if __name__ == "__main__":
    mode, *arguments = sys.argv[1:]
    if mode == "keycheck":
        keycheck()
    elif mode == "depth":
        bf16 = arguments[0] == "--bf16"
        arguments = arguments[1:] if bf16 else arguments
        depth(arguments[0], arguments[1], arguments[2:], bf16)
    elif mode == "eval":
        bf16 = arguments[0] == "--bf16"
        arguments = arguments[1:] if bf16 else arguments
        evaluate(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4:], bf16)
    elif mode == "compare-depth":
        compare_depth(arguments[0], arguments[1], arguments[2], arguments[3:])
    else:
        summarise(arguments[0], arguments[1])
