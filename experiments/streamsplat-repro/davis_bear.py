"""DAVIS helpers, run inside the StreamSplat image: davis_eval.py (G8, G12, G16) imports depth; psnr and checks served G4.

  depth <frames-dir> <depth-dir> : Depth Anything V2 depth for every frame, made exactly as the authors'
                                   preprocess_depth_davis.py makes it (their Predictor and resize functions,
                                   min-max to uint8), saved as <stem>_pred.png. The Predictor's two steps are
                                   called one by one so that depth(..., bf16=True) (G16, davis_eval.py depth
                                   --bf16) can run the depth model alone under bf16 autocast and interpolate in
                                   fp32, as bench/capacity.py's --depth-bf16 server does; with bf16 off the
                                   computation is the Predictor's.
  psnr <output-dir> <gap> <json> : PSNR of each rendered keyframe against its input keyframe, read from the two
                                   videos splat_inference.py writes (keyframe i is render frame i * gap).
  checks <output-dir> <gap> <frames-dir> : bug checks on psnr, added after G4 missed its gate: the mean PSNR at
                                   render offsets -1, 0, +1 (alignment is right if 0 is highest), and the PSNR of the
                                   input video against the JPEG keyframes it was made from (input-side codec loss).
"""
import json, sys
from pathlib import Path


def depth(frames_dir, depth_dir, bf16=False):
    import cv2, numpy as np, torch, torch.nn.functional as functional, torchvision.transforms as transforms
    from PIL import Image
    from configs.options import Options
    from preprocess_depth_davis import Predictor, resize_to_multiple_of_14, resize_to_shorter_side
    model = Predictor(Options()).to("cuda").eval()
    to_tensor = transforms.ToTensor()
    Path(depth_dir).mkdir(parents=True, exist_ok=True)
    frames = sorted(Path(frames_dir).glob("*.jpg"))
    with torch.no_grad():
        for frame_path in frames:  # as DAVISDataset.__getitem__ and run_inference do it, one frame per batch
            image = Image.open(frame_path).convert("RGB")
            frame = to_tensor(image).unsqueeze(0).cuda()
            depth_input = to_tensor(resize_to_multiple_of_14(resize_to_shorter_side(image))).unsqueeze(0).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                disparity = model.depth_prior(depth_input).float()  # [1, h, w], what Predictor.forward computes first
            predicted = functional.interpolate(disparity[:, None], size=frame.shape[-2:], mode="bilinear",
                                               align_corners=True)[0, 0].cpu().numpy()  # then this, at frame size
            normalised = cv2.normalize(predicted, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            cv2.imwrite(str(Path(depth_dir) / f"{frame_path.stem}_pred.png"), normalised)
    print(f"depth maps: {len(frames)}")


def psnr(output_dir, gap, metrics_path):
    import imageio.v2 as iio, numpy as np
    rendered = [frame.astype(np.float64) / 255 for frame in iio.get_reader(str(Path(output_dir) / "render_video_0.mp4"))]
    inputs = [frame.astype(np.float64) / 255 for frame in iio.get_reader(str(Path(output_dir) / "input_video_0.mp4"))]
    keyframes = min(len(inputs), (len(rendered) - 1) // gap + 1)
    values = [float(10 * np.log10(1 / np.mean((rendered[index * gap] - inputs[index]) ** 2))) for index in range(keyframes)]
    result = {"psnr_mean": float(np.mean(values)), "psnr_min": min(values), "psnr_max": max(values), "keyframes": keyframes,
              "rendered_frames": len(rendered), "input_frames": len(inputs), "psnr": values}
    Path(metrics_path).write_text(json.dumps(result, indent=1))
    print(f"keyframe PSNR mean {result['psnr_mean']:.2f} dB over {keyframes} keyframes "
          f"(min {result['psnr_min']:.2f}, max {result['psnr_max']:.2f})")


def checks(output_dir, gap, frames_dir):
    import cv2, imageio.v2 as iio, numpy as np
    rendered = [frame.astype(np.float64) / 255 for frame in iio.get_reader(str(Path(output_dir) / "render_video_0.mp4"))]
    inputs = [frame.astype(np.float64) / 255 for frame in iio.get_reader(str(Path(output_dir) / "input_video_0.mp4"))]
    psnr = lambda a, b: float(10 * np.log10(1 / np.mean((a - b) ** 2)))
    height, width = inputs[0].shape[:2]
    for offset in (-1, 0, 1):
        pairs = [(index * gap + offset, index) for index in range(len(inputs)) if 0 <= index * gap + offset < len(rendered)]
        print(f"offset {offset:+d}: mean PSNR {np.mean([psnr(rendered[r], inputs[i]) for r, i in pairs]):.2f} dB over {len(pairs)}")
    keyframes = sorted(Path(frames_dir).glob("*.jpg"))[::gap][:len(inputs)]  # as splat_inference.py selects and resizes
    jpegs = [cv2.resize(cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB), (width, height),
                        interpolation=cv2.INTER_LINEAR).astype(np.float64) / 255 for path in keyframes]
    print(f"input video vs JPEG keyframes: mean PSNR {np.mean([psnr(a, b) for a, b in zip(inputs, jpegs)]):.2f} dB over {len(jpegs)}")


if __name__ == "__main__":
    mode, *arguments = sys.argv[1:]
    if mode == "depth":
        depth(*arguments)
    elif mode == "checks":
        checks(arguments[0], int(arguments[1]), arguments[2])
    else:
        psnr(arguments[0], int(arguments[1]), arguments[2])
