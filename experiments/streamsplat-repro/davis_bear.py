"""G4 helper, run inside the StreamSplat image by run_G4_davis-bear.sh.

  depth <frames-dir> <depth-dir> : Depth Anything V2 depth for every frame, made exactly as the authors'
                                   preprocess_depth_davis.py makes it (their Predictor and resize functions,
                                   min-max to uint8), saved as <stem>_pred.png.
  psnr <output-dir> <gap> <json> : PSNR of each rendered keyframe against its input keyframe, read from the two
                                   videos splat_inference.py writes (keyframe i is render frame i * gap).
"""
import json, sys
from pathlib import Path


def depth(frames_dir, depth_dir):
    import cv2, numpy as np, torch, torchvision.transforms as transforms
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
            predicted = model(frame, depth_input)[0, 0].cpu().numpy()
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


if __name__ == "__main__":
    mode, *arguments = sys.argv[1:]
    depth(*arguments) if mode == "depth" else psnr(arguments[0], int(arguments[1]), arguments[2])
