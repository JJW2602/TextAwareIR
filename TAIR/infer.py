import argparse
import os

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from terediff.utils.common import instantiate_from_config, text_to_image
from terediff.model import ControlLDM, Diffusion
from terediff.sampler import SpacedSampler
import initialize


MODEL_SIZE = 512
VALID_EXT = (".jpg", ".jpeg", ".png")
REGION_COLORS = [
    (0, 255, 0),
    (255, 64, 64),
    (64, 160, 255),
    (255, 192, 0),
    (255, 64, 255),
    (0, 224, 224),
]
PATCH_COLORS = [
    (255, 64, 64),
    (64, 160, 255),
    (255, 192, 0),
    (0, 224, 224),
    (255, 64, 255),
    (0, 200, 96),
]


def collect_image_paths(input_path):
    if os.path.isfile(input_path):
        if not input_path.lower().endswith(VALID_EXT):
            raise ValueError(f"Unsupported image extension: {input_path}")
        return [input_path]

    if not os.path.isdir(input_path):
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    image_paths = [
        os.path.join(input_path, filename)
        for filename in os.listdir(input_path)
        if filename.lower().endswith(VALID_EXT)
    ]
    return sorted(image_paths)


def draw_text_regions(image, ts_result):
    vis_img = image.copy()
    draw = ImageDraw.Draw(vis_img)
    pred_polys = ts_result.get("pred_polys", [])
    pred_texts = ts_result.get("pred_texts", [])

    for idx, poly in enumerate(pred_polys):
        pts = np.asarray(poly, dtype=np.int32).reshape(-1, 2)
        if len(pts) < 2:
            continue
        points = [(int(x), int(y)) for x, y in pts]
        color = REGION_COLORS[idx % len(REGION_COLORS)]
        draw.line(points + [points[0]], fill=color, width=3)

        label = pred_texts[idx] if idx < len(pred_texts) else ""
        if label:
            label_x = max(0, min(x for x, _ in points))
            label_y = max(0, min(y for _, y in points) - 14)
            draw.text((label_x, label_y), label, fill=color)

    return vis_img


def format_patch_name(row_idx, col_idx, x, y, width, height):
    return (
        f"row{row_idx:02d}_col{col_idx:02d}"
        f"_x{x:04d}_y{y:04d}_w{width:04d}_h{height:04d}"
    )


def save_prompt_trace(output_dir, image_id, patch_name, patch_box, initial_prompt, ts_results):
    prompt_dir = os.path.join(output_dir, "prompt_traces", image_id)
    os.makedirs(prompt_dir, exist_ok=True)
    prompt_path = os.path.join(prompt_dir, f"{patch_name}.txt")
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(f"patch_box_xywh: {patch_box}\n")
        f.write(f"initial_prompt: {initial_prompt}\n")
        for step_idx, ts_result in enumerate(ts_results, start=1):
            texts = ", ".join(ts_result["pred_texts"])
            f.write(
                f"step={step_idx:03d} "
                f"timestep={int(ts_result['timestep']):04d} "
                f"texts={texts}\n"
            )
            f.write(f"prompt={ts_result['pred_prompt']}\n")


def save_pred_text_image(output_dir, image_id, patch_name, patch_box, initial_prompt, ts_results):
    pred_text_dir = os.path.join(output_dir, "pred_texts", image_id)
    os.makedirs(pred_text_dir, exist_ok=True)

    lines = [
        f"patch_box_xywh: {patch_box}\n",
        "initial input prompt:\n",
    ]
    width = 80
    for i in range(0, len(initial_prompt), width):
        lines.append(initial_prompt[i:i + width] + "\n")
    lines.append("\n")
    for ts_result in ts_results:
        timestep = ts_result["timestep"]
        pred_texts = ", ".join(ts_result["pred_texts"])
        lines.append(f"timestep: {timestep:<4} /  pred_texts: {pred_texts}\n")

    text_to_image(lines).save(os.path.join(pred_text_dir, f"{patch_name}.png"))


def save_text_region_overlays(output_dir, image_id, patch_name, patch_img, ts_results):
    region_dir = os.path.join(output_dir, "text_regions", image_id, patch_name)
    os.makedirs(region_dir, exist_ok=True)
    for step_idx, ts_result in enumerate(ts_results, start=1):
        timestep = int(ts_result["timestep"])
        vis_img = draw_text_regions(patch_img, ts_result)
        vis_img.save(
            os.path.join(region_dir, f"step_{step_idx:03d}_t{timestep:04d}.png")
        )


def draw_patch_boundaries(image, patch_records):
    vis_img = image.copy()
    draw = ImageDraw.Draw(vis_img)
    for patch_idx, patch_record in enumerate(patch_records):
        color = patch_record["color"]
        x, y, width, height = patch_record["box"]
        x1 = x + width - 1
        y1 = y + height - 1
        draw.rectangle((x, y, x1, y1), outline=color, width=4)
        draw.text((x + 6, y + 6), patch_record["name"], fill=color)
    return vis_img


def draw_text_regions_on_full_image(image, patch_records):
    vis_img = image.copy()
    draw = ImageDraw.Draw(vis_img)

    for patch_record in patch_records:
        x_offset, y_offset, _, _ = patch_record["box"]
        patch_color = patch_record["color"]
        for text_region in patch_record["text_regions"]:
            pts = np.asarray(text_region["polygon"], dtype=np.int32).reshape(-1, 2)
            if len(pts) < 2:
                continue
            pts[:, 0] += x_offset
            pts[:, 1] += y_offset
            points = [(int(x), int(y)) for x, y in pts]
            draw.line(points + [points[0]], fill=patch_color, width=3)

            label = text_region["text"]
            if label:
                label_x = max(0, min(x for x, _ in points))
                label_y = max(0, min(y for _, y in points) - 14)
                draw.text((label_x, label_y), label, fill=patch_color)

    return vis_img


def collect_final_text_regions(ts_results):
    if not ts_results:
        return []

    final_result = ts_results[-1]
    pred_polys = final_result.get("pred_polys", [])
    pred_texts = final_result.get("pred_texts", [])
    regions = []
    for idx, poly in enumerate(pred_polys):
        regions.append(
            {
                "polygon": np.asarray(poly, dtype=np.int32),
                "text": pred_texts[idx] if idx < len(pred_texts) else "",
            }
        )
    return regions


def compute_patch_starts(length, patch_size, stride):
    if length <= patch_size:
        return [0]

    starts = list(range(0, length - patch_size + 1, stride))
    last_start = length - patch_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def build_patch_boxes(width, height, patch_size, overlap):
    stride = patch_size - overlap
    if stride <= 0:
        raise ValueError("patch.overlap must be smaller than patch.size")

    xs = compute_patch_starts(width, patch_size, stride)
    ys = compute_patch_starts(height, patch_size, stride)
    boxes = []
    for row_idx, y in enumerate(ys):
        for col_idx, x in enumerate(xs):
            boxes.append((row_idx, col_idx, x, y))
    return boxes


def extract_patch(image, x, y, patch_size):
    width, height = image.size
    crop = image.crop((x, y, min(x + patch_size, width), min(y + patch_size, height)))
    actual_width, actual_height = crop.size
    if crop.size == (patch_size, patch_size):
        return crop, actual_width, actual_height

    crop_np = np.asarray(crop)
    pad_height = patch_size - actual_height
    pad_width = patch_size - actual_width
    padded_np = np.pad(
        crop_np,
        ((0, pad_height), (0, pad_width), (0, 0)),
        mode="edge",
    )
    return Image.fromarray(padded_np), actual_width, actual_height


def make_blend_mask(patch_size, overlap):
    blend_1d = np.ones(patch_size, dtype=np.float32)
    if overlap > 0:
        ramp = np.linspace(0, 1, overlap + 2, dtype=np.float32)[1:-1]
        ramp = 0.5 - 0.5 * np.cos(np.pi * ramp)
        blend_1d[:overlap] = ramp
        blend_1d[-overlap:] = ramp[::-1]
    return np.outer(blend_1d, blend_1d)[..., None]


def infer_patch(
    patch_img,
    models,
    pure_cldm,
    sampler,
    cfg,
    infer_cfg,
    accelerator,
    device,
    gen,
):
    patch_tensor = TF.to_tensor(patch_img).unsqueeze(0).to(device)
    with torch.no_grad():
        clean = models["swinir"](patch_tensor)
        cond = pure_cldm.prepare_condition(clean, [infer_cfg.prompt.initial])
        uncond = pure_cldm.prepare_condition(clean, [infer_cfg.prompt.negative])
        noise = torch.randn(
            (1, 4, MODEL_SIZE // 8, MODEL_SIZE // 8),
            generator=gen,
            device=device,
            dtype=torch.float32,
        )

        models["testr"].test_score_threshold = infer_cfg.tsm.score_threshold
        z, ts_results = sampler.val_sample(
            model=models["cldm"],
            device=device,
            steps=infer_cfg.sampling.steps,
            x_size=(1, 4, MODEL_SIZE // 8, MODEL_SIZE // 8),
            cond=cond,
            uncond=uncond,
            cfg_scale=infer_cfg.sampling.cfg_scale,
            x_T=noise,
            progress=accelerator.is_main_process,
            cfg=cfg,
            pure_cldm=pure_cldm,
            ts_model=models["testr"],
            val_prompt=[infer_cfg.prompt.initial],
            update_prompt=infer_cfg.tsm.update_prompt,
        )
        restored = torch.clamp((pure_cldm.vae_decode(z) + 1) / 2, 0, 1)

    return restored.squeeze(0).cpu(), clean.squeeze(0).cpu(), ts_results


def save_patch_artifacts(
    infer_cfg,
    image_id,
    patch_name,
    patch_box,
    patch_img,
    ts_results,
):
    if infer_cfg.visualization.save_prompt_traces:
        save_prompt_trace(
            infer_cfg.io.output_dir,
            image_id,
            patch_name,
            patch_box,
            infer_cfg.prompt.initial,
            ts_results,
        )
    if infer_cfg.visualization.save_pred_text_images:
        save_pred_text_image(
            infer_cfg.io.output_dir,
            image_id,
            patch_name,
            patch_box,
            infer_cfg.prompt.initial,
            ts_results,
        )
    if infer_cfg.visualization.save_patch_text_regions:
        save_text_region_overlays(
            infer_cfg.io.output_dir,
            image_id,
            patch_name,
            patch_img,
            ts_results,
        )


def override_model_config(cfg, infer_cfg, args):
    cfg.exp_args.mode = "VAL"
    cfg.exp_args.prompt_style = infer_cfg.prompt.style
    cfg.log_args.log_tool = None

    if infer_cfg.weights.sd_path:
        cfg.train.sd_path = infer_cfg.weights.sd_path
    if infer_cfg.weights.swinir_path:
        cfg.train.swinir_path = infer_cfg.weights.swinir_path
    if infer_cfg.weights.diffbir_ckpt:
        cfg.train.resume = infer_cfg.weights.diffbir_ckpt
    if infer_cfg.weights.terediff_ckpt:
        cfg.exp_args.resume_ckpt_dir = infer_cfg.weights.terediff_ckpt
    if infer_cfg.weights.testr_ckpt:
        cfg.exp_args.testr_ckpt_dir = infer_cfg.weights.testr_ckpt


def load_infer_config(args):
    infer_cfg = OmegaConf.load(args.infer_config)
    cli_overrides = OmegaConf.from_dotlist(args.override)
    infer_cfg = OmegaConf.merge(infer_cfg, cli_overrides)

    if args.input is not None:
        infer_cfg.io.input = args.input
    if args.output_dir is not None:
        infer_cfg.io.output_dir = args.output_dir
    if args.config_testr is not None:
        infer_cfg.model.config_testr = args.config_testr

    OmegaConf.resolve(infer_cfg)
    return infer_cfg


def main(args):
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(split_batches=False, kwargs_handlers=[kwargs])
    infer_cfg = load_infer_config(args)
    set_seed(infer_cfg.runtime.seed, device_specific=False)
    device = accelerator.device
    gen = torch.Generator(device)
    gen.manual_seed(infer_cfg.runtime.seed)

    cfg = OmegaConf.load(args.config)
    override_model_config(cfg, infer_cfg, args)
    args.config_testr = infer_cfg.model.config_testr

    image_paths = collect_image_paths(infer_cfg.io.input)
    if infer_cfg.io.max_images is not None:
        image_paths = image_paths[: infer_cfg.io.max_images]
    if not image_paths:
        raise ValueError(f"No supported images found in: {infer_cfg.io.input}")

    models, _ = initialize.load_model(accelerator, device, args, cfg)
    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)
    diffusion.to(device)
    sampler = SpacedSampler(
        diffusion.betas,
        diffusion.parameterization,
        rescale_cfg=infer_cfg.sampling.rescale_cfg,
    )

    models = {name: accelerator.prepare(model) for name, model in models.items()}
    pure_cldm: ControlLDM = accelerator.unwrap_model(models["cldm"])

    for model in models.values():
        if isinstance(model, nn.Module):
            model.eval()

    if accelerator.is_main_process:
        os.makedirs(infer_cfg.io.output_dir, exist_ok=True)

    for image_path in tqdm(
        image_paths,
        desc="infer",
        disable=not accelerator.is_main_process,
    ):
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        lq_img = Image.open(image_path).convert("RGB")
        original_size = lq_img.size
        width, height = original_size

        if infer_cfg.patch.enabled:
            patch_size = infer_cfg.patch.size
            overlap = infer_cfg.patch.overlap
            if patch_size != MODEL_SIZE:
                raise ValueError(
                    f"patch.size must be {MODEL_SIZE} because TESTR assumes 512x512 patches"
                )

            patch_boxes = build_patch_boxes(width, height, patch_size, overlap)
            blend_mask = make_blend_mask(patch_size, overlap)
            restored_sum = np.zeros((height, width, 3), dtype=np.float32)
            cleaned_sum = (
                np.zeros((height, width, 3), dtype=np.float32)
                if infer_cfg.io.save_cleaned
                else None
            )
            weight_sum = np.zeros((height, width, 1), dtype=np.float32)
            patch_records = []

            for patch_idx, (row_idx, col_idx, x, y) in enumerate(patch_boxes):
                patch_img, actual_width, actual_height = extract_patch(
                    lq_img,
                    x,
                    y,
                    patch_size,
                )
                restored_patch, clean_patch, ts_results = infer_patch(
                    patch_img,
                    models,
                    pure_cldm,
                    sampler,
                    cfg,
                    infer_cfg,
                    accelerator,
                    device,
                    gen,
                )
                patch_name = format_patch_name(
                    row_idx,
                    col_idx,
                    x,
                    y,
                    actual_width,
                    actual_height,
                )
                patch_records.append(
                    {
                        "name": patch_name,
                        "box": (x, y, actual_width, actual_height),
                        "color": PATCH_COLORS[patch_idx % len(PATCH_COLORS)],
                        "text_regions": collect_final_text_regions(ts_results),
                    }
                )

                restored_np = restored_patch.permute(1, 2, 0).numpy()
                clean_np = clean_patch.permute(1, 2, 0).numpy()
                valid_mask = blend_mask[:actual_height, :actual_width]
                y_slice = slice(y, y + actual_height)
                x_slice = slice(x, x + actual_width)

                restored_sum[y_slice, x_slice] += (
                    restored_np[:actual_height, :actual_width] * valid_mask
                )
                if cleaned_sum is not None:
                    cleaned_sum[y_slice, x_slice] += (
                        clean_np[:actual_height, :actual_width] * valid_mask
                    )
                weight_sum[y_slice, x_slice] += valid_mask

                if accelerator.is_main_process:
                    save_patch_artifacts(
                        infer_cfg,
                        image_id,
                        patch_name,
                        (x, y, actual_width, actual_height),
                        patch_img,
                        ts_results,
                    )

            if not accelerator.is_main_process:
                continue

            restored_np = np.clip(restored_sum / np.maximum(weight_sum, 1e-8), 0, 1)
            restored_pil = Image.fromarray((restored_np * 255).round().astype(np.uint8))
            restored_pil.save(
                os.path.join(infer_cfg.io.output_dir, f"restored_{image_id}.png")
            )

            if cleaned_sum is not None:
                cleaned_np = np.clip(cleaned_sum / np.maximum(weight_sum, 1e-8), 0, 1)
                Image.fromarray((cleaned_np * 255).round().astype(np.uint8)).save(
                    os.path.join(infer_cfg.io.output_dir, f"cleaned_{image_id}.png")
                )
            if infer_cfg.visualization.visualize_patches:
                draw_patch_boundaries(lq_img, patch_records).save(
                    os.path.join(
                        infer_cfg.io.output_dir,
                        f"patch_boundaries_{image_id}.png",
                    )
                )
            if infer_cfg.visualization.visualize_text_regions:
                draw_text_regions_on_full_image(lq_img, patch_records).save(
                    os.path.join(
                        infer_cfg.io.output_dir,
                        f"text_regions_{image_id}.png",
                    )
                )
            continue

        resized_input = lq_img.resize((MODEL_SIZE, MODEL_SIZE), Image.BICUBIC)
        restored_patch, clean_patch, ts_results = infer_patch(
            resized_input,
            models,
            pure_cldm,
            sampler,
            cfg,
            infer_cfg,
            accelerator,
            device,
            gen,
        )

        if not accelerator.is_main_process:
            continue

        restored_pil = TF.to_pil_image(restored_patch)
        if infer_cfg.io.output_size == "original":
            restored_pil = restored_pil.resize(original_size, Image.BICUBIC)
        restored_pil.save(
            os.path.join(infer_cfg.io.output_dir, f"restored_{image_id}.png")
        )

        if infer_cfg.io.save_cleaned:
            TF.to_pil_image(clean_patch).save(
                os.path.join(infer_cfg.io.output_dir, f"cleaned_{image_id}.png")
            )
        save_patch_artifacts(
            infer_cfg,
            image_id,
            format_patch_name(0, 0, 0, 0, MODEL_SIZE, MODEL_SIZE),
            (0, 0, MODEL_SIZE, MODEL_SIZE),
            resized_input,
            ts_results,
        )


def build_parser():
    parser = argparse.ArgumentParser(description="Run TeReDiff inference on LQ images.")
    parser.add_argument("--config", default="configs/val/val_terediff.yaml")
    parser.add_argument("--infer-config", default="configs/infer/infer_terediff.yaml")
    parser.add_argument(
        "--config_testr",
        default=None,
    )
    parser.add_argument("--input", help="Override io.input from the inference config.")
    parser.add_argument(
        "--output-dir",
        help="Override io.output_dir from the inference config.",
    )
    parser.add_argument(
        "override",
        nargs="*",
        help="OmegaConf dotlist overrides, for example sampling.steps=25.",
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
