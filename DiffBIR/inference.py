from types import SimpleNamespace
from typing import Any

import hydra
import torch
from accelerate.utils import set_seed
from omegaconf import DictConfig

from diffbir.inference import (
    BFRInferenceLoop,
    BIDInferenceLoop,
    BSRInferenceLoop,
    CustomInferenceLoop,
    UnAlignedBFRInferenceLoop,
)


VALID_TASKS = {"sr", "face", "denoise", "unaligned_face"}
VALID_VERSIONS = {"v1", "v2", "v2.1", "custom"}
VALID_SAMPLERS = {
    "dpm++_m2",
    "spaced",
    "ddim",
    "edm_euler",
    "edm_euler_a",
    "edm_heun",
    "edm_dpm_2",
    "edm_dpm_2_a",
    "edm_lms",
    "edm_dpm++_2s_a",
    "edm_dpm++_sde",
    "edm_dpm++_2m",
    "edm_dpm++_2m_sde",
    "edm_dpm++_3m_sde",
}
VALID_START_POINT_TYPES = {"noise", "cond"}
VALID_CAPTIONERS = {"none", "llava", "ram"}
VALID_G_LOSSES = {"mse", "w_mse"}
VALID_DEVICES = {"cpu", "cuda", "mps"}
VALID_PRECISIONS = {"fp32", "fp16", "bf16"}
VALID_LLAVA_BITS = {"16", "8", "4"}


def check_device(device: str) -> str:
    if device == "cuda":
        if not torch.cuda.is_available():
            print(
                "CUDA not available because the current PyTorch install was not "
                "built with CUDA enabled."
            )
            device = "cpu"
    elif device == "mps":
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print(
                    "MPS not available because the current PyTorch install was not "
                    "built with MPS enabled."
                )
                device = "cpu"
            else:
                print(
                    "MPS not available because the current MacOS version is not 12.3+ "
                    "and/or you do not have an MPS-enabled device on this machine."
                )
                device = "cpu"
    print(f"using device {device}")
    return device


def require_choice(name: str, value: Any, valid_values: set[str]) -> None:
    if value not in valid_values:
        options = ", ".join(sorted(valid_values))
        raise ValueError(f"Unsupported {name}: {value!r}. Expected one of: {options}.")


def build_args(cfg: DictConfig) -> SimpleNamespace:
    require_choice("task", cfg.task, VALID_TASKS)
    require_choice("model.version", cfg.model.version, VALID_VERSIONS)
    require_choice("sampling.sampler", cfg.sampling.sampler, VALID_SAMPLERS)
    require_choice(
        "sampling.start_point_type",
        cfg.sampling.start_point_type,
        VALID_START_POINT_TYPES,
    )
    require_choice("caption.name", cfg.caption.name, VALID_CAPTIONERS)
    require_choice("caption.llava_bit", str(cfg.caption.llava_bit), VALID_LLAVA_BITS)
    require_choice("guidance.loss", cfg.guidance.loss, VALID_G_LOSSES)
    require_choice("runtime.device", cfg.runtime.device, VALID_DEVICES)
    require_choice("runtime.precision", cfg.runtime.precision, VALID_PRECISIONS)

    if cfg.model.version == "custom":
        if not cfg.model.train_cfg:
            raise ValueError("model.train_cfg is required when model.version=custom.")
        if not cfg.model.ckpt:
            raise ValueError("model.ckpt is required when model.version=custom.")

    return SimpleNamespace(
        task=cfg.task,
        upscale=cfg.upscale,
        version=cfg.model.version,
        train_cfg=cfg.model.train_cfg,
        ckpt=cfg.model.ckpt,
        sampler=cfg.sampling.sampler,
        steps=cfg.sampling.steps,
        start_point_type=cfg.sampling.start_point_type,
        cleaner_tiled=cfg.tiling.cleaner.enabled,
        cleaner_tile_size=cfg.tiling.cleaner.tile_size,
        cleaner_tile_stride=cfg.tiling.cleaner.tile_stride,
        vae_encoder_tiled=cfg.tiling.vae_encoder.enabled,
        vae_encoder_tile_size=cfg.tiling.vae_encoder.tile_size,
        vae_decoder_tiled=cfg.tiling.vae_decoder.enabled,
        vae_decoder_tile_size=cfg.tiling.vae_decoder.tile_size,
        cldm_tiled=cfg.tiling.cldm.enabled,
        cldm_tile_size=cfg.tiling.cldm.tile_size,
        cldm_tile_stride=cfg.tiling.cldm.tile_stride,
        captioner=cfg.caption.name,
        pos_prompt=cfg.caption.pos_prompt,
        neg_prompt=cfg.caption.neg_prompt,
        cfg_scale=cfg.sampling.cfg_scale,
        rescale_cfg=cfg.sampling.rescale_cfg,
        noise_aug=cfg.sampling.noise_aug,
        s_churn=cfg.sampling.s_churn,
        s_tmin=cfg.sampling.s_tmin,
        s_tmax=cfg.sampling.s_tmax,
        s_noise=cfg.sampling.s_noise,
        eta=cfg.sampling.eta,
        order=cfg.sampling.order,
        strength=cfg.sampling.strength,
        batch_size=cfg.runtime.batch_size,
        guidance=cfg.guidance.enabled,
        g_loss=cfg.guidance.loss,
        g_scale=cfg.guidance.scale,
        input=cfg.io.input,
        n_samples=cfg.io.n_samples,
        output=cfg.io.output,
        seed=cfg.runtime.seed,
        device=cfg.runtime.device,
        precision=cfg.runtime.precision,
        llava_bit=str(cfg.caption.llava_bit),
    )


@hydra.main(version_base=None, config_path="configs/inference", config_name="default")
def main(cfg: DictConfig) -> None:
    args = build_args(cfg)
    args.device = check_device(args.device)
    set_seed(args.seed)

    if args.version != "custom":
        loops = {
            "sr": BSRInferenceLoop,
            "denoise": BIDInferenceLoop,
            "face": BFRInferenceLoop,
            "unaligned_face": UnAlignedBFRInferenceLoop,
        }
        loops[args.task](args).run()
    else:
        CustomInferenceLoop(args).run()
    print("done!")


if __name__ == "__main__":
    main()
