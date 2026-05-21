# TAIRL DDPO/GRPO-LoRA for DiffBIR

This directory configures a first-pass RL fine-tuning loop for DiffBIR v2.1:

1. Load DiffBIR SD v2.1-zsnr, DiffBIR v2.1 ControlNet, and RealESRGAN SwinIR.
2. Freeze the pretrained model and insert LoRA adapters into DiffBIR ControlNet.
3. Sample restored images with a spaced DDPM sampler while recording reverse-transition log probabilities.
4. Run Bridge text spotting on sampled images.
5. Compute the same per-image reward family used by `Slurm/Baselines/DiffBIR/helpers/compute_image_reward.py`.
6. Update only LoRA parameters with a PPO-style DDPO or GRPO objective.

## Training Data

The default configs train from the first 10,000 samples of:

```text
/scratch2/james2602/TextAwareIR/Dataset/SA-Text
```

The fixed subset is recorded under:

```text
/scratch2/james2602/TextAwareIR/TAIRL/train_data/sa_text_train_10000.jsonl
/scratch2/james2602/TextAwareIR/TAIRL/train_data/sa_text_train_10000_ids.txt
/scratch2/james2602/TextAwareIR/TAIRL/train_data/sa_text_train_10000_summary.json
```

`jsonl` stores `sample_index`, `id`, parquet shard path, and row index. `ids.txt` is the compact identifier list. Regenerate the files with `TAIRL/tools/make_sa_text_manifest.py` if the subset definition changes.

## Reward

For one generated image:

```text
matched_mean_reward = mean over IoU-matched text pairs of max(1 - levenshtein(pred, gt) / len(gt), 0)
missed = num_gt - num_matched
false_positive = num_pred - num_matched
final_reward = matched_mean_reward - miss_penalty * missed - false_positive_penalty * false_positive
final_reward_norm = matched_mean_reward
                    - miss_penalty * missed / max(num_gt, 1)
                    - false_positive_penalty * false_positive / max(num_gt, 1)
```

The default search uses `final_reward_norm` because it is less dominated by images with many text boxes. Default penalties are `miss_penalty=1.0` and `false_positive_penalty=0.25`: missed GT text is weighted more strongly, while unmatched detections are penalized more softly to discourage over-detection without suppressing valid text recovery too early.

## Hyperparameter Search

Run:

```bash
bash /scratch2/james2602/TextAwareIR/TAIRL/run_hparam_search.sh
```

The Slurm array runs 11 short trials. Each trial uses the 10,000-sample SA-Text manifest, batch size 1, 8 denoise steps, and 50 RL updates. This is intentionally short in update count: the goal is to see whether reward and policy statistics move in the right direction before spending a full run.

The main search axes are:

- `learning_rate`: `1e-5`, `3e-5`, `1e-4`. RL gradients are noisier than supervised DiffBIR training, so this brackets conservative, middle, and aggressive LoRA updates.
- `lora.rank`: `4`, `8`, `16`. Rank 4 checks whether a tiny adapter can move reward; rank 8 is the default capacity; rank 16 tests whether reward needs more capacity.
- `ddpo.clip_range`: `0.1`, `0.2`. The sampler uses mean log-probability per transition, so standard PPO-style clipping is appropriate; `0.1` is safer, `0.2` allows faster movement.
- `ddpo.kl_coef`: `0.02`, `0.0`. The KL penalty limits how far LoRA moves the policy away from pretrained DiffBIR. The zero-KL trial checks whether this reference regularization is over-constraining the first pass.
- `reward.variant` / penalties: mostly normalized reward, plus one raw reward trial. `miss_penalty` candidates are `1.0` and `0.5`; `false_positive_penalty` candidates are `0.0`, `0.25`, and `0.5`.

Outputs are written under:

```text
/scratch2/james2602/TextAwareIR/Results/TAIRL/ddpo_lora_hparam/<trial_name>/
```

Check `metrics.jsonl` for reward mean, pretrained-reference KL, PPO approximate KL, ratio mean, clip fraction, gradient norm, and per-image reward details. LoRA checkpoints are saved in `checkpoints/`.

## Weights & Biases

W&B logging is enabled by default in the provided configs:

```yaml
wandb.enabled: true
wandb.project: TAIRL
wandb.entity: null
wandb.mode: online
```

`wandb.entity: null` means W&B uses the default entity for the account/API key that is logged in on the machine. Set `wandb.entity=<team_or_user>` if the run must go to a specific team.

Before submitting jobs, log in once in the DiffBIR environment:

```bash
/home/james2602/miniconda3/envs/diffbir/bin/python -m wandb login
```

For cluster runs without network access, use offline mode:

```bash
WANDB_MODE=offline bash /scratch2/james2602/TextAwareIR/TAIRL/run_grpo_g6.sh
```

The run logs the essential RL health signals:

- reward: mean/std/min/max, matched text score, final reward, normalized reward, GT match rate, missed-GT rate, false-positive-per-GT rate
- PPO/GRPO: clipped policy loss, pretrained-reference KL, old/new approximate KL diagnostic, clip fraction, ratio mean, advantage stats
- LoRA/update: grad norm, LoRA parameter L2 norm, learning rate
- speed/VRAM: rollout/reward/update/step seconds, images per update, CUDA allocated/reserved memory
- samples: generated images and per-image reward tables every `wandb.log_images_every` / `wandb.log_tables_every` steps

## GRPO G=6

Run:

```bash
bash /scratch2/james2602/TextAwareIR/TAIRL/run_grpo_g6.sh
```

GRPO mode sets `train.algorithm=grpo` and `grpo.group_size=6`. Each optimizer step uses one source image, samples 6 different denoising trajectories from different noise, scores the 6 restored images with the same Bridge reward, and computes group-relative advantages:

```text
adv_i = (reward_i - mean(reward_group)) / (std(reward_group) + eps)
```

The default `grpo.generation_microbatch=1` generates those 6 images sequentially to reduce VRAM. Increasing it to 2, 3, or 6 makes generation more parallel but raises VRAM.

## KL Term

Both DDPO and GRPO configs include a KL coefficient:

```yaml
ddpo.kl_coef: 0.02
grpo.kl_coef: 0.02
```

The loss uses pretrained-reference KL. During the update, TAIRL computes the current transition Gaussian with LoRA enabled, then temporarily disables all LoRA adapters to compute the pretrained DiffBIR reference transition Gaussian. The KL term is the analytic Gaussian KL between those reverse denoising transitions:

```text
loss = clipped_policy_loss + kl_coef * KL(policy_lora || policy_pretrained)
```

`approx_kl` is still logged, but it is only the PPO old/new policy movement diagnostic used alongside clipping. It is not the configured KL penalty.
