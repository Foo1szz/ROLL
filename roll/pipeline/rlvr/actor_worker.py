import numpy as np
import torch
from typing import Optional

from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.base_worker import ActorWorker as BaseActorWorker
from roll.utils.functionals import masked_mean, agg_loss, compute_approx_kl


class ActorWorker(BaseActorWorker):

    def _get_partition_metrics(
        self,
        data: DataProto,
        response_mask: torch.Tensor,
        final_response_mask: torch.Tensor,
        base_final_response_mask: torch.Tensor,
        sample_weights: torch.Tensor,
        valid_samples: torch.Tensor,
        ratio: torch.Tensor,
        surr1: torch.Tensor,
        surr2: torch.Tensor,
        pg_loss_mat: torch.Tensor,
        kl_loss_mat: torch.Tensor,
        approxkl: torch.Tensor,
        policykl: torch.Tensor,
        train_infer_ratio: torch.Tensor,
        train_infer_diff: torch.Tensor,
        train_infer_ratio_mask: Optional[torch.Tensor],
        train_infer_diff_mask: Optional[torch.Tensor],
        train_infer_ratio_seq_mask: Optional[torch.Tensor],
        train_infer_diff_seq_mask: Optional[torch.Tensor],
        loss_scale: Optional[float],
    ):
        if "is_noisy" not in data.non_tensor_batch:
            return {}

        is_noisy = np.asarray([bool(val) for val in data.non_tensor_batch["is_noisy"]], dtype=bool)
        metrics = {}
        partitions = {"clean": ~is_noisy, "noisy": is_noisy}

        for name, mask_np in partitions.items():
            if not mask_np.any():
                continue

            mask = torch.as_tensor(mask_np, dtype=torch.bool, device=response_mask.device)
            sub_response_mask = response_mask[mask]
            sub_final_response_mask = final_response_mask[mask]
            sub_base_final_response_mask = base_final_response_mask[mask]
            sub_weights = sample_weights[mask]
            sub_valid_samples = valid_samples[mask]
            sub_ratio = ratio[mask]
            sub_surr1 = surr1[mask]
            sub_surr2 = surr2[mask]

            sub_pg_loss = agg_loss(
                loss_mat=pg_loss_mat[mask],
                loss_mask=sub_final_response_mask,
                loss_agg_mode=self.pipeline_config.loss_agg_mode,
                loss_scale=loss_scale,
            )
            sub_weighted_pg_loss = agg_loss(
                loss_mat=pg_loss_mat[mask],
                loss_mask=sub_final_response_mask,
                loss_agg_mode=self.pipeline_config.loss_agg_mode,
                weights=sub_weights,
                loss_scale=loss_scale,
            )
            sub_kl_loss = agg_loss(
                loss_mat=kl_loss_mat[mask],
                loss_mask=sub_final_response_mask,
                loss_agg_mode=self.pipeline_config.loss_agg_mode,
                loss_scale=loss_scale,
            )
            if self.pipeline_config.use_kl_loss:
                sub_total_loss = sub_weighted_pg_loss + sub_kl_loss * self.pipeline_config.kl_loss_coef
            else:
                sub_total_loss = sub_weighted_pg_loss
            sub_total_loss = sub_total_loss * self.pipeline_config.rl_loss_coef

            metrics.update(
                {
                    f"{name}/actor/ppo_ratio_high_clipfrac": sub_ratio.gt(
                        1 + (
                            self.pipeline_config.pg_clip_high
                            if self.pipeline_config.use_pg_clip_range
                            else self.pipeline_config.pg_clip
                        )
                    )
                    .float()
                    .mean()
                    .item(),
                    f"{name}/actor/ppo_ratio_low_clipfrac": sub_ratio.lt(
                        1 - (
                            self.pipeline_config.pg_clip_low
                            if self.pipeline_config.use_pg_clip_range
                            else self.pipeline_config.pg_clip
                        )
                    )
                    .float()
                    .mean()
                    .item(),
                    f"{name}/actor/ppo_ratio_clipfrac": (
                        sub_ratio.lt(
                            1 - (
                                self.pipeline_config.pg_clip_low
                                if self.pipeline_config.use_pg_clip_range
                                else self.pipeline_config.pg_clip
                            )
                        )
                        | sub_ratio.gt(
                            1 + (
                                self.pipeline_config.pg_clip_high
                                if self.pipeline_config.use_pg_clip_range
                                else self.pipeline_config.pg_clip
                            )
                        )
                    )
                    .float()
                    .mean()
                    .item(),
                    f"{name}/actor/ratio_mean": masked_mean(sub_ratio, sub_response_mask, dim=-1).mean().detach().item(),
                    f"{name}/actor/ratio_max": torch.max(sub_ratio * sub_response_mask).detach().item(),
                    f"{name}/actor/ratio_min": torch.min(
                        sub_ratio * sub_response_mask + (1 - sub_response_mask) * 1e10
                    )
                    .detach()
                    .item(),
                    f"{name}/actor/clipfrac": agg_loss(
                        loss_mat=torch.lt(sub_surr2, sub_surr1).float(),
                        loss_mask=sub_response_mask,
                        loss_agg_mode=self.pipeline_config.loss_agg_mode,
                        loss_scale=loss_scale,
                    )
                    .detach()
                    .item(),
                    f"{name}/actor/pg_loss": sub_pg_loss.detach().item(),
                    f"{name}/actor/weighted_pg_loss": sub_weighted_pg_loss.detach().item(),
                    f"{name}/actor/kl_loss": sub_kl_loss.detach().item(),
                    f"{name}/actor/total_loss": sub_total_loss.detach().item(),
                    f"{name}/actor/approxkl": agg_loss(
                        loss_mat=approxkl[mask],
                        loss_mask=sub_response_mask,
                        loss_agg_mode=self.pipeline_config.loss_agg_mode,
                    )
                    .detach()
                    .item(),
                    f"{name}/actor/policykl": agg_loss(
                        loss_mat=policykl[mask],
                        loss_mask=sub_response_mask,
                        loss_agg_mode=self.pipeline_config.loss_agg_mode,
                    )
                    .detach()
                    .item(),
                    f"{name}/actor/valid_samples": sub_valid_samples.sum().detach().item(),
                    f"{name}/actor/total_samples": float(sub_valid_samples.size(0)),
                    f"{name}/actor/valid_sample_ratio": (
                        sub_valid_samples.sum() / (sub_valid_samples.size(0) + 1e-8)
                    )
                    .detach()
                    .item(),
                    f"{name}/actor/sample_weights_mean": sub_weights.mean().detach().item(),
                    f"{name}/actor/sample_weights_min": sub_weights.min().detach().item(),
                    f"{name}/actor/sample_weights_max": sub_weights.max().detach().item(),
                    f"{name}/actor/train_infer_ratio_mean": masked_mean(
                        train_infer_ratio[mask], sub_response_mask, dim=-1
                    )
                    .mean()
                    .detach()
                    .item(),
                    f"{name}/actor/train_infer_diff_mean": masked_mean(
                        train_infer_diff[mask], sub_response_mask, dim=-1
                    )
                    .mean()
                    .detach()
                    .item(),
                    f"{name}/actor/train_infer_ratio_mask_mean": (
                        masked_mean(
                            train_infer_ratio_mask[mask], sub_base_final_response_mask, dim=-1
                        )
                        .mean()
                        .detach()
                        .item()
                        if train_infer_ratio_mask is not None
                        else 1.0
                    ),
                    f"{name}/actor/train_infer_diff_mask_mean": (
                        masked_mean(
                            train_infer_diff_mask[mask], sub_base_final_response_mask, dim=-1
                        )
                        .mean()
                        .detach()
                        .item()
                        if train_infer_diff_mask is not None
                        else 1.0
                    ),
                    f"{name}/actor/train_infer_ratio_seq_mask_mean": (
                        masked_mean(
                            train_infer_ratio_seq_mask[mask], sub_base_final_response_mask, dim=-1
                        )
                        .mean()
                        .detach()
                        .item()
                        if train_infer_ratio_seq_mask is not None
                        else 1.0
                    ),
                    f"{name}/actor/train_infer_diff_seq_mask_mean": (
                        masked_mean(
                            train_infer_diff_seq_mask[mask], sub_base_final_response_mask, dim=-1
                        )
                        .mean()
                        .detach()
                        .item()
                        if train_infer_diff_seq_mask is not None
                        else 1.0
                    ),
                }
            )

        return metrics

    def loss_func(self, data: DataProto, output_tensor: torch.Tensor):
        """
        loss func接口定义:
            data: DataProto, 由train_step透传
            output_tensor: torch.Tensor, model.forward()的输出Tensor
        """
        response_mask = data.batch["response_mask"][:, 1:].long()
        final_response_mask = data.batch.get("final_response_mask", response_mask)
        ref_log_probs = data.batch["ref_log_probs"]
        advantages = data.batch["advantages"]

        log_probs = self.strategy.op_compute_log_probs(
            logits=output_tensor, input_ids=data.batch["input_ids"], attention_mask=data.batch["response_mask"]
        )
        old_log_probs = self.get_old_log_probs_with_cache(data, log_probs)
        infer_log_probs = data.batch.get("infer_logprobs", old_log_probs)
        infer_log_probs = infer_log_probs if len(infer_log_probs) > 0 else old_log_probs

        loss_scale =None
        if self.worker_config.use_dynamic_batching_in_train and self.pipeline_config.loss_agg_mode == "seq-mean-token-sum":
            micro_batch_indices = data.meta_info["micro_batch_indices"]
            mini_batch_size = micro_batch_indices[-1][-1] - micro_batch_indices[0][0]
            num_micro_batch = len(micro_batch_indices)
            micro_batch_size = data.batch.batch_size[0]
            loss_scale = num_micro_batch * micro_batch_size / mini_batch_size

        valid_samples = torch.any(final_response_mask > 0, dim=1).float()
        sample_weights = self.compute_sample_weights(data, response_mask)


        kl_loss_mat = compute_approx_kl(
            log_probs=log_probs, log_probs_base=ref_log_probs, action_mask=final_response_mask, kl_penalty="k3"
        )
        kl_loss = agg_loss(loss_mat=kl_loss_mat,
                        loss_mask=final_response_mask,
                        loss_agg_mode=self.pipeline_config.loss_agg_mode,
                        loss_scale=loss_scale)

        approxkl = compute_approx_kl(
            log_probs=log_probs, log_probs_base=old_log_probs, action_mask=response_mask, kl_penalty="mse"
        )
        policykl = compute_approx_kl(
            log_probs=log_probs, log_probs_base=old_log_probs, action_mask=response_mask, kl_penalty="kl"
        )

        train_infer_ratio = (old_log_probs - infer_log_probs).exp()
        train_infer_diff = old_log_probs.exp() - infer_log_probs.exp()
        train_infer_ratio_seq = masked_mean(old_log_probs - infer_log_probs, response_mask, dim=-1).exp().unsqueeze(-1).expand_as(train_infer_ratio)
        train_infer_diff_seq = masked_mean(old_log_probs.exp() - infer_log_probs.exp(), response_mask, dim=-1).unsqueeze(-1).expand_as(train_infer_diff)

        train_infer_ratio_mask_mean = 1.0
        train_infer_diff_mask_mean = 1.0
        train_infer_ratio_seq_mask_mean = 1.0
        train_infer_diff_seq_mask_mean = 1.0
        train_infer_ratio_mask = None
        train_infer_diff_mask = None
        train_infer_ratio_seq_mask = None
        train_infer_diff_seq_mask = None

        base_final_response_mask = final_response_mask.clone()

        if self.pipeline_config.train_infer_ratio_mask:
            train_infer_ratio_mask = (train_infer_ratio <= self.pipeline_config.train_infer_ratio_threshold_high).float() * (train_infer_ratio >= self.pipeline_config.train_infer_ratio_threshold_low).float()
            train_infer_ratio_mask_mean = masked_mean(train_infer_ratio_mask, final_response_mask, dim=-1).mean().detach().item()
            final_response_mask = final_response_mask * train_infer_ratio_mask
        if self.pipeline_config.train_infer_diff_mask:
            train_infer_diff_mask = (train_infer_diff <= self.pipeline_config.train_infer_diff_threshold_high).float() * (train_infer_diff >= self.pipeline_config.train_infer_diff_threshold_low).float()
            train_infer_diff_mask_mean = masked_mean(train_infer_diff_mask, final_response_mask, dim=-1).mean().detach().item()
            final_response_mask = final_response_mask * train_infer_diff_mask

        if self.pipeline_config.train_infer_ratio_seq_mask:
            train_infer_ratio_seq_mask = (train_infer_ratio_seq <= self.pipeline_config.train_infer_ratio_seq_threshold_high).float() * (train_infer_ratio_seq >= self.pipeline_config.train_infer_ratio_seq_threshold_low).float()
            train_infer_ratio_seq_mask_mean = masked_mean(train_infer_ratio_seq_mask, final_response_mask, dim=-1).mean().detach().item()
            final_response_mask = final_response_mask * train_infer_ratio_seq_mask
        if self.pipeline_config.train_infer_diff_seq_mask:
            train_infer_diff_seq_mask = (train_infer_diff_seq <= self.pipeline_config.train_infer_diff_seq_threshold_high).float() * (train_infer_diff_seq >= self.pipeline_config.train_infer_diff_seq_threshold_low).float()
            train_infer_diff_seq_mask_mean = masked_mean(train_infer_diff_seq_mask, final_response_mask, dim=-1).mean().detach().item()
            final_response_mask = final_response_mask * train_infer_diff_seq_mask

        if self.pipeline_config.importance_sampling == "token":
            ratio = (log_probs - old_log_probs).exp()
        elif self.pipeline_config.importance_sampling == "seq":
            log_ratio = log_probs - old_log_probs
            masked_log_ratio = masked_mean(log_ratio, final_response_mask, dim=-1)
            ratio = masked_log_ratio.exp().unsqueeze(-1).expand_as(log_ratio)        

        pg_clip_low = self.pipeline_config.pg_clip_low if self.pipeline_config.use_pg_clip_range else self.pipeline_config.pg_clip
        pg_clip_high = self.pipeline_config.pg_clip_high if self.pipeline_config.use_pg_clip_range else self.pipeline_config.pg_clip
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - pg_clip_low, 1 + pg_clip_high) * advantages

        loss = -torch.min(surr1, surr2)

        if self.pipeline_config.dual_clip_loss:
            dual_clip_loss = -torch.max(-loss, (1 + self.pipeline_config.pg_clip * 2) * advantages)
            loss = torch.where(advantages < 0, dual_clip_loss, loss)

        if self.pipeline_config.use_rollout_importance_sampling_ratio:
            rollout_importance_sampling_clip = (train_infer_ratio > self.pipeline_config.rollout_importance_sampling_ratio_upper_bound).float()
            loss = train_infer_ratio.clamp(0, self.pipeline_config.rollout_importance_sampling_ratio_upper_bound) * loss

        weighted_pg_loss = agg_loss(loss_mat=loss, loss_mask=final_response_mask,
                                    loss_agg_mode=self.pipeline_config.loss_agg_mode,
                                    weights=sample_weights, loss_scale=loss_scale)
        original_pg_loss = agg_loss(loss_mat=loss, loss_mask=final_response_mask,
                                    loss_agg_mode=self.pipeline_config.loss_agg_mode,
                                    loss_scale=loss_scale)

        clipped_low = (ratio < 1 - pg_clip_low).float()
        clipped_high = (ratio > 1 + pg_clip_high).float()
        clipped = (clipped_low + clipped_high).float()

        if self.pipeline_config.use_kl_loss:
            total_loss = weighted_pg_loss + kl_loss * self.pipeline_config.kl_loss_coef
        else:
            total_loss = weighted_pg_loss

        total_loss = total_loss * self.pipeline_config.rl_loss_coef

        if self.pipeline_config.entropy_loss_coef > 0:
            entropy = self.strategy.op_compute_entropy(logits=output_tensor, attention_mask=data.batch["response_mask"])
            entropy_loss = agg_loss(
                loss_mat=entropy,
                loss_mask=data.batch["response_mask"][:, 1:],
                loss_agg_mode=self.pipeline_config.loss_agg_mode,
                loss_scale=loss_scale
            )
            total_loss = total_loss - entropy_loss * self.pipeline_config.entropy_loss_coef

        metrics = {}
        if self.pipeline_config.postive_loss_coef > 0:
            response_positive_mask = (data.batch['scores'] > 0).unsqueeze(-1).expand_as(final_response_mask)
            # TODO: 是否应该乘上adv？
            postive_loss = agg_loss(loss_mat=-log_probs * advantages, loss_mask=final_response_mask * response_positive_mask,
                                loss_agg_mode=self.pipeline_config.loss_agg_mode, weights=torch.ones_like(sample_weights),
                                loss_scale=loss_scale)
            total_loss = total_loss + postive_loss * self.pipeline_config.postive_loss_coef
            metrics['actor/postive_loss'] = postive_loss.detach().item()
            
        if self.pipeline_config.use_topr_neg_loss_coef > 0:
            response_negative_mask = (data.batch['scores'] <= 0).unsqueeze(-1).expand_as(final_response_mask)
            clipped_ratio = torch.clamp((log_probs.detach() - old_log_probs).exp(), 0 , 1)
            topr_neg_loss = agg_loss(loss_mat=-clipped_ratio * log_probs * advantages, loss_mask=final_response_mask * response_negative_mask,
                                loss_agg_mode=self.pipeline_config.loss_agg_mode, weights=torch.ones_like(sample_weights),
                                loss_scale=loss_scale)
            total_loss = total_loss + topr_neg_loss * self.pipeline_config.use_topr_neg_loss_coef
            metrics['actor/topr_neg_loss'] = topr_neg_loss.detach().item()

        train_infer_prob_metric = {
            "actor/train_infer_ratio_mean": masked_mean(train_infer_ratio, response_mask, dim=-1).mean().detach().item(),
            "actor/train_infer_diff_mean": masked_mean(train_infer_diff, response_mask, dim=-1).mean().detach().item(),
            "actor/train_infer_ratio_mask_mean": train_infer_ratio_mask_mean,
            "actor/train_infer_diff_mask_mean": train_infer_diff_mask_mean,
            "actor/train_infer_ratio_seq_mask_mean": train_infer_ratio_seq_mask_mean,
            "actor/train_infer_diff_seq_mask_mean": train_infer_diff_seq_mask_mean,
        }

        loss_metric = {
            "actor/ppo_ratio_high_clipfrac": clipped_high.mean().detach().item(),
            "actor/ppo_ratio_low_clipfrac": clipped_low.mean().detach().item(),
            "actor/ppo_ratio_clipfrac": clipped.mean().detach().item(),
            "actor/ratio_mean": masked_mean(ratio, response_mask, dim=-1).mean().detach().item(),
            "actor/ratio_max": torch.max(ratio * response_mask).detach().item(),
            "actor/ratio_min": torch.min(ratio * response_mask + (1 - response_mask) * 1e10).detach().item(),
            "actor/clipfrac": agg_loss(loss_mat=torch.lt(surr2, surr1).float(), loss_mask=response_mask,
                                loss_agg_mode=self.pipeline_config.loss_agg_mode, loss_scale=loss_scale).detach().item(),
        } 

        if self.pipeline_config.use_rollout_importance_sampling_ratio:
            loss_metric["actor/rollout_importance_sampling_clip"] = rollout_importance_sampling_clip.mean().detach().item()

        pg_metrics = {
            "actor/pg_loss": original_pg_loss.detach().item(),
            "actor/weighted_pg_loss": weighted_pg_loss.detach().item(),
            "actor/kl_loss": kl_loss.detach().item(),
            "actor/total_loss": total_loss.detach().item(),
            "actor/approxkl": agg_loss(loss_mat=approxkl, loss_mask=response_mask,
                                       loss_agg_mode=self.pipeline_config.loss_agg_mode).detach().item(),
            "actor/policykl": agg_loss(loss_mat=policykl, loss_mask=response_mask,
                                       loss_agg_mode=self.pipeline_config.loss_agg_mode).detach().item(),
            "actor/valid_samples": valid_samples.sum().detach().item(),
            "actor/total_samples": float(valid_samples.size(0)),
            "actor/valid_sample_ratio": (valid_samples.sum() / valid_samples.size(0)).detach().item(),
            "actor/sample_weights_mean": sample_weights.mean().detach().item(),
            "actor/sample_weights_min": sample_weights.min().detach().item(),
            "actor/sample_weights_max": sample_weights.max().detach().item(),
            **metrics,
            **loss_metric,
            **train_infer_prob_metric
        }
        pg_metrics.update(
            self._get_partition_metrics(
                data=data,
                response_mask=response_mask,
                final_response_mask=final_response_mask,
                base_final_response_mask=base_final_response_mask,
                sample_weights=sample_weights,
                valid_samples=valid_samples,
                ratio=ratio,
                surr1=surr1,
                surr2=surr2,
                pg_loss_mat=loss,
                kl_loss_mat=kl_loss_mat,
                approxkl=approxkl,
                policykl=policykl,
                train_infer_ratio=train_infer_ratio,
                train_infer_diff=train_infer_diff,
                train_infer_ratio_mask=train_infer_ratio_mask,
                train_infer_diff_mask=train_infer_diff_mask,
                train_infer_ratio_seq_mask=train_infer_ratio_seq_mask,
                train_infer_diff_seq_mask=train_infer_diff_seq_mask,
                loss_scale=loss_scale,
            )
        )

        return total_loss, pg_metrics

    def compute_sample_weights(self, data: DataProto, response_mask: torch.Tensor):
        """
        可以基于难度和长度的样本权重
        """
        batch_size = response_mask.shape[0]
        sample_weights = torch.ones(batch_size, device=response_mask.device)

        # 1. 基于难度的权重 - 例如：难度越高，权重越大
        if self.pipeline_config.difficulty_loss_weight and "difficulty" in data.non_tensor_batch:
            try:
                difficulty = data.non_tensor_batch["difficulty"]
                if isinstance(difficulty, np.ndarray):
                    difficulty = torch.tensor(difficulty, dtype=torch.float32, device=response_mask.device)
                elif not isinstance(difficulty, torch.Tensor):
                    difficulty = torch.tensor(difficulty, dtype=torch.float32, device=response_mask.device)
                norm_difficulty = torch.clamp(difficulty, 0.0, 1.0)
                difficulty_weights = 0.5 + 1.5 * norm_difficulty
                sample_weights = sample_weights * difficulty_weights
            except Exception as e:
                self.logger.warning(f"跳过difficulty权重计算：{str(e)}")

        # 2. 基于长度的权重 - 例如：长度越长，权重越小
        response_lengths = response_mask.sum(dim=1).float()
        if self.pipeline_config.length_loss_weight:
            # 同样归一化长度到[0.5, 2.0]范围
            norm_lengths = (response_lengths - response_lengths.min()) / (
                    response_lengths.max() - response_lengths.min() + 1e-8
            )
            length_weights = 1.5 - norm_lengths
            sample_weights = sample_weights * length_weights

        if sample_weights.sum() > 0:
            sample_weights = sample_weights * (batch_size / (sample_weights.sum() + 1e-8))

        return sample_weights
