from typing import Dict, Any, List, Optional
import torch
import numpy as np
from ray.util.timer import _Timer
from codetiming import Timer

from roll.utils.functionals import agg_loss, compute_approx_kl, masked_mean, reduce_metrics


def _to_bool_numpy(values) -> np.ndarray:
    return np.asarray([bool(val) for val in values], dtype=bool)


def _std_value(values: torch.Tensor) -> float:
    if values.numel() <= 1:
        return 0.0
    return torch.std(values.float(), unbiased=False).item()


class MetricsManager:
    """
    Organizes metrics for PPO pipeline
    """

    def __init__(self):
        self.metrics = {}
        self.domain_metrics = {}
        self.timers = {}

    def add_metric(self, name: str, value: Any) -> None:
        self.metrics[name] = value

    def add_metrics(self, metrics_dict: Dict[str, Any]) -> None:
        self.metrics.update(metrics_dict)

    def add_reduced_metrics(self, metrics_dict: Dict[str, Any], prefix: str = "", reduce_func=np.mean) -> None:
        reduced = reduce_metrics(metrics_dict, reduce_func=reduce_func)
        if prefix:
            reduced = {f"{prefix}/{k}": v for k, v in reduced.items()}
        self.metrics.update(reduced)

    def add_domain_metrics(self, domain: str, metrics_dict: Dict[str, Any]) -> None:
        if not metrics_dict:
            return

        if domain not in self.domain_metrics:
            self.domain_metrics[domain] = {}

        self.domain_metrics[domain].update(metrics_dict)

    def get_metrics(self) -> Dict[str, Any]:
        all_metrics = self.metrics.copy()

        for domain, domain_metrics in self.domain_metrics.items():
            for key, value in domain_metrics.items():
                all_metrics[f"{domain}/{key}"] = value

        return all_metrics

    def clear_metrics(self) -> None:
        self.metrics = {}
        self.domain_metrics = {}

    def add_system_metrics(
        self, global_step: int, batch_size: int, resource_manager=None, actor_infer=None, actor_train=None
    ) -> None:
        self.metrics["system/global_step"] = global_step
        self.metrics["system/batch_size"] = batch_size
        self.metrics["system/samples"] = (global_step + 1) * batch_size
        for name, timer in self.timers.items():
            if hasattr(timer, "mean_throughput"):
                self.metrics[f"system/{name}/tps"] = timer.mean_throughput
                if hasattr(timer, "mean"):
                    self.metrics[f"system/time/{name}_mean"] = timer.mean

                if resource_manager and name == "tps":
                    self.metrics["system/tps_gpu"] = timer.mean_throughput / resource_manager.num_gpus

                if actor_infer and name == "actor_infer":
                    self.metrics["system/actor_infer/tps_gpu"] = timer.mean_throughput / actor_infer.world_size
                    self.metrics["system/actor_infer/tps_dp"] = timer.mean_throughput / actor_infer.dp_size

                if actor_infer and name == "actor_infer_response":
                    self.metrics["system/actor_infer/response/tps_gpu"] = (
                        timer.mean_throughput / actor_infer.world_size
                    )
                    self.metrics["system/actor_infer/response/tps_dp"] = timer.mean_throughput / actor_infer.dp_size

                if actor_train and name == "actor_train":
                    self.metrics["system/actor_train/tps_gpu"] = timer.mean_throughput / actor_train.world_size
                    self.metrics["system/actor_train/tps_dp"] = timer.mean_throughput / actor_train.dp_size

    def add_timer_metrics(self, timer_dict: Dict[str, Timer]) -> None:
        for name, timer in timer_dict.items():
            if hasattr(timer, "last"):
                self.metrics[f"time/{name}"] = timer.last

    def add_token_metrics(self, batch, prefix: str = "token", record: bool = True) -> Dict[str, Any]:
        response_mask = batch.batch["response_mask"][:, 1:].bool()
        prompt_mask = batch.batch["prompt_mask"].bool()

        max_response_length = batch.batch["responses"].shape[-1]
        prompt_length = prompt_mask.sum(-1).float()
        response_length = response_mask.sum(-1).float()
        sequence_score = batch.batch["scores"]

        max_score = 1
        min_score = 0
        correct_mask = sequence_score == max_score
        incorrect_mask = sequence_score == min_score

        metrics = {}

        prompt_length_max = torch.max(prompt_length).detach().item()
        prompt_length_min = torch.min(prompt_length).detach().item()
        prompt_length_mean = torch.mean(prompt_length).detach().item()

        metrics[f"{prefix}/prompt_length/mean"] = prompt_length_mean
        metrics[f"{prefix}/prompt_length/max"] = prompt_length_max
        metrics[f"{prefix}/prompt_length/min"] = prompt_length_min

        response_length_max = torch.max(response_length).detach().item()
        response_length_min = torch.min(response_length).detach().item()
        response_length_mean = torch.mean(response_length).detach().item()
        response_length_diff = response_length_max - response_length_min

        metrics[f"{prefix}/response_length/mean"] = response_length_mean
        metrics[f"{prefix}/response_length/max"] = response_length_max
        metrics[f"{prefix}/response_length/min"] = response_length_min
        metrics[f"{prefix}/response_length/diff"] = response_length_diff

        total_length = prompt_length + response_length
        total_length_max = torch.max(total_length).detach().item()
        total_length_min = torch.min(total_length).detach().item()
        total_length_mean = torch.mean(total_length).detach().item()
        total_length_diff = total_length_max - total_length_min

        metrics[f"{prefix}/total_length/mean"] = total_length_mean
        metrics[f"{prefix}/total_length/max"] = total_length_max
        metrics[f"{prefix}/total_length/min"] = total_length_min
        metrics[f"{prefix}/total_length/diff"] = total_length_diff

        try:
            metrics[f"{prefix}/total_response_length/clip"] = (
                torch.sum(response_length == max_response_length).detach().item()
            )
        except:
            pass

        try:
            metrics[f"{prefix}/right_response_length/clip"] = (
                torch.sum(response_length[correct_mask] == max_response_length).detach().item()
            )
            metrics[f"{prefix}/right_response_length/mean"] = (
                torch.mean(response_length[correct_mask & (response_length != max_response_length)]).detach().item()
            )
            metrics[f"{prefix}/right_response_length/max"] = (
                torch.max(response_length[correct_mask & (response_length != max_response_length)]).detach().item()
            )
            metrics[f"{prefix}/right_response_length/min"] = (
                torch.min(response_length[correct_mask & (response_length != max_response_length)]).detach().item()
            )
        except:
            pass

        try:
            metrics[f"{prefix}/error_response_length/clip"] = (
                torch.sum(response_length[incorrect_mask] == max_response_length).detach().item()
            )
            metrics[f"{prefix}/error_response_length/mean"] = (
                torch.mean(response_length[incorrect_mask & (response_length != max_response_length)]).detach().item()
            )
            metrics[f"{prefix}/error_response_length/max"] = (
                torch.max(response_length[incorrect_mask & (response_length != max_response_length)]).detach().item()
            )
            metrics[f"{prefix}/error_response_length/min"] = (
                torch.min(response_length[incorrect_mask & (response_length != max_response_length)]).detach().item()
            )
        except:
            pass
        if record:
            self.add_metrics(metrics)
        return metrics

    def add_values_metrics(
        self,
        batch,
        prefix: str = "critic",
        record: bool = True,
        entropy_value: Optional[float] = None,
    ) -> Dict[str, Any]:
        metrics = {}

        sequence_score = batch.batch["scores"]
        sequence_reward = batch.batch["token_level_rewards"].sum(-1)
        sequence_reward_mean = batch.batch["token_level_rewards"].mean(-1)

        advantages = batch.batch["advantages"]
        prompt_mask = batch.batch["prompt_mask"].bool()
        response_mask = batch.batch["final_response_mask"].clone().bool()
        raw_advantages = batch.batch["raw_advantages"]
        returns = batch.batch["returns"]
        if entropy_value is None:
            agg_entropy = batch.meta_info.get("agg_entropy", torch.tensor(0))
            entropy_value = agg_entropy.item() if torch.is_tensor(agg_entropy) else float(agg_entropy)

        max_score = 1
        min_score = 0

        correct_mask = sequence_score == max_score
        incorrect_mask = sequence_score == min_score

        metrics[f"{prefix}/entropy/mean"] = float(entropy_value)
        metrics[f"{prefix}/correct/mean"] = (sequence_score == max_score).detach().float().mean().item()

        metrics[f"{prefix}/score_distribution/max_value"] = max_score
        metrics[f"{prefix}/score_distribution/min_value"] = min_score
        metrics[f"{prefix}/score_distribution/correct_samples_ratio"] = (
            (sequence_score == max_score).float().mean().item()
        )
        metrics[f"{prefix}/score_distribution/incorrect_samples_ratio"] = (
            (sequence_score == min_score).float().mean().item()
        )

        metrics[f"{prefix}/score/mean"] = torch.mean(sequence_score).detach().item()
        metrics[f"{prefix}/score/max"] = torch.max(sequence_score).detach().item()
        metrics[f"{prefix}/score/min"] = torch.min(sequence_score).detach().item()

        metrics[f"{prefix}/rewards/mean"] = torch.mean(sequence_reward).detach().item()
        metrics[f"{prefix}/rewards/max"] = torch.max(sequence_reward).detach().item()
        metrics[f"{prefix}/rewards/min"] = torch.min(sequence_reward).detach().item()
        metrics[f"{prefix}/token_level_rewards_mean/mean"] = torch.mean(sequence_reward_mean).detach().item()
        metrics[f"{prefix}/token_level_rewards_mean/max"] = torch.max(sequence_reward_mean).detach().item()
        metrics[f"{prefix}/token_level_rewards_mean/min"] = torch.min(sequence_reward_mean).detach().item()

        metrics[f"{prefix}/advantages/mean"] = masked_mean(advantages, response_mask).detach().item()
        if torch.any(response_mask):
            metrics[f"{prefix}/advantages/max"] = torch.max(advantages[response_mask]).detach().item()
            metrics[f"{prefix}/advantages/min"] = torch.min(advantages[response_mask]).detach().item()

        correct_mask_expanded = correct_mask.unsqueeze(-1).expand_as(response_mask)
        correct_response_mask = response_mask & correct_mask_expanded
        if torch.any(correct_response_mask):
            metrics[f"{prefix}/right_response/advantages/mean"] = (
                masked_mean(advantages, correct_response_mask).detach().item()
            )
            metrics[f"{prefix}/right_response/advantages/max"] = (
                torch.max(advantages[correct_response_mask]).detach().item()
            )
            metrics[f"{prefix}/right_response/advantages/min"] = (
                torch.min(advantages[correct_response_mask]).detach().item()
            )
            metrics[f"{prefix}/right_response/advantages/std"] = (
                torch.std(advantages[correct_response_mask]).detach().item()
            )

        incorrect_mask_expanded = incorrect_mask.unsqueeze(-1).expand_as(response_mask)
        incorrect_response_mask = response_mask & incorrect_mask_expanded
        if torch.any(incorrect_response_mask):
            metrics[f"{prefix}/error_response/advantages/mean"] = (
                masked_mean(advantages, incorrect_response_mask).detach().item()
            )
            metrics[f"{prefix}/error_response/advantages/max"] = (
                torch.max(advantages[incorrect_response_mask]).detach().item()
            )
            metrics[f"{prefix}/error_response/advantages/min"] = (
                torch.min(advantages[incorrect_response_mask]).detach().item()
            )
            metrics[f"{prefix}/error_response/advantages/std"] = (
                torch.std(advantages[incorrect_response_mask]).detach().item()
            )

        metrics[f"{prefix}/raw_advantages/mean"] = masked_mean(raw_advantages, response_mask).detach().item()
        if torch.any(response_mask):
            metrics[f"{prefix}/raw_advantages/max"] = torch.max(raw_advantages[response_mask]).detach().item()
            metrics[f"{prefix}/raw_advantages/min"] = torch.min(raw_advantages[response_mask]).detach().item()

        if torch.any(correct_response_mask):
            metrics[f"{prefix}/right_response/raw_advantages/mean"] = (
                masked_mean(raw_advantages, correct_response_mask).detach().item()
            )
            metrics[f"{prefix}/right_response/raw_advantages/max"] = (
                torch.max(raw_advantages[correct_response_mask]).detach().item()
            )
            metrics[f"{prefix}/right_response/raw_advantages/min"] = (
                torch.min(raw_advantages[correct_response_mask]).detach().item()
            )
            metrics[f"{prefix}/right_response/raw_advantages/std"] = (
                torch.std(raw_advantages[correct_response_mask]).detach().item()
            )

        if torch.any(incorrect_response_mask):
            metrics[f"{prefix}/error_response/raw_advantages/mean"] = (
                masked_mean(raw_advantages, incorrect_response_mask).detach().item()
            )
            metrics[f"{prefix}/error_response/raw_advantages/max"] = (
                torch.max(raw_advantages[incorrect_response_mask]).detach().item()
            )
            metrics[f"{prefix}/error_response/raw_advantages/min"] = (
                torch.min(raw_advantages[incorrect_response_mask]).detach().item()
            )
            metrics[f"{prefix}/error_response/raw_advantages/std"] = (
                torch.std(raw_advantages[incorrect_response_mask]).detach().item()
            )

        metrics[f"{prefix}/returns/mean"] = masked_mean(returns, response_mask).detach().item()
        if torch.any(response_mask):
            metrics[f"{prefix}/returns/max"] = torch.max(returns[response_mask]).detach().item()
            metrics[f"{prefix}/returns/min"] = torch.min(returns[response_mask]).detach().item()

        if "values" in batch.batch.keys():
            values = batch.batch["values"]
            metrics[f"{prefix}/values/mean"] = masked_mean(values, response_mask).detach().item()
            if torch.any(response_mask):
                metrics[f"{prefix}/values/max"] = torch.max(values[response_mask]).detach().item()
                metrics[f"{prefix}/values/min"] = torch.min(values[response_mask]).detach().item()

        if record:
            self.add_metrics(metrics)
        return metrics

    def add_group_metrics(self, batch, n_sample: int, prefix: str = "group", record: bool = True) -> Dict[str, Any]:
        if n_sample <= 1:
            return {}

        metrics = {}

        sequence_score = batch.batch["scores"]
        response_mask = batch.batch["response_mask"][:, 1:].bool()
        response_length = response_mask.sum(-1).float()
        advantages = batch.batch["advantages"]

        total_samples = sequence_score.shape[0]
        num_prompts = total_samples // n_sample

        grouped_scores = sequence_score.reshape(num_prompts, n_sample)
        grouped_response_length = response_length.reshape(num_prompts, n_sample)

        max_length_per_group = torch.max(grouped_response_length, dim=1)[0]
        min_length_per_group = torch.min(grouped_response_length, dim=1)[0]
        length_diff_per_group = max_length_per_group - min_length_per_group

        metrics[f"{prefix}/response_length_diff/mean"] = torch.mean(length_diff_per_group).item()
        metrics[f"{prefix}/response_length_diff/max"] = torch.max(length_diff_per_group).item()
        metrics[f"{prefix}/response_length_diff/min"] = torch.min(length_diff_per_group).item()

        max_score = 1
        min_score = 0

        correct_mask_grouped = grouped_scores == max_score
        incorrect_mask_grouped = grouped_scores == min_score

        correct_ratio_per_group = correct_mask_grouped.float().mean(dim=1)
        metrics[f"{prefix}/correct_ratio/mean"] = torch.mean(correct_ratio_per_group).item()
        metrics[f"{prefix}/correct_ratio/std"] = torch.std(correct_ratio_per_group).item()

        all_correct_groups = torch.sum(correct_mask_grouped.all(dim=1)).item()
        all_incorrect_groups = torch.sum(incorrect_mask_grouped.all(dim=1)).item()
        mixed_groups = num_prompts - all_correct_groups - all_incorrect_groups

        metrics[f"{prefix}/all_correct_groups_ratio"] = all_correct_groups / num_prompts
        metrics[f"{prefix}/all_incorrect_groups_ratio"] = all_incorrect_groups / num_prompts
        metrics[f"{prefix}/mixed_groups_ratio"] = mixed_groups / num_prompts

        if "advantages" in batch.batch:
            mean_adv_per_sample = masked_mean(advantages, response_mask, dim=1)
            grouped_advantages = mean_adv_per_sample.reshape(num_prompts, n_sample)

            max_adv_per_group = torch.max(grouped_advantages, dim=1)[0]
            min_adv_per_group = torch.min(grouped_advantages, dim=1)[0]
            adv_diff_per_group = max_adv_per_group - min_adv_per_group

            metrics[f"{prefix}/advantage_diff/mean"] = torch.mean(adv_diff_per_group).item()
            metrics[f"{prefix}/advantage_diff/max"] = torch.max(adv_diff_per_group).item()
            metrics[f"{prefix}/advantage_diff/min"] = torch.min(adv_diff_per_group).item()

            for group_idx in range(num_prompts):
                group_correct_mask = correct_mask_grouped[group_idx]
                group_incorrect_mask = incorrect_mask_grouped[group_idx]

                if torch.any(group_correct_mask) and torch.any(group_incorrect_mask):
                    correct_adv = grouped_advantages[group_idx, group_correct_mask]
                    incorrect_adv = grouped_advantages[group_idx, group_incorrect_mask]

                    if len(correct_adv) > 0 and len(incorrect_adv) > 0:
                        correct_adv_mean = torch.mean(correct_adv)
                        incorrect_adv_mean = torch.mean(incorrect_adv)

                        if "correct_incorrect_adv_diff" not in locals():
                            correct_incorrect_adv_diff = []

                        correct_incorrect_adv_diff.append(correct_adv_mean - incorrect_adv_mean)

            if "correct_incorrect_adv_diff" in locals() and len(correct_incorrect_adv_diff) > 0:
                correct_incorrect_adv_diff = torch.stack(correct_incorrect_adv_diff)
                metrics[f"{prefix}/correct_vs_incorrect_advantage_diff/mean"] = torch.mean(
                    correct_incorrect_adv_diff
                ).item()
                metrics[f"{prefix}/correct_vs_incorrect_advantage_diff/std"] = torch.std(
                    correct_incorrect_adv_diff
                ).item()

        if record:
            self.add_metrics(metrics)
        return metrics

    def add_mask_metrics(self, batch, prefix: str = "actor", record: bool = True) -> Dict[str, Any]:
        response_mask = batch.batch["response_mask"][:, 1:].bool()
        final_response_mask = batch.batch.get("final_response_mask", response_mask).bool()
        valid_samples = torch.any(final_response_mask, dim=1).float()

        metrics = {
            f"{prefix}/samples_used": valid_samples.sum().item(),
            f"{prefix}/samples_total": float(valid_samples.numel()),
            f"{prefix}/final_mask_ratio": valid_samples.mean().item(),
        }

        if response_mask.sum() > 0:
            metrics[f"{prefix}/token_keep_ratio"] = (
                final_response_mask.float().sum() / (response_mask.float().sum() + 1e-8)
            ).item()

        if record:
            self.add_metrics(metrics)
        return metrics

    def add_kl_metrics(
        self,
        batch,
        prefix: str = "critic",
        record: bool = True,
        kl_penalty: str = "kl",
    ) -> Dict[str, Any]:
        if "old_log_probs" not in batch.batch or "ref_log_probs" not in batch.batch:
            return {}

        response_mask = batch.batch["response_mask"][:, 1:].bool()
        if response_mask.sum() <= 0:
            return {}

        kld = compute_approx_kl(
            log_probs=batch.batch["old_log_probs"],
            log_probs_base=batch.batch["ref_log_probs"],
            action_mask=response_mask,
            kl_penalty=kl_penalty,
        )
        current_kl = masked_mean(kld, response_mask, dim=-1).mean().item()
        metrics = {f"{prefix}/kl": current_kl}

        if record:
            self.add_metrics(metrics)
        return metrics

    def _compute_entropy_value(self, batch, entropy_tensor: Optional[torch.Tensor]) -> Optional[float]:
        if entropy_tensor is None:
            return None
        response_mask = batch.batch["response_mask"][:, 1:]
        if response_mask.sum() <= 0:
            return 0.0
        return agg_loss(loss_mat=entropy_tensor, loss_mask=response_mask, loss_agg_mode="token-mean").item()

    def add_partition_all_metrics(
        self,
        batch,
        n_sample: int = -1,
        entropy_tensor: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if "is_noisy" not in batch.non_tensor_batch:
            return {}

        is_noisy = _to_bool_numpy(batch.non_tensor_batch["is_noisy"])
        total_count = len(is_noisy)
        metrics = {}
        partitions = {"clean": ~is_noisy, "noisy": is_noisy}

        for name, mask_np in partitions.items():
            sample_count = int(mask_np.sum())
            sample_ratio = float(sample_count / total_count) if total_count else 0.0
            metrics[f"{name}/data/sample_count"] = sample_count
            metrics[f"{name}/data/sample_ratio"] = sample_ratio
            if sample_count == 0:
                continue

            sub_batch = batch.select_idxs(mask_np)
            sub_metrics = {}
            sub_metrics.update(self.add_mask_metrics(sub_batch, prefix="actor", record=False))
            sub_metrics.update(self.add_kl_metrics(sub_batch, prefix="critic", record=False))
            sub_metrics.update(self.add_token_metrics(sub_batch, prefix="token", record=False))
            sub_metrics.update(
                self.add_values_metrics(
                    sub_batch,
                    prefix="critic",
                    record=False,
                    entropy_value=self._compute_entropy_value(
                        sub_batch,
                        entropy_tensor[torch.as_tensor(mask_np, dtype=torch.bool, device=entropy_tensor.device)]
                        if entropy_tensor is not None
                        else None,
                    ),
                )
            )
            if n_sample > 1 and len(sub_batch) >= n_sample and len(sub_batch) % n_sample == 0:
                sub_metrics.update(self.add_group_metrics(sub_batch, n_sample, prefix="group", record=False))
            metrics.update({f"{name}/{key}": value for key, value in sub_metrics.items()})

        self.add_metrics(metrics)
        return metrics

    def add_noisy_gold_target_metrics(
        self,
        batch,
        n_sample: int = -1,
        entropy_tensor: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if "is_noisy" not in batch.non_tensor_batch or "gold_scores" not in batch.batch:
            return {}

        noisy_mask_np = _to_bool_numpy(batch.non_tensor_batch["is_noisy"])
        noisy_count = int(noisy_mask_np.sum())
        if noisy_count == 0:
            return {}

        noisy_batch = batch.select_idxs(noisy_mask_np)
        metrics = {}

        target_correct = noisy_batch.batch["scores"].float() > 0.5
        gold_correct = noisy_batch.batch["gold_scores"].float() > 0.5
        response_rewards = noisy_batch.batch["response_level_rewards"].float()
        response_mask = noisy_batch.batch["response_mask"][:, 1:].bool()
        final_response_mask = noisy_batch.batch.get("final_response_mask", response_mask).bool()
        response_length = response_mask.sum(-1).float()

        metrics["noisy/target/correct_ratio"] = target_correct.float().mean().item()
        metrics["noisy/gold/correct_ratio"] = gold_correct.float().mean().item()
        metrics["noisy/diagnostics/target_gold_disagree_rate"] = (target_correct != gold_correct).float().mean().item()
        metrics["noisy/diagnostics/gold_minus_target_correct_rate"] = (
            gold_correct.float() - target_correct.float()
        ).mean().item()
        metrics["noisy/diagnostics/reward_flip_pos_rate"] = (
            (response_rewards > 0) & (~gold_correct)
        ).float().mean().item()
        metrics["noisy/diagnostics/reward_flip_neg_rate"] = (
            (response_rewards <= 0) & gold_correct
        ).float().mean().item()
        if gold_correct.any():
            metrics["noisy/diagnostics/neg_reward_on_gold_correct_rate"] = (
                (response_rewards[gold_correct] <= 0).float().mean().item()
            )
        if gold_correct.any():
            metrics["noisy/diagnostics/gold_correct_reward_mean"] = response_rewards[gold_correct].mean().item()
            metrics["noisy/diagnostics/gold_correct_length_mean"] = response_length[gold_correct].mean().item()
        if (~gold_correct).any():
            metrics["noisy/diagnostics/pos_reward_on_gold_wrong_rate"] = (
                (response_rewards[~gold_correct] > 0).float().mean().item()
            )
            metrics["noisy/diagnostics/gold_wrong_reward_mean"] = response_rewards[~gold_correct].mean().item()
            metrics["noisy/diagnostics/gold_wrong_length_mean"] = response_length[~gold_correct].mean().item()

        if "advantages" in noisy_batch.batch:
            sample_adv = masked_mean(noisy_batch.batch["advantages"], final_response_mask, dim=-1)
            metrics["noisy/diagnostics/adv_on_gold_correct_mean"] = (
                sample_adv[gold_correct].mean().item() if gold_correct.any() else 0.0
            )
            metrics["noisy/diagnostics/adv_on_gold_wrong_mean"] = (
                sample_adv[~gold_correct].mean().item() if (~gold_correct).any() else 0.0
            )
            if gold_correct.any():
                metrics["noisy/diagnostics/neg_adv_on_gold_correct_rate"] = (
                    (sample_adv[gold_correct] < 0).float().mean().item()
                )
            if (~gold_correct).any():
                metrics["noisy/diagnostics/pos_adv_on_gold_wrong_rate"] = (
                    (sample_adv[~gold_correct] > 0).float().mean().item()
                )

        if entropy_tensor is not None:
            noisy_entropy = entropy_tensor[torch.as_tensor(noisy_mask_np, dtype=torch.bool, device=entropy_tensor.device)]
            sample_entropy = masked_mean(noisy_entropy, response_mask, dim=-1)
            if gold_correct.any():
                metrics["noisy/diagnostics/gold_correct_entropy_mean"] = sample_entropy[gold_correct].mean().item()
            if (~gold_correct).any():
                metrics["noisy/diagnostics/gold_wrong_entropy_mean"] = sample_entropy[~gold_correct].mean().item()

        if n_sample > 1 and len(noisy_batch) >= n_sample and len(noisy_batch) % n_sample == 0:
            num_groups = len(noisy_batch) // n_sample
            grouped_target = target_correct.reshape(num_groups, n_sample)
            grouped_gold = gold_correct.reshape(num_groups, n_sample)
            target_count = grouped_target.sum(dim=1).float()
            gold_count = grouped_gold.sum(dim=1).float()
            gold_gap = gold_count - target_count

            metrics["noisy/group/target_count/mean"] = target_count.mean().item()
            metrics["noisy/group/target_count/std"] = _std_value(target_count)
            metrics["noisy/group/target_count/max"] = target_count.max().item()
            metrics["noisy/group/target_count/min"] = target_count.min().item()
            metrics["noisy/group/target_pass_rate"] = (target_count > 0).float().mean().item()
            metrics["noisy/group/target_correct_ratio/mean"] = (target_count / n_sample).mean().item()
            metrics["noisy/group/target_correct_ratio/std"] = _std_value(target_count / n_sample)

            metrics["noisy/group/gold_count/mean"] = gold_count.mean().item()
            metrics["noisy/group/gold_count/std"] = _std_value(gold_count)
            metrics["noisy/group/gold_count/max"] = gold_count.max().item()
            metrics["noisy/group/gold_count/min"] = gold_count.min().item()
            metrics["noisy/group/gold_pass_rate"] = (gold_count > 0).float().mean().item()
            metrics["noisy/group/gold_correct_ratio/mean"] = (gold_count / n_sample).mean().item()
            metrics["noisy/group/gold_correct_ratio/std"] = _std_value(gold_count / n_sample)
            metrics["noisy/group/gold_target_gap/mean"] = gold_gap.mean().item()
            metrics["noisy/group/gold_target_gap/std"] = _std_value(gold_gap)
            metrics["noisy/group/gold_target_gap/max"] = gold_gap.max().item()
            metrics["noisy/group/gold_target_gap/min"] = gold_gap.min().item()

            gold_all_correct = grouped_gold.all(dim=1).float().mean().item()
            gold_all_incorrect = (~grouped_gold).all(dim=1).float().mean().item()
            metrics["noisy/group/gold_all_correct_groups_ratio"] = gold_all_correct
            metrics["noisy/group/gold_all_incorrect_groups_ratio"] = gold_all_incorrect
            metrics["noisy/group/gold_mixed_groups_ratio"] = 1.0 - gold_all_correct - gold_all_incorrect

        self.add_metrics(metrics)
        return metrics

    def add_all_metrics(
        self, global_step, batch, n_sample=-1, resource_manager=None, actor_infer=None, actor_train=None
    ) -> None:
        batch_size = batch.batch.shape[0]
        # 添加system相关的指标
        self.add_system_metrics(
            global_step,
            batch_size,
            resource_manager=resource_manager,
            actor_infer=actor_infer,
            actor_train=actor_train,
        )
        self.add_mask_metrics(batch)
        self.add_kl_metrics(batch)
        # 添加token相关的指标
        self.add_token_metrics(batch)
        # 添加values相关的指标
        self.add_values_metrics(batch)

        if hasattr(batch, "meta_info") and "generation_config" in batch.meta_info:
            n_sample = batch.meta_info["generation_config"].get("num_return_sequences", 1)
        if n_sample > 1:
            self.add_group_metrics(batch, n_sample)

    def add_domain_all_metrics(self, global_step, batch_grouped: Dict[str, Any]) -> None:
        for domain, domain_batch in batch_grouped.items():
            original_metrics = self.metrics.copy()
            domain_metrics = self.add_values_metrics(batch=domain_batch)
            self.add_domain_metrics(domain, domain_metrics)

            token_metrics = self.add_token_metrics(batch=domain_batch)
            self.add_domain_metrics(domain, token_metrics)
            self.metrics = original_metrics
