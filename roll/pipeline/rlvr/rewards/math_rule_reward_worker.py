import logging

# WARNING:latex2sympy2_extended.math_normalization:equations is deprecated, as it handled by the parser now
logging.getLogger('latex2sympy2_extended.math_normalization').setLevel(logging.ERROR)

from functools import partial
from typing import Optional, Union, Iterator
import json
import re

import ray
import torch
from math_verify import parse, verify
from codetiming import Timer
from tqdm import tqdm
import signal
import multiprocessing

from roll.configs.worker_config import WorkerConfig
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.factory import create_strategy
from roll.distributed.strategy.strategy import InferenceStrategy, TrainStrategy
from roll.models.model_providers import default_reward_model_provider, default_tokenizer_provider
from roll.utils.context_managers import state_offload_manger

class timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)
        
def _extract_after_last_end_think(response: str, prompt: str, start_think: str='<think>', end_think: str='</think>') -> str:
    """
    提取字符串中最后一个 "</think>" 标签之后的所有文本。

    校验逻辑会根据 prompt 的结尾而变化：
    - (1) 如果 prompt 的结尾（去掉换行符后）是以 "<think>" 结尾：
        - response 中不允许包含开标签 "<think>"。
        - response 中包含的闭标签 "</think>" 不能超过一个。
        - 若不满足，则返回空字符串。
    - (2) 否则（prompt 不以 "<think>" 结尾）：
        - response 中包含的闭标签 "</think>" 不能超过一个。
        - 如果 response 中包含开标签 "<think>"，它必须出现在字符串的开头。
        - 若不满足，则返回空字符串。

    如果校验通过，则执行提取逻辑：
    1. 优先按最后一个 '</think>' 分割。
    2. 如果找不到，则回退到按最后一个双换行符 '\n\n' 分割。
    3. 如果都找不到，则返回空字符串。

    Args:
        response (str): 输入的完整文本。
        prompt (str): 用于生成 response 的提示文本。

    Returns:
        str: 提取出的文本块（已去除首尾空格），或空字符串。
    """
    # 检查 prompt 是否以 <think> 结尾
    is_prompt_ending_with_think = prompt.rstrip('\n').endswith(start_think)

    if is_prompt_ending_with_think:
        if start_think in response or response.count(end_think) > 1:
            return ""
    else:        
        if response.count(end_think) > 1 or start_think in response and not response.startswith(start_think):
            return ""

    # 1. 优先尝试按 '</think>' 分割
    _before_think, sep_think, after_think = response.rpartition(end_think)

    if sep_think:
        # 如果找到了 '</think>'，则返回它后面的部分，并清理首尾空格
        return after_think.strip()
    else:
        # 2. 如果没找到 '</think>'，则尝试按最后一个 '\n\n' 分割
        _before_newline, sep_newline, after_newline = response.rpartition('\n\n')
        if sep_newline:
            # 如果找到了 '\n\n'，返回它后面的部分，并清理首尾空格
            return after_newline.strip()
        else:
            # 3. 如果连 '\n\n' 都没找到，则返回空字符串
            return ""

def _last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed")
    prefix = "\\boxed"
    if idx < 0:
        idx = string.rfind("\\fbox")
        prefix = "\\fbox"
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        elif string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return string[idx:right_brace_idx + 1]


def _remove_boxed(boxed_string: Optional[str]) -> Optional[str]:
    if boxed_string is None:
        return None

    for left in ("\\boxed{", "\\fbox{"):
        if boxed_string.startswith(left) and boxed_string.endswith("}"):
            return boxed_string[len(left):-1]
    return None


def _extract_verl_boxed_answer(cleaned_response: str, raw_response: Optional[str] = None) -> Optional[str]:
    candidate_response = cleaned_response.strip() if cleaned_response else ""
    if not candidate_response and raw_response is not None:
        candidate_response = raw_response.strip()
    if not candidate_response:
        return None
    last_line = candidate_response.splitlines()[-1].strip()
    if not last_line:
        return None
    return _last_boxed_only_string(last_line)


def _extract_predicted_answer(cleaned_response: str, reward_type: Optional[str], raw_response: Optional[str] = None):
    if reward_type == "verl_boxed":
        extracted_answer = _extract_verl_boxed_answer(cleaned_response, raw_response=raw_response)
        if extracted_answer is None:
            return None, ""
        parsed_answers = parse(extracted_answer, fallback_mode="no_fallback")
        if not parsed_answers:
            return None, extracted_answer
        return parsed_answers[0], extracted_answer

    parsed_answers = parse(cleaned_response, fallback_mode="no_fallback")
    if not parsed_answers:
        return None, ""
    return parsed_answers[0], str(parsed_answers[0])


def _hf_verify_math_sample_multi(response, answers, result, prompt, reward_type=None):
    try:
        # 在解析之前，先对模型的原始输出进行预处理
        cleaned_response = _extract_after_last_end_think(response, prompt)
        """
        --- `parse` 函数完整参数介绍与使用建议 ---
        `parse` 函数用于从文本中提取并解析数学答案，其主要参数如下：
        
        1. `pred` (位置参数): 需要被解析的输入字符串。
           => 建议：传入净化后的文本（如 cleaned_response），可以显著提高准确率。
        
        2. `extraction_config` (关键字参数): 定义要寻找的答案类型。
           => 默认值: [LatexExtractionConfig(), ExprExtractionConfig()] (寻找LaTeX和纯数字)
           => 建议：对于数学计算题，保持默认即可。
        
        3. `fallback_mode` (关键字参数): 定义当找到答案文本但无法成功解析时怎么办。
           => 默认值: "first_match" (返回原始匹配的字符串)
           => 强烈建议: 设为 "no_fallback"，这样在解析失败时会返回空列表[]，避免输出垃圾内容。
        
        4. `extraction_mode` (关键字参数): 定义搜寻答案的范围。
           => 默认值: "any_match" (搜寻全文，找到第一个能成功解析的答案)
           => 建议：保持默认值，因为它更可能在复杂文本中找到正确答案。
        
        5. `parsing_timeout` (关键字参数): 解析单个表达式的超时时间（秒）。
           => 默认值: 5
           => 建议：保留默认值，作为防止程序卡死的安全保护。
        
        6. `raise_on_error` (关键字参数): 遇到内部程序错误时是否抛出异常。
           => 默认值: False (不抛出异常，返回空列表)
           => 建议：保持默认值，确保程序的健壮性，不会因单个样本出错而中断。
        """
        exect_answer, extracted_response = _extract_predicted_answer(
            cleaned_response,
            reward_type,
            raw_response=response,
        )

        verify_results = []
        for answer in answers:
            parsed_target = parse(answer)
            if parsed_target is None or exect_answer is None:
                verify_results.append((False, "", ""))
            else:
                ans = verify(parsed_target[0], exect_answer)
                verify_results.append((ans, str(parsed_target[0]), str(extracted_response)))
        result.append(verify_results)
            
    except Exception as e:
        # 捕获任何潜在的异常，确保进程不会崩溃
        result.append([(False, "", "") for _ in answers])


def hf_verify_math_sample_multi(response, answers, prompt, timeout_sec=5.0, reward_type=None):
    with multiprocessing.Manager() as manager:
        result = manager.list()
        
        p = multiprocessing.Process(
            target=_hf_verify_math_sample_multi,
            args=(response, answers, result, prompt, reward_type)
        )
        
        p.start()
        try:
            max_timeout = min(timeout_sec + 1, 10)
            p.join(timeout=max_timeout)
        except Exception as e:
            pass
        finally:
            if p.is_alive():
                p.terminate()
                p.join(timeout=2)
                if p.is_alive():
                    p.kill()
            p.join(timeout=2)
        if not result:
            return [(False, "", "") for _ in answers]
        return result[0]


def hf_verify_math_sample(response, answer, prompt, timeout_sec=5.0, reward_type=None):
    return hf_verify_math_sample_multi(
        response=response,
        answers=[answer],
        prompt=prompt,
        timeout_sec=timeout_sec,
        reward_type=reward_type,
    )[0]

def get_repetition_penalty_reward(ngram_size: int, max_penalty: float):
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")
    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])
    def repetition_penalty_reward(response, **kwargs) -> float:
        if response == "" or len(response.split()) < ngram_size:
            return 0.0
        ngrams = set()
        total = 0
        for ng in zipngram(response, ngram_size):
            ngrams.add(ng)
            total += 1
        scaling = 1 - len(ngrams) / total
        reward = scaling * max_penalty
        return reward
    return repetition_penalty_reward

def long_block_penalty_reward_fn(text: str, max_length: int = 100) -> float:
    max_block_len = max([len(i) for i in text.split(" ")])
    reward = -float(max_block_len > max_length)
    return reward

def format_reward_fn(text: str, pattern: Optional[str] = r"^<think>.*?</think>.*?<answer>.*?</answer>$"):
    if pattern is None:
        pattern: str = r"^<think>.*?</think>.*?<answer>.*?</answer>$"
    matche = re.match(pattern, text, re.DOTALL | re.MULTILINE)
    reward = 0 if matche else -1
    return reward


class MathRuleRewardWorker(Worker):
    """
    (x)Reward Model 使用 AutoModelForSequenceClassification 协议
    面向math的rule reward model
    """

    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        self.rank_info.dp_rank = self.rank_info.rank
        self.rank_info.dp_size = self.rank_info.world_size
        self.tokenizer = default_tokenizer_provider(model_args=self.worker_config.model_args)
        self.strategy: Optional[Union[InferenceStrategy, TrainStrategy]] = None
        self.repetition_penalty_reward_fn = get_repetition_penalty_reward(ngram_size=3, max_penalty=-0.1)
        self.format_pattern = getattr(self.worker_config, "format_pattern", None)
        self.reward_type = getattr(self.worker_config, "reward_type", None)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        pass

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE, clear_cache=False)
    def compute_rewards(self, data: DataProto):
        verify_answer = []
        gold_verify_answer = []
        repetition_penalty_rewards = []
        long_block_penalty_rewards = []
        response_length_rewards = []
        format_rewards = []
        
        response_text_list = self.tokenizer.batch_decode(data.batch["responses"], skip_special_tokens=False)
        prompt_text_list = self.tokenizer.batch_decode(data.batch["prompts"], skip_special_tokens=False)
        gold_answer_list = data.non_tensor_batch.get("gold_answer", data.non_tensor_batch["ground_truth"])
        for response, answer, gold_answer, prompt in zip(
            response_text_list,
            data.non_tensor_batch["ground_truth"],
            gold_answer_list,
            prompt_text_list,
        ):
            
            prompt = prompt.replace("<|endoftext|>", "").replace("<pad>", "")
            response = response.replace("<|endoftext|>", "").replace("<pad>", "")
            # self.logger.info(json.dumps({
            #     "prompt": prompt}, ensure_ascii=False))
            
            try:
                with timeout(5):
                    verify_inputs = [f"${answer}$"]
                    gold_answer = answer if gold_answer is None else str(gold_answer)
                    if gold_answer != answer:
                        verify_inputs.append(f"${gold_answer}$")
                    verify_results = hf_verify_math_sample_multi(
                        response,
                        verify_inputs,
                        prompt,
                        reward_type=self.reward_type,
                    )

                correct, extracted_ground_truth, extracted_response = verify_results[0]
                if gold_answer == answer:
                    gold_correct = correct
                else:
                    gold_correct = verify_results[1][0]
            
                log_data = {
                    "response": response,
                    "extracted_response": extracted_response,
                    "answer": answer,
                    "gold_answer": gold_answer,
                    "extracted_ground_truth": extracted_ground_truth,
                    "correct": correct,
                    "gold_correct": gold_correct,
                }
                # self.logger.info(json.dumps(log_data, ensure_ascii=False))

            except Exception as e:
                self.logger.error(f"timeout or error during hf_verify_math_sample. answer: {answer}, response: {response}")
                correct = False
                gold_correct = False
                extracted_response = ""
                extracted_ground_truth = ""
            
            if correct:
                verify_answer.append(1)
            else:
                verify_answer.append(0)
            if gold_correct:
                gold_verify_answer.append(1)
            else:
                gold_verify_answer.append(0)
            repetition_penalty_rewards.append(self.repetition_penalty_reward_fn(response))
            format_rewards.append(format_reward_fn(response, self.format_pattern))
            long_block_penalty_rewards.append(long_block_penalty_reward_fn(response))
            response_length_rewards.append(len(response) / 20000)
            
        token_level_rewards = torch.zeros_like(data.batch["responses"], dtype=torch.float16)
        response_length_rewards = torch.tensor(response_length_rewards, dtype=torch.float16)
        repetition_penalty_rewards = torch.tensor(repetition_penalty_rewards, dtype=torch.float16)
        long_block_penalty_rewards = torch.tensor(long_block_penalty_rewards, dtype=torch.float16)
        format_rewards = torch.tensor(format_rewards, dtype=torch.float16)
        scores = torch.tensor(verify_answer, dtype=torch.float16)
        gold_scores = torch.tensor(gold_verify_answer, dtype=torch.float16)
        response_level_rewards = torch.tensor(verify_answer, dtype=torch.float16)

        output = DataProto.from_dict(
            tensors={
                "token_level_rewards": token_level_rewards,
                "response_level_rewards": response_level_rewards,
                "scores": scores,
                "gold_scores": gold_scores,
            }
        )

        self.logger.debug(f"reward output: {output}, response_level_rewards: {response_level_rewards}")
        return output
