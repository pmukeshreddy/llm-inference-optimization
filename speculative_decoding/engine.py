"""
Speculative Decoding Engine
Production implementation with adaptive speculation
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Generator
from dataclasses import dataclass
import time


@dataclass
class SpeculativeConfig:
    max_speculation_length: int = 5
    min_speculation_length: int = 2
    adaptive_k: bool = True
    temperature: float = 1.0
    top_p: float = 0.9


@dataclass
class GenerationOutput:
    tokens: torch.Tensor
    text: str
    metrics: Dict
    

class SpeculativeDecodingEngine:
    def __init__(
        self,
        draft_model,
        target_model,
        tokenizer,
        config: Optional[SpeculativeConfig] = None,
        device: str = "cuda"
    ):
        self.draft_model = draft_model
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.config = config or SpeculativeConfig()
        self.device = device
        self.current_k = self.config.max_speculation_length
        self.acceptance_history = []
        
    def _sample_token(
        self, 
        logits: torch.Tensor, 
        temperature: float = 1.0,
        top_p: float = 0.9
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if temperature == 0:
            token = logits.argmax(dim=-1, keepdim=True)
            prob = torch.ones_like(token, dtype=torch.float)
            return token, prob
            
        logits = logits / temperature
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = float('-inf')
        
        probs = F.softmax(logits, dim=-1)
        token = torch.multinomial(probs, num_samples=1)
        prob = probs.gather(dim=-1, index=token)
        
        return token, prob
    
    def _draft_generate(self, input_ids: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        draft_tokens = []
        draft_probs = []
        current_ids = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(k):
                outputs = self.draft_model(current_ids)
                logits = outputs.logits[:, -1, :]
                token, prob = self._sample_token(logits, self.config.temperature, self.config.top_p)
                draft_tokens.append(token)
                draft_probs.append(prob)
                current_ids = torch.cat([current_ids, token], dim=-1)
        
        return torch.cat(draft_tokens, dim=-1), torch.cat(draft_probs, dim=-1)
    
    def _verify_tokens(
        self,
        input_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        draft_probs: torch.Tensor
    ) -> Tuple[torch.Tensor, int]:
        full_sequence = torch.cat([input_ids, draft_tokens], dim=-1)
        
        with torch.no_grad():
            outputs = self.target_model(full_sequence)
            target_logits = outputs.logits
        
        k = draft_tokens.shape[-1]
        start_pos = input_ids.shape[-1] - 1
        
        accepted_tokens = []
        n_accepted = 0
        
        for i in range(k):
            pos = start_pos + i
            target_probs = F.softmax(target_logits[:, pos, :], dim=-1)
            draft_token = draft_tokens[:, i:i+1]
            
            target_prob = target_probs.gather(dim=-1, index=draft_token).squeeze(-1)
            draft_prob = draft_probs[:, i]
            
            acceptance_ratio = (target_prob / (draft_prob + 1e-10)).clamp(max=1.0)
            
            if torch.rand(1, device=self.device) < acceptance_ratio:
                accepted_tokens.append(draft_token)
                n_accepted += 1
            else:
                adjusted_logits = target_logits[:, pos, :]
                new_token, _ = self._sample_token(adjusted_logits, self.config.temperature, self.config.top_p)
                accepted_tokens.append(new_token)
                n_accepted += 1
                break
        
        if n_accepted == k:
            final_logits = target_logits[:, start_pos + k, :]
            bonus_token, _ = self._sample_token(final_logits, self.config.temperature, self.config.top_p)
            accepted_tokens.append(bonus_token)
            n_accepted += 1
        
        return torch.cat(accepted_tokens, dim=-1), n_accepted
    
    def _update_adaptive_k(self, acceptance_rate: float):
        if not self.config.adaptive_k:
            return
        self.acceptance_history.append(acceptance_rate)
        if len(self.acceptance_history) > 10:
            self.acceptance_history = self.acceptance_history[-10:]
        
        avg_acceptance = sum(self.acceptance_history) / len(self.acceptance_history)
        
        if avg_acceptance > 0.8 and self.current_k < self.config.max_speculation_length:
            self.current_k += 1
        elif avg_acceptance < 0.5 and self.current_k > self.config.min_speculation_length:
            self.current_k -= 1
    
    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        stream: bool = False
    ) -> GenerationOutput:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        generated_tokens = input_ids.clone()
        
        total_draft_tokens = 0
        total_accepted_tokens = 0
        target_forward_passes = 0
        ttft = None
        start_time = time.perf_counter()
        
        while generated_tokens.shape[-1] - input_ids.shape[-1] < max_new_tokens:
            draft_tokens, draft_probs = self._draft_generate(generated_tokens, self.current_k)
            total_draft_tokens += self.current_k
            
            accepted, n_accepted = self._verify_tokens(generated_tokens, draft_tokens, draft_probs)
            target_forward_passes += 1
            total_accepted_tokens += n_accepted
            
            if ttft is None:
                ttft = time.perf_counter() - start_time
            
            generated_tokens = torch.cat([generated_tokens, accepted], dim=-1)
            self._update_adaptive_k(n_accepted / self.current_k)
            
            if self.tokenizer.eos_token_id in accepted:
                break
            
            if stream:
                yield self.tokenizer.decode(accepted[0], skip_special_tokens=True)
        
        total_time = time.perf_counter() - start_time
        new_tokens = generated_tokens.shape[-1] - input_ids.shape[-1]
        
        metrics = {
            "total_tokens": new_tokens,
            "total_time_s": total_time,
            "tokens_per_second": new_tokens / total_time,
            "ttft_s": ttft,
            "acceptance_rate": total_accepted_tokens / max(total_draft_tokens, 1),
            "avg_tokens_per_forward": total_accepted_tokens / max(target_forward_passes, 1),
            "target_forward_passes": target_forward_passes,
            "speculation_length": self.current_k,
        }
        
        output_text = self.tokenizer.decode(generated_tokens[0, input_ids.shape[-1]:], skip_special_tokens=True)
        
        return GenerationOutput(tokens=generated_tokens, text=output_text, metrics=metrics)
    
    @torch.inference_mode()
    def generate_vanilla(self, prompt: str, max_new_tokens: int = 100) -> GenerationOutput:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        generated_tokens = input_ids.clone()
        start_time = time.perf_counter()
        ttft = None
        
        for _ in range(max_new_tokens):
            outputs = self.target_model(generated_tokens)
            logits = outputs.logits[:, -1, :]
            token, _ = self._sample_token(logits, self.config.temperature, self.config.top_p)
            
            if ttft is None:
                ttft = time.perf_counter() - start_time
            
            generated_tokens = torch.cat([generated_tokens, token], dim=-1)
            
            if token.item() == self.tokenizer.eos_token_id:
                break
        
        total_time = time.perf_counter() - start_time
        new_tokens = generated_tokens.shape[-1] - input_ids.shape[-1]
        
        metrics = {
            "total_tokens": new_tokens,
            "total_time_s": total_time,
            "tokens_per_second": new_tokens / total_time,
            "ttft_s": ttft,
            "method": "vanilla"
        }
        
        output_text = self.tokenizer.decode(generated_tokens[0, input_ids.shape[-1]:], skip_special_tokens=True)
        return GenerationOutput(tokens=generated_tokens, text=output_text, metrics=metrics)
