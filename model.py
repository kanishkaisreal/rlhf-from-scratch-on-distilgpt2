"""
RLHF from Scratch on DistilGPT2

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_distilgpt2_tokenizer
from transformers import AutoTokenizer

def load_distilgpt2_tokenizer(model_name="sshleifer/tiny-gpt2"):
    # TODO: load and return the Hugging Face tokenizer for the given model name.
    tokenizer =  AutoTokenizer.from_pretrained(model_name)
    return tokenizer

# Step 2 - load_distilgpt2_model
from transformers import AutoModelForCausalLM

def load_distilgpt2_model(model_name="sshleifer/tiny-gpt2"):
    # TODO: load a causal LM by name and return it in eval mode
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    model.eval()
    return model

# Step 3 - set_pad_token_to_eos
def set_pad_token_to_eos(tokenizer):
    # TODO: assign tokenizer.pad_token = tokenizer.eos_token and return the tokenizer
    tokenizer.pad_token = tokenizer.eos_token 
    return tokenizer

# Step 4 - generate_and_decode
def generate_and_decode(model, tokenizer, prompt, max_new_tokens=8):
    # TODO: tokenize prompt, generate continuation greedily, decode and return as a string
    tokenizer = set_pad_token_to_eos(tokenizer)

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return text

# Step 5 - greedy_decode
import torch

def greedy_decode(logits):
    """Return the argmax token id from a single-row logits vector."""
    # TODO: return the token id with the largest logit as a Python int
    return torch.argmax(logits).item()

# Step 6 - sample_with_temperature
def sample_with_temperature(logits, temperature):
    # TODO: rescale logits by temperature, softmax, and sample one token id

    scaled_logits = logits / temperature
    probs = torch.softmax(scaled_logits, dim=-1)
    token_id = torch.multinomial(probs, num_samples=1)
    return token_id.squeeze(-1).item()

# Step 7 - top_k_filter
def top_k_filter(logits, k):
    # TODO: keep the k largest entries of logits and set the rest to -inf.
     # logits vector of shape (V,),
    if k<= 0 or k>= logits.shape[0]:
        return logits
    
    top_indices = torch.topk(logits, k).indices

    filtered = torch.full_like(logits, float("-inf"))
    filtered[top_indices] = logits[top_indices]
    return filtered

# Step 8 - top_p_filter
def top_p_filter(logits, p):
    # TODO: mask logits outside the smallest cumulative-probability nucleus of size p.
    logits = torch.as_tensor(logits)
    
    if p >= 1.0 :
        return logits
    
    sorted_logits, sorted_indices = torch.sort(logits, descending = True)

    sorted_probs  = torch.softmax(sorted_logits, dim=0)

    cumulative_probs = torch.cumsum(sorted_probs, dim = 0 ) 

    keep_sorted = cumulative_probs <=p 

    first_crossing = torch.nonzero(cumulative_probs >= p)[0]
    keep_sorted[first_crossing] = True

    keep_sorted[0] = True

    keep_indices = sorted_indices[keep_sorted]

    filtered = torch.full_like(logits, float("-inf"))
    filtered[keep_indices] = logits[keep_indices]

    return filtered

# Step 9 - build_synthetic_instruction_dataset
def build_synthetic_instruction_dataset():
    return [
        {
            "prompt": "Explain reinforcement learning in one sentence.",
            "response": "Reinforcement learning trains an agent to make good decisions by rewarding useful actions."
        },
        {
            "prompt": "What does a tokenizer do?",
            "response": "A tokenizer converts text into token IDs and token IDs back into text."
        },
        {
            "prompt": "Explain greedy decoding.",
            "response": "Greedy decoding chooses the token with the highest logit at each generation step."
        },
        {
            "prompt": "What is supervised fine-tuning?",
            "response": "Supervised fine-tuning trains a pretrained model on prompt and response examples."
        },
        {
            "prompt": "Why do we use an attention mask?",
            "response": "An attention mask tells the model which tokens are real and which tokens are padding."
        },
    ]

# Step 10 - format_example
def format_example(example):
    prompt = example["prompt"]
    response = example["response"]

    return f"### Instruction:\n{prompt}\n\n### Response:\n{response}"

# Step 11 - apply_template
def apply_template(examples):
    # TODO: apply format_example to each item in examples and return the list of strings.
     return [format_example(example) for example in examples]

# Step 12 - tokenize_example
def tokenize_example(tokenizer, text, max_length=64):
    # TODO: encode `text` with truncation at max_length, no padding, return list[int]
    encoded = tokenizer(
        text, 
        truncation=True,
        max_length = max_length,
        padding= False,
    )

    return encoded["input_ids"]

# Step 13 - build_labels
def build_labels(input_ids):
    # TODO: return a fresh list equal to input_ids to serve as next-token labels
    return input_ids.copy()

# Step 14 - mask_prompt_labels
def mask_prompt_labels(labels, prompt_length):
    # TODO: replace the first prompt_length entries of labels with -100 and return the new list
    masked = labels.copy()
    n = min(prompt_length, len(masked))
    masked[:prompt_length] = [-100] * n

    return masked

# Step 15 - pad_batch
def pad_batch(sequences, pad_id):
    # TODO: right-pad a list of token id sequences to the longest length using pad_id
    max_len = max(len(seq) for seq in sequences)

    padded = [] 
    for seq in sequences:
        num_pad = max_len - len(seq)
        padded_seq = seq + [pad_id] *num_pad 
        padded.append(padded_seq)
    
    return padded

# Step 16 - make_attention_mask (not yet solved)
# TODO: implement

# Step 17 - collate_lm_batch (not yet solved)
# TODO: implement

# Step 18 - iterate_minibatches (not yet solved)
# TODO: implement

# Step 19 - train_val_split (not yet solved)
# TODO: implement

# Step 20 - shift_logits_and_labels (not yet solved)
# TODO: implement

# Step 21 - cross_entropy_loss (not yet solved)
# TODO: implement

# Step 22 - adamw_update (not yet solved)
# TODO: implement

# Step 23 - linear_warmup_schedule (not yet solved)
# TODO: implement

# Step 24 - clip_grad_norm (not yet solved)
# TODO: implement

# Step 25 - accumulate_gradients (not yet solved)
# TODO: implement

# Step 26 - sft_train_step (not yet solved)
# TODO: implement

# Step 27 - evaluate_loss (not yet solved)
# TODO: implement

# Step 28 - lora_delta (not yet solved)
# TODO: implement

# Step 29 - lora_linear_forward (not yet solved)
# TODO: implement

# Step 30 - init_lora_weights (not yet solved)
# TODO: implement

# Step 31 - freeze_base_params (not yet solved)
# TODO: implement

# Step 32 - count_trainable_params (not yet solved)
# TODO: implement

# Step 33 - merge_lora (not yet solved)
# TODO: implement

# Step 34 - build_synthetic_preference_dataset (not yet solved)
# TODO: implement

# Step 35 - format_preference (not yet solved)
# TODO: implement

# Step 36 - reward_head_forward (not yet solved)
# TODO: implement

# Step 37 - pairwise_reward_loss (not yet solved)
# TODO: implement

# Step 38 - reward_bce_loss (not yet solved)
# TODO: implement

# Step 39 - pairwise_accuracy (not yet solved)
# TODO: implement

# Step 40 - reward_train_step (not yet solved)
# TODO: implement

# Step 41 - sequence_logprob (not yet solved)
# TODO: implement

# Step 42 - per_token_kl (not yet solved)
# TODO: implement

# Step 43 - compute_returns (not yet solved)
# TODO: implement

# Step 44 - gae_advantages (not yet solved)
# TODO: implement

# Step 45 - policy_ratio (not yet solved)
# TODO: implement

# Step 46 - clipped_surrogate (not yet solved)
# TODO: implement

# Step 47 - value_function_loss (not yet solved)
# TODO: implement

# Step 48 - entropy_bonus (not yet solved)
# TODO: implement

# Step 49 - ppo_loss (not yet solved)
# TODO: implement

# Step 50 - kl_penalized_reward (not yet solved)
# TODO: implement

# Step 51 - batch_sequence_logprob (not yet solved)
# TODO: implement

# Step 52 - dpo_logratios (not yet solved)
# TODO: implement

# Step 53 - dpo_ref_logratios (not yet solved)
# TODO: implement

# Step 54 - dpo_loss (not yet solved)
# TODO: implement

# Step 55 - ipo_loss (not yet solved)
# TODO: implement

# Step 56 - kto_loss (not yet solved)
# TODO: implement

# Step 57 - orpo_loss (not yet solved)
# TODO: implement

# Step 58 - simpo_loss (not yet solved)
# TODO: implement

# Step 59 - build_eval_prompt_set (not yet solved)
# TODO: implement

# Step 60 - generate_completions (not yet solved)
# TODO: implement

# Step 61 - score_with_reward (not yet solved)
# TODO: implement

# Step 62 - win_rate (not yet solved)
# TODO: implement

# Step 63 - stream_tokens (not yet solved)
# TODO: implement

# Step 64 - apply_stop_tokens (not yet solved)
# TODO: implement

# Step 65 - chat (not yet solved)
# TODO: implement

