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

# Step 16 - make_attention_mask
def make_attention_mask(padded_ids, pad_id):
    # TODO: return a same-shape 0/1 mask with 1 where token != pad_id else 0
    mask = []

    for seq in padded_ids:
        row =[] 
        for token_id in seq: 
            if token_id == pad_id:
                row.append(0)
            else:
                row.append(1)
        mask.append(row)
    
    return mask

# Step 17 - collate_lm_batch
def collate_lm_batch(batch, pad_id):
    # TODO: pad input_ids and labels, build attention mask, return dict of LongTensors
    input_ids = [example["input_ids"] for example in batch]
    labels = [example["labels"] for example in batch]

    padded_input_ids = pad_batch(input_ids, pad_id)
    padded_labels = pad_batch(labels, -100)
    attention_mask = make_attention_mask(padded_input_ids, pad_id)

    return {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "labels": torch.tensor(padded_labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }

# Step 18 - iterate_minibatches
import random 

def iterate_minibatches(examples, batch_size, seed=0):
    # TODO: yield shuffled minibatches of size batch_size from examples (deterministic per seed).
    rng = random.Random(seed)

    shuffled = examples.copy()
    rng.shuffle(shuffled)

    batches = [] 
    for start in range(0, len(shuffled), batch_size):
        batch = shuffled[start:start + batch_size]
        batches.append(batch)
    
    return batches

# Step 19 - train_val_split
import random 

def train_val_split(examples, val_ratio=0.2, seed=0):
    # TODO: deterministically split examples into (train, val) using seed and val_ratio
    rng = random.Random(seed)

    shuffled = examples.copy()
    rng.shuffle(shuffled)    

    n_val = int(len(shuffled) *val_ratio)

    val = shuffled[: n_val]

    train = shuffled[n_val:]

    return train, val

# Step 20 - shift_logits_and_labels
def shift_logits_and_labels(logits, labels):
    # TODO: drop the last logit position and the first label position so token t predicts t+1
    shifted_logits = logits[:,:-1,:] 
    shifted_labels = labels[:,1:]

    return shifted_logits, shifted_labels







# Suppose:

# input tokens / labels:
# [A, B, C, D]

# Original positions:

# labels:
# position 0: A
# position 1: B
# position 2: C
# position 3: D

# Model logits:

# logits position 0 predicts next token after A
# logits position 1 predicts next token after B
# logits position 2 predicts next token after C
# logits position 3 predicts next token after D

# But there is no target after D inside this sequence.

# So we use:

# shifted_logits:
# position 0, 1, 2

# shifted_labels:
# position 1, 2, 3

# Step 21 - cross_entropy_loss
import torch
import torch.nn.functional as F

def cross_entropy_loss(shift_logits, shift_labels):
    """Mean next-token cross-entropy, ignoring label positions equal to -100."""
    # TODO: reduce (B, T-1, V) logits and (B, T-1) labels to a scalar loss tensor.
    B, T_minus_1, V = shift_logits.shape 

    flat_logits = shift_logits.reshape(B*T_minus_1, V)
    flat_labels = shift_labels.reshape(B*T_minus_1)

    loss = F.cross_entropy(
        flat_logits,
        flat_labels,
        ignore_index = -100
    )

    return loss

# Step 22 - adamw_update
import torch

def adamw_update(param, grad, state, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
    """Apply one in-place AdamW step to `param` using `grad` and persistent `state`."""
    # TODO: initialize state on first call, then update moments and apply the decoupled AdamW step
    beta1, beta2 = betas 

    if "step" not in state:
        state["step"] = 0
        state["m"] = torch.zeros_like(param)
        state["v"] = torch.zeros_like(param)

    # Step counter must increase every update
    state["step"] += 1
    step = state["step"]

    m = state["m"]
    v = state["v"]

    # Adam moment updates
    m.mul_(beta1).add_(grad, alpha=1 - beta1)
    v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

    # Bias correction
    m_hat = m / (1 - beta1 ** step)
    v_hat = v / (1 - beta2 ** step)

    # AdamW decoupled weight decay
    if weight_decay != 0.0:
        param.mul_(1 - lr * weight_decay)

    # Adaptive parameter update
    param.addcdiv_(m_hat, torch.sqrt(v_hat) + eps, value=-lr)

    return param

# Step 23 - linear_warmup_schedule
def linear_warmup_schedule(step, warmup_steps):
    # TODO: return a linear warmup multiplier in [0, 1] given the current step and warmup window.
    if warmup_steps == 0 : 
        return 1 
    return min(1, step/ warmup_steps)

# Step 24 - clip_grad_norm
def clip_grad_norm(grads, max_norm):
    # TODO: compute the global L2 norm of grads and rescale in place if it exceeds max_norm.
    total_sq = 0.0

    for grad in grads:
        total_sq += grad.pow(2).sum().item()

    total_norm  = total_sq ** 0.5 

    if total_norm  > max_norm :
        scale = max_norm / total_norm 
    
        for grad in grads:
            grad.mul_(scale)

    return total_norm

# Step 25 - accumulate_gradients
import torch

def accumulate_gradients(grad_list):
    """Average a list of equally-shaped gradient tensors across micro-batches."""
    # TODO: average a list of equally-shaped gradient tensors and return the mean tensor
    total = torch.zeros_like(grad_list[0])

    for grad in grad_list:
        total = total + grad

    avg_grad = total / len(grad_list)

    return avg_grad

# Step 26 - sft_train_step
import torch

def sft_train_step(model, batch, optimizer):
    """Run one SFT forward/backward/step and return the loss as a float."""
    # TODO: forward the batch, compute shifted cross-entropy loss, backprop, step optimizer

    model.train()

    optimizer.zero_grad()

    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )

    logits = outputs.logits

    shifted_logits, shifted_labels = shift_logits_and_labels(
        logits,
        batch["labels"],
    )

    loss = cross_entropy_loss(shifted_logits, shifted_labels)

    loss.backward()

    optimizer.step()

    return loss.item()

# Step 27 - evaluate_loss
import torch

def evaluate_loss(model, batches):
    """Mean LM loss over validation batches, no grad."""
    # TODO: iterate batches under no_grad, shift logits/labels, average cross-entropy.

    model.eval()

    losses = []

    with torch.no_grad():
        for batch in batches:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

            logits = outputs.logits

            shifted_logits, shifted_labels = shift_logits_and_labels(
                logits,
                batch["labels"],
            )

            loss = cross_entropy_loss(shifted_logits, shifted_labels)

            losses.append(loss.item())

    return sum(losses) / len(losses)

# Step 28 - lora_delta
def lora_delta(A, B, alpha, r):
    # TODO: build the scaled low-rank weight update from factors A and B.
    rank = A.shape[0] 
    scale = alpha / rank
    return scale * (B @ A)

# Step 29 - lora_linear_forward
def lora_linear_forward(x, base_weight, A, B, alpha, r, bias=None):
    # TODO: return x @ (base_weight + lora_delta).T (+ bias) using lora_delta(A, B, alpha, r)
    delta = lora_delta(A, B, alpha, r)

    effective_weight = base_weight + delta 

    output = x @ effective_weight.T

    if bias is not None:
        output = output + bias 
    
    return output

# Step 30 - init_lora_weights
import torch

def init_lora_weights(in_features, out_features, r, seed=0):
    """Return (A, B) LoRA factors with random A and zero B so the initial delta is zero."""
    # TODO: seed torch, build A of shape (r, in_features) and B of shape (out_features, r)
    torch.manual_seed(seed)

    A = 0.01 * torch.randn(r, in_features)
    B = torch.zeros(out_features, r)

    return A, B

# Step 31 - freeze_base_params
def freeze_base_params(model):
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model

# Step 32 - count_trainable_params
def count_trainable_params(model):
    # TODO: sum p.numel() over parameters with requires_grad=True
    total = 0 
    for param in model.parameters():
        if param.requires_grad:
            total += param.numel()
        
    
    return total 



# numel 
# number of scalar values inside the tensor

# weight.shape = (4, 3)

# Then:

# weight.numel() = 4 * 3 = 12

# A shape = (r, d_in)
# B shape = (d_out, r)

# Trainable LoRA count:

# A params = r * d_in
# B params = d_out * r

# total LoRA params = r * d_in + d_out * r
#                   = r * (d_in + d_out)

# Step 33 - merge_lora
def merge_lora(base_weight, lora_a, lora_b, scaling):
    # TODO: fold the scaled low-rank update B @ A back into the base weight matrix.
    delta = scaling * (lora_b @ lora_a)
    merged_weight = base_weight + delta
    return merged_weight

# Step 34 - build_synthetic_preference_dataset
def build_synthetic_preference_dataset(num_examples=8, seed=0):
    pool = [
        {
            "prompt": "What is the capital of France?",
            "chosen": "The capital of France is Paris.",
            "rejected": "I do not know.",
        },
        {
            "prompt": "What is 2 + 2?",
            "chosen": "2 + 2 equals 4.",
            "rejected": "2 + 2 equals 5.",
        },
        {
            "prompt": "What color is the sky on a clear day?",
            "chosen": "The sky is blue on a clear day.",
            "rejected": "The sky is green.",
        },
        {
            "prompt": "What animal says meow?",
            "chosen": "A cat says meow.",
            "rejected": "A dog says meow.",
        },
        {
            "prompt": "What is the opposite of hot?",
            "chosen": "The opposite of hot is cold.",
            "rejected": "The opposite of hot is hotter.",
        },
        {
            "prompt": "How many days are in a week?",
            "chosen": "There are 7 days in a week.",
            "rejected": "There are 10 days in a week.",
        },
        {
            "prompt": "What do plants need to grow?",
            "chosen": "Plants need water, sunlight, and nutrients to grow.",
            "rejected": "Plants need only darkness to grow.",
        },
        {
            "prompt": "What is the first letter of the alphabet?",
            "chosen": "The first letter of the alphabet is A.",
            "rejected": "The first letter of the alphabet is Z.",
        },
    ]

    examples = []

    for i in range(num_examples):
        index = (seed + i) % len(pool)
        examples.append(pool[index].copy())

    return examples

# Step 35 - format_preference
def format_preference(example):
    chosen_text = example["prompt"] + " " + example["chosen"]
    rejected_text = example["prompt"] + " " + example["rejected"]


    return {
        "chosen_text": chosen_text,
        "rejected_text": rejected_text,
    }

# Step 36 - reward_head_forward
import torch

def reward_head_forward(hidden_state, weight, bias):
    """Map a final hidden state to a scalar reward via a linear projection."""
    # TODO: project hidden_state (B, D) through weight (D,) plus scalar bias to get (B,) rewards
    rewards = hidden_state @ weight.T + bias
    return rewards.squeeze(-1)

# Step 37 - pairwise_reward_loss
import torch
import torch.nn.functional as F

def pairwise_reward_loss(chosen, rejected):
    diff = chosen - rejected
    loss = -F.logsigmoid(diff).mean()
    return loss

# Step 38 - reward_bce_loss
import numpy as np

def reward_bce_loss(chosen, rejected):
    # TODO: BCE-style reward loss with chosen as positives and rejected as negatives.
    chosen_loss = np.logaddexp(0, -chosen)
    rejected_loss = np.logaddexp(0, rejected)

    loss = np.mean(
        np.concatenate([chosen_loss, rejected_loss])
    )

    return loss

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

