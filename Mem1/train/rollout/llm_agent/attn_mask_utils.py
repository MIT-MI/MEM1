import torch
from torch.nn.utils.rnn import pad_sequence

def flatten_trajectory(traj, pad_token_id):
    """
    Returns:
      seq:       LongTensor [L]
      kind:      LongTensor [L]  (0=q,1=t,2=r,3=i)
      step:      LongTensor [L]
      prompt_len: int
    """
    segments = []
    kinds    = []
    steps    = []
    prompt_len = 0

    def append_segment(tokens, kind_code, step_idx):
        t = torch.tensor(tokens, dtype=torch.long) if not isinstance(tokens, torch.Tensor) else tokens
        mask = t.ne(pad_token_id)
        if mask.any():
            sel = t[mask]
            segments.append(sel)
            kinds.append(torch.full_like(sel, kind_code))
            steps.append(torch.full_like(sel, step_idx))

    # 1) question segment
    append_segment(traj["q"], kind_code=0, step_idx=0)
    prompt_len = segments[0].size(0)

    # 2) interaction steps t_j, r_j, (optional i_j)
    j = 0
    while j < traj["num_rounds"]:
        if f"t{j}" in traj:
            append_segment(traj[f"t{j}"], kind_code=1, step_idx=j)
        if f"r{j}" in traj:
            append_segment(traj[f"r{j}"], kind_code=2, step_idx=j)
        if f"i{j}" in traj:
            append_segment(traj[f"i{j}"], kind_code=3, step_idx=j)
        j += 1

    seq  = torch.cat(segments, dim=0)
    kind = torch.cat(kinds,    dim=0)
    step = torch.cat(steps,    dim=0)

    return seq, kind, step, prompt_len


def make_attention_mask(kind, step):
    """
    kind: LongTensor [L], values in {0=q,1=t,2=r,3=i}
    step: LongTensor [L]

    Returns:
      mask:       BoolTensor [L,L]
      info_mask:  BoolTensor [L]   (False for kind==3)
    """
    L = kind.size(0)
    ki = kind.unsqueeze(1)  # [L,1]
    kj = kind.unsqueeze(0)  # [1,L]
    si = step.unsqueeze(1)  # [L,1]
    sj = step.unsqueeze(0)  # [1,L]

    mask = torch.zeros((L, L), dtype=torch.bool)

    # each block attends to itself
    mask = (ki == kj) & (si == sj)

    # t tokens at step j may attend to q and (r,i) tokens at step j-1
    mask |= (ki == 1) & (kj == 0)  # attend to q
    mask |= (ki == 1) & (sj == si - 1) & ((kj == 2) | (kj == 3))  # attend to r,i at previous step

    # r tokens at step j may attend to q, (r,i) tokens at step j-1, and t tokens at the same step
    mask |= (ki == 2) & (kj == 0)  # attend to q
    mask |= (ki == 2) & (sj == si - 1) & ((kj == 2) | (kj == 3))  # attend to r,i at previous step
    mask |= (ki == 2) & (sj == si) & (kj == 1)  # attend to t at same step

    # i tokens (external info) attend to themselves
    mask |= (ki == 3) & (kj == 3) & (si == sj)

    # info_mask is False only for external information (kind == 3)
    info_mask = kind.ne(3)

    # make sure it is causal mask
    mask = torch.tril(mask)

    return mask, info_mask

def compose_final_output(trajectories, pad_token_id=0):
    # --- Flatten all trajectories ---
    results = [flatten_trajectory(traj, pad_token_id) for traj in trajectories]
    seqs, kinds, steps, prompt_lens = zip(*results)
    B = len(seqs)

    # --- Build batched prompts (left-padded) ---
    prompt_segments = [seq[:p] for seq, p in zip(seqs, prompt_lens)]
    rev_prompts     = [seg.flip(0) for seg in prompt_segments]
    rev_padded      = pad_sequence(rev_prompts, batch_first=True, padding_value=pad_token_id)
    prompts         = rev_padded.flip(1)  # [B, P_max]

    # --- Build batched responses (right-padded) ---
    response_segments = [seq[p:] for seq, p in zip(seqs, prompt_lens)]
    responses         = pad_sequence(response_segments, batch_first=True, padding_value=pad_token_id)  # [B, R_max]

    # --- Check if prompts and responses exceed 8196 ---
    # --- If so, truncate ---
    if prompts.size(1) + responses.size(1) > 8196:
        max_length = 8196 - prompts.size(1)
        responses = responses[:, :max_length]

    # --- Concatenate ---
    input_ids      = torch.cat([prompts, responses], dim=1)               # [B, S]
    attention_mask = input_ids.ne(pad_token_id).long()                    # [B, S]
    position_ids   = (attention_mask.cumsum(dim=1) - 1) * attention_mask  # [B, S]

    # --- Build 4D masks batch-wise ---
    masks = []
    info_masks = []
    P_max = prompts.size(1)
    for k, s, p_len in zip(kinds, steps, prompt_lens):
        small_mask, info_small = make_attention_mask(k, s)  # [L,L], [L]
        L = small_mask.size(0)
        S = input_ids.size(1)
        offset = P_max - p_len

        # make sure the small mask is not larger than the max allowed size
        L = min(L, S - offset)
        small_mask = small_mask[:L, :L]
        info_small = info_small[:L]

        big_mask = torch.zeros((S, S), dtype=torch.bool)
        big_mask[offset:offset+L, offset:offset+L] = small_mask
        im = torch.zeros((S,), dtype=torch.bool)
        im[offset:offset+L] = info_small

        masks.append(big_mask)
        info_masks.append(im)

    attention_mask_4d = torch.stack(masks,   dim=0).unsqueeze(1)  # [B,1,S,S]
    info_mask         = torch.stack(info_masks, dim=0)           # [B, S]

    return {
        "input_ids":         input_ids,
        "attention_mask":    attention_mask,
        "position_ids":      position_ids,
        "attention_mask_4d": attention_mask_4d.bool(),
        "info_mask":         info_mask,
        "prompts":           prompts,
        "responses":         responses,
    }


# Example usage:
if __name__ == "__main__":
    # generate [512, 4096] trajectorie
    trajectories = [
        { "q": [1,2,3], "t0":[201,202], "r0":[301],   "i0":[401,402], "t1":[203],  "r1":[302,303], "i1": [302, 303],  "t2":[204],     "r2":[304,305], "i2":[404,405],  "num_rounds": 3 },
        # { "q": [1,2,3], "t0":[213],      "r0":[313,314],"i0":[413],      "t1":[214],     "r1":[314],     "num_rounds": 2 },
    ]
    # trajectories = trajectories * 256
    batch = compose_final_output(trajectories, pad_token_id=0)
    print("input_ids shape:        ", batch["input_ids"].shape)        # [2, S]
    print("attention_mask_4d shape:", batch["attention_mask_4d"].shape) # [2, 1, S, S]

    print("prompts shape:          ", batch["prompts"].shape)           # [2, P_max]
    print("responses shape:        ", batch["responses"].shape)         # [2, R_max]
