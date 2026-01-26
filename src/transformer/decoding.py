"""
Decoding strategies for sequence generation.

Provides:
- greedy_decode: Simple greedy decoding
- beam_search: Beam search decoding (more sophisticated)

These are used for inference with encoder-decoder models.
"""

import torch
from .attention import subsequent_mask


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    """
    Greedy decoding for sequence generation.

    At each step, selects the token with highest probability.
    Simple but may not find the globally optimal sequence.

    Args:
        model: EncoderDecoder model
        src: Source sequence, shape (1, src_len)
        src_mask: Source attention mask
        max_len: Maximum output sequence length
        start_symbol: Start token index (typically <s> or <bos>)

    Returns:
        Generated sequence, shape (1, output_len)

    Example:
        >>> model.eval()
        >>> src = torch.LongTensor([[1, 2, 3, 4, 5]])
        >>> src_mask = torch.ones(1, 1, 5)
        >>> output = greedy_decode(model, src, src_mask, max_len=10, start_symbol=0)
    """
    # Encode source sequence
    memory = model.encode(src, src_mask)

    # Start with just the start symbol
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)

    for _ in range(max_len - 1):
        # Decode current sequence
        out = model.decode(
            memory,
            src_mask,
            ys,
            subsequent_mask(ys.size(1)).type_as(src.data)
        )

        # Get probabilities for next token
        prob = model.generator(out[:, -1])

        # Select token with highest probability
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]

        # Append to sequence
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)],
            dim=1
        )

    return ys


def beam_search(
    model,
    src,
    src_mask,
    max_len,
    start_symbol,
    end_symbol,
    beam_size=4,
    length_penalty=0.6,
):
    """
    Beam search decoding for sequence generation.

    Maintains multiple hypothesis sequences and selects the best one.
    Generally produces better results than greedy decoding but is slower.

    Args:
        model: EncoderDecoder model
        src: Source sequence, shape (1, src_len)
        src_mask: Source attention mask
        max_len: Maximum output sequence length
        start_symbol: Start token index
        end_symbol: End token index (for early stopping)
        beam_size: Number of beams to maintain (default: 4)
        length_penalty: Penalty factor for sequence length (default: 0.6)
                       Higher values favor longer sequences.

    Returns:
        Best generated sequence, shape (1, output_len)

    Example:
        >>> model.eval()
        >>> src = torch.LongTensor([[1, 2, 3, 4, 5]])
        >>> src_mask = torch.ones(1, 1, 5)
        >>> output = beam_search(
        ...     model, src, src_mask,
        ...     max_len=20, start_symbol=0, end_symbol=1,
        ...     beam_size=4
        ... )
    """
    device = src.device

    # Encode source sequence
    memory = model.encode(src, src_mask)

    # Initialize beams: (score, sequence)
    # Start with just the start symbol
    beams = [(0.0, [start_symbol])]

    # Completed sequences
    completed = []

    for _ in range(max_len - 1):
        all_candidates = []

        for score, seq in beams:
            # Check if this beam is complete
            if seq[-1] == end_symbol:
                completed.append((score, seq))
                continue

            # Create tensor from sequence
            ys = torch.LongTensor([seq]).to(device)

            # Expand memory for this beam
            mem = memory

            # Decode
            out = model.decode(
                mem,
                src_mask,
                ys,
                subsequent_mask(ys.size(1)).to(device)
            )

            # Get log probabilities for next token
            log_probs = model.generator(out[:, -1])  # (1, vocab)

            # Get top k candidates
            top_log_probs, top_indices = torch.topk(
                log_probs[0], beam_size
            )

            for log_prob, idx in zip(top_log_probs.tolist(), top_indices.tolist()):
                new_score = score + log_prob
                new_seq = seq + [idx]
                all_candidates.append((new_score, new_seq))

        # Select top beam_size candidates
        # Apply length penalty for fair comparison
        def score_with_penalty(item):
            score, seq = item
            length = len(seq)
            return score / (length ** length_penalty)

        all_candidates.sort(key=score_with_penalty, reverse=True)
        beams = all_candidates[:beam_size]

        # Early stopping if all beams are complete
        if all(seq[-1] == end_symbol for _, seq in beams):
            break

    # Add remaining beams to completed
    completed.extend(beams)

    # Select best sequence
    if completed:
        completed.sort(key=lambda x: x[0] / (len(x[1]) ** length_penalty), reverse=True)
        best_seq = completed[0][1]
    else:
        best_seq = beams[0][1]

    return torch.LongTensor([best_seq]).to(device)


def sample_decode(
    model,
    src,
    src_mask,
    max_len,
    start_symbol,
    temperature=1.0,
    top_k=None,
    top_p=None,
):
    """
    Sampling-based decoding with temperature and top-k/top-p filtering.

    More diverse than greedy/beam search, useful for creative generation.

    Args:
        model: EncoderDecoder model
        src: Source sequence, shape (1, src_len)
        src_mask: Source attention mask
        max_len: Maximum output sequence length
        start_symbol: Start token index
        temperature: Sampling temperature (default: 1.0)
                    Higher = more random, lower = more deterministic
        top_k: Keep only top k tokens (optional)
        top_p: Keep tokens with cumulative probability >= top_p (optional)

    Returns:
        Generated sequence, shape (1, output_len)
    """
    device = src.device

    # Encode source sequence
    memory = model.encode(src, src_mask)

    # Start with just the start symbol
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)

    for _ in range(max_len - 1):
        # Decode current sequence
        out = model.decode(
            memory,
            src_mask,
            ys,
            subsequent_mask(ys.size(1)).type_as(src.data)
        )

        # Get log probabilities for next token
        log_probs = model.generator(out[:, -1])  # (1, vocab)
        probs = (log_probs / temperature).exp()

        # Apply top-k filtering
        if top_k is not None:
            top_k_probs, top_k_indices = torch.topk(probs, top_k)
            probs = torch.zeros_like(probs).scatter_(1, top_k_indices, top_k_probs)

        # Apply top-p (nucleus) filtering
        if top_p is not None:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Keep at least one token
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            probs[indices_to_remove] = 0

        # Normalize
        probs = probs / probs.sum(dim=-1, keepdim=True)

        # Sample
        next_word = torch.multinomial(probs, num_samples=1)

        # Append to sequence
        ys = torch.cat([ys, next_word], dim=1)

    return ys
