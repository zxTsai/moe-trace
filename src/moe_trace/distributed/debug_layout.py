"""Explicit selector-row mapping for the pinned eager Ascend prepare paths."""
from ..errors import IntegrityError, ConfigError


def split_span(length, size, rank):
    if type(length) is not int or length < 0 or type(size) is not int or size < 1 or not 0 <= rank < size:
        raise IntegrityError("invalid tensor split")
    q, r = divmod(length, size)
    start = rank * q + min(rank, r)
    return start, start + q + (rank < r)


def selector_rows(kind, *, input_size, valid_indices, tp_size, tp_rank, dp_rank,
                  output_size, sequence_parallel=False, padded_size=None,
                  dp_sizes=None, ep_sizes=None):
    """Return (forward row, selector row), never infer the mode from shape alone.

    No collective is added. Foreign DP tokens and padding are not attributed to
    local requests. Full DP AllGather is emitted only by TP0.
    """
    if (type(input_size) is not int or input_size < 1 or type(output_size) is not int or output_size < 0
            or type(tp_size) is not int or tp_size < 1 or type(tp_rank) is not int
            or not 0 <= tp_rank < tp_size or type(dp_rank) is not int or dp_rank < 0):
        raise IntegrityError("invalid topology or tensor sizes")
    indices = list(valid_indices)
    if (len(set(indices)) != len(indices)
            or any(type(i) is not int or not 0 <= i < input_size for i in indices)):
        raise IntegrityError("invalid effective forward indices")
    if kind in {"PrepareAndFinalizeWithAll2All", "PrepareAndFinalizeWithMC2"}:
        if sequence_parallel:
            chunk = (input_size + tp_size - 1) // tp_size
            start, end = tp_rank * chunk, (tp_rank + 1) * chunk
            expected = chunk
            if kind.endswith("MC2"):
                if type(padded_size) is not int or padded_size % tp_size or padded_size // tp_size < chunk:
                    raise IntegrityError("invalid MC2 SP padding")
                expected = padded_size // tp_size
        else:
            total = max(input_size, tp_size) if kind.endswith("All2All") else padded_size
            if type(total) is not int or total < input_size:
                raise IntegrityError("invalid prepare padding")
            start, end = split_span(total, tp_size, tp_rank)
            expected = end - start
        if output_size != expected:
            raise IntegrityError("selector rows differ from actual prepare layout")
        return [(i, i - start) for i in indices if start <= i < end]
    if kind == "PrepareAndFinalizeWithAllGather":
        if not dp_sizes or any(type(n) is not int or n < 0 for n in dp_sizes) or dp_rank >= len(dp_sizes):
            raise IntegrityError("DP token metadata required")
        if dp_sizes[dp_rank] != input_size:
            raise IntegrityError("DP metadata differs from actual forward size")
        if sequence_parallel:
            if ep_sizes is not None:
                if (len(ep_sizes) != len(dp_sizes) * tp_size
                        or any(type(n) is not int or n < 0 for n in ep_sizes)):
                    raise IntegrityError("unsupported EP chunk metadata")
                for dp, n in enumerate(dp_sizes):
                    chunk = (n + tp_size - 1) // tp_size
                    sizes = ep_sizes[dp * tp_size:(dp + 1) * tp_size]
                    if (any(s != chunk for s in sizes[:-1])
                            or sizes[-1] not in {n - chunk * (tp_size - 1), chunk}):
                        raise IntegrityError("EP chunks do not preserve SP row ordering")
                expected = sum(ep_sizes)
                offset = sum(ep_sizes[:dp_rank * tp_size])
            else:
                expected = sum(dp_sizes)
                offset = sum(dp_sizes[:dp_rank])
        else:
            expected = max(dp_sizes) * len(dp_sizes)
            offset = dp_rank * max(dp_sizes)
        if output_size != expected:
            raise IntegrityError("AllGather selector rows differ from declared DP/EP layout")
        return [(i, offset + i) for i in indices] if tp_rank == 0 else []
    raise ConfigError(f"unsupported prepare path: {kind}")
