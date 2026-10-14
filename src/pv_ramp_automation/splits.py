from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChronologicalSplit:
    train_start: int
    train_end: int
    validation_start: int
    validation_end: int
    test_start: int
    test_end: int


def chronological_split_indices(n_rows: int, train_fraction: float = 0.70, validation_fraction: float = 0.15) -> ChronologicalSplit:
    if n_rows < 3:
        raise ValueError("Need at least 3 rows for a train/validation/test split.")
    train_end = int(n_rows * train_fraction)
    validation_end = train_end + int(n_rows * validation_fraction)
    train_end = max(1, min(train_end, n_rows - 2))
    validation_end = max(train_end + 1, min(validation_end, n_rows - 1))
    return ChronologicalSplit(
        train_start=0,
        train_end=train_end,
        validation_start=train_end,
        validation_end=validation_end,
        test_start=validation_end,
        test_end=n_rows,
    )
