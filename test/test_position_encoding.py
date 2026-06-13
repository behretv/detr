"""Position-encoding model tests."""

import torch

from detr.models.position_encoding import (
    PositionEmbeddingLearned,
    PositionEmbeddingSine,
)


def test_position_encoding_script():
    m1, m2 = PositionEmbeddingSine(), PositionEmbeddingLearned()
    torch.jit.script(m1), torch.jit.script(m2)  # noqa
