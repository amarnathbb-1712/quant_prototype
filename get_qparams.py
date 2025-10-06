import torch
import torch.nn as nn
from enum import Enum, auto

# --- Mapping type enum ---
class MappingType(Enum):
    SYMMETRIC = auto()
    SYMMETRIC_NO_CLIPPING_ERR = auto()
    ASYMMETRIC = auto()

# --- Function to compute qparams ---
def get_qparams(x_min, x_max, qmin, qmax, mapping_type: MappingType):
    """Compute scale and zero_point based on mapping type."""
    eps = 1e-8  # to prevent division-by-zero
    x_min, x_max = float(x_min), float(x_max)

    if mapping_type == MappingType.SYMMETRIC:
        max_abs = max(abs(x_min), abs(x_max))
        scale = max_abs / ((qmax - qmin) / 2)
        zero_point = 0

    elif mapping_type == MappingType.SYMMETRIC_NO_CLIPPING_ERR:
        smin = abs(x_min) / abs(qmin) if qmin != 0 else 0
        smax = abs(x_max) / abs(qmax) if qmax != 0 else 0
        scale = max(smin, smax) + eps
        zero_point = 0

    elif mapping_type == MappingType.ASYMMETRIC:
        scale = (x_max - x_min) / (qmax - qmin + eps)
        zero_point = qmin - round(x_min / scale)
        zero_point = int(torch.clamp(torch.tensor(zero_point), qmin, qmax).item())

    else:
        raise ValueError(f"Unsupported mapping type: {mapping_type}")

    return scale, zero_point

# --- Example Linear layer ---
layer = nn.Linear(3, 6)  # 6 outputs, 3 inputs
weight_tensor = layer.weight  # shape: (6, 3)

# --- Compute per-row min and max ---
x_min_rows = weight_tensor.min(dim=1).values  # shape: (6,)
x_max_rows = weight_tensor.max(dim=1).values  # shape: (6,)
print("Per-row min:", x_min_rows)
print("Per-row max:", x_max_rows)

# --- Compute per-row scales and zero-points ---
scales = []
zero_points = []

for i in range(weight_tensor.shape[0]):  # iterate over rows
    scale, zero_point = get_qparams(
        x_min_rows[i].item(),
        x_max_rows[i].item(),
        qmin=-8,
        qmax=7,
        mapping_type=MappingType.SYMMETRIC
    )
    scales.append(scale)
    zero_points.append(zero_point)

print("Per-row scales:", scales)
print("Per-row zero-points:", zero_points)
