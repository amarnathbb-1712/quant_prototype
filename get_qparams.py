import torch
import torch.nn as nn
from enums import MappingEnum, GranularityEnum

def compute_qparams(x_min, x_max, qmin, qmax, mapping_type: MappingEnum):
    eps = 1e-8
    x_min, x_max = float(x_min), float(x_max)

    if mapping_type == MappingEnum.SYMMETRIC:
        max_abs = max(abs(x_min), abs(x_max))
        scale = max_abs / ((qmax - qmin) / 2)
        zero_point = 0

    elif mapping_type == MappingEnum.ASYMMETRIC:
        scale = (x_max - x_min) / (qmax - qmin + eps)
        zero_point = qmin - round(x_min / scale)
        zero_point = int(torch.clamp(torch.tensor(zero_point), qmin, qmax).item())

    else:
        raise ValueError(f"Unsupported mapping type: {mapping_type}")

    return scale, zero_point

def get_qparams(weight_tensor, qmin, qmax, mapping_type, granularity):
    scales = []
    zero_points = []

    if granularity == GranularityEnum.PER_ROW:
        x_min_vals = weight_tensor.min(dim=1).values
        x_max_vals = weight_tensor.max(dim=1).values
        for i in range(weight_tensor.shape[0]):
            scale, zero_point = compute_qparams(
                x_min_vals[i].item(),
                x_max_vals[i].item(),
                qmin, qmax,
                mapping_type
            )
            scales.append(scale)
            zero_points.append(zero_point)

    # elif granularity == Granularity.PER_COLUMN:
    #     x_min_vals = weight_tensor.min(dim=0).values
    #     x_max_vals = weight_tensor.max(dim=0).values
    #     for i in range(weight_tensor.shape[1]):
    #         scale, zero_point = compute_qparams(
    #             x_min_vals[i].item(),
    #             x_max_vals[i].item(),
    #             qmin, qmax,
    #             mapping_type
    #         )
    #         scales.append(scale)
    #         zero_points.append(zero_point)

    elif granularity == GranularityEnum.PER_TENSOR:
        x_min = weight_tensor.min().item()
        x_max = weight_tensor.max().item()
        scale, zero_point = compute_qparams(
            x_min,
            x_max,
            qmin, qmax,
            mapping_type
        )
        scales.append(scale)
        zero_points.append(zero_point)

    else:
        raise ValueError(f"Unsupported granularity: {granularity}")

    return scales, zero_points

layer = nn.Linear(3, 6)
weight_tensor = layer.weight  # shape (6,3)

qmin, qmax = -8, 7
mapping_type = MappingEnum.SYMMETRIC
granularity = GranularityEnum.PER_ROW

scales, zero_points = get_qparams(
    weight_tensor, qmin, qmax, mapping_type, granularity
)

print("Scales:", scales)
print("Zero points:", zero_points)
