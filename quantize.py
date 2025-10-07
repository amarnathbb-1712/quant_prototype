import torch
from enums import MappingEnum, GranularityEnum
from get_qparams import get_qparams

def quantize(
    self,
    weights: torch.Tensor,
    granularity: GranularityEnum = GranularityEnum.PER_ROW,
    mapping_type: MappingEnum = MappingEnum.SYMMETRIC,
):
    if weights is None:
        raise ValueError("Expected weights tensor for quantization.")

    # TODO use dtype and mappingenum to derive quantization_range
    # Define quantization range based on mapping type
    if mapping_type == MappingEnum.SYMMETRIC:
        qmin, qmax = -128, 127
    elif mapping_type == MappingEnum.ASYMMETRIC:
        qmin, qmax = 0, 255
    else:
        raise ValueError(f"Unsupported mapping type: {mapping_type}")

    # Compute quantization parameters
    scales_list, zero_points_list = get_qparams(weights, qmin, qmax, mapping_type, granularity)

    # Convert to tensors based on granularity
    if granularity == GranularityEnum.PER_ROW:
        scales_tensor = torch.tensor(scales_list, dtype=torch.float32).unsqueeze(1)
        zero_points_tensor = torch.tensor(zero_points_list, dtype=torch.float32).unsqueeze(1)
    elif granularity == GranularityEnum.PER_TENSOR:
        scales_tensor = torch.tensor(scales_list[0], dtype=torch.float32)
        zero_points_tensor = torch.tensor(zero_points_list[0], dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported granularity: {granularity}")

    # Apply quantization
    if mapping_type == MappingEnum.SYMMETRIC:
        int8_weights = torch.round(weights / scales_tensor).clamp(qmin, qmax).to(torch.int8)
    elif mapping_type == MappingEnum.ASYMMETRIC:
        int8_weights = torch.round((weights - zero_points_tensor) / scales_tensor).clamp(qmin, qmax).to(torch.uint8)

    # Store quantized artifacts in module
    self.int8_weights = int8_weights
    self.scales = scales_tensor
    self.zero_points = zero_points_tensor