from enums import MappingEnum, GranularityEnum
from get_qparams import get_qparams_
import torch
import torch.nn.functional as F

def int8_forward(
    weight: torch.Tensor,
    input: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Performs a forward pass for a quantized linear layer using int8 weights.

    Args:
        weight (torch.Tensor): Quantized weight tensor of shape (output_featureures, input_featureures), dtype int8.
        input (torch.Tensor): Input tensor of shape (batch_size, input_featureures), typically float32 or float16.
        scales (torch.Tensor): Scale tensor of shape (output_featureures,) used to dequantize the weights.
        bias (torch.Tensor | None, optional): Optional bias tensor of shape (1, output_featureures). Defaults to None.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, output_featureures) after applying the linear transformation
                      and scaling. Same dtype as `input`.
    """
    # Cast int8 weights to input's dtype for computation
    casted_weights = weight.to(input.dtype)
    # Linear transformation followed by scaling
    output = F.linear(input, casted_weights) * scales

    if bias is not None:
        output = output + bias

    return output


class QuantizedLinear(torch.nn.Module):
    """
    A linear layer module with quantized int8 weights.

    Attributes:
        input_feature (int): Number of input features.
        output_feature (int): Number of output features.
        bias (bool): Whether to include a bias term.
        dtype (torch.dtype): Data type for bias and computation.
        int8_weights (torch.Tensor): Stored int8 weight tensor.
        scales (torch.Tensor | None): Scale tensor for dequantizing weights.
        zero_points (torch.Tensor | None): Zero-point tensor (for asymmetric quantization, currently unused).
        bias (torch.Tensor | None): Bias tensor if bias=True, else None.
    """

    def __init__(
        self,
        input_feature: int,
        output_feature: int,
        bias: bool = True,
        dtype: torch.dtype = torch.float32
    ):
        """
        Initializes a QuantizedLinear layer.

        Args:
            input_feature (int): Number of input features.
            output_feature (int): Number of output features.
            bias (bool, optional): If True, adds a learnable bias to the output. Defaults to True.
            dtype (torch.dtype, optional): Data type for bias and computation. Defaults to torch.float32.
        """
        super().__init__()
        self.input_feature = input_feature
        self.output_feature = output_feature
        self.bias = bias
        self.dtype = dtype

        # Initialize int8 weights randomly
        self.register_buffer(
            "int8_weights",
            torch.randint(-128, 127, (output_feature, input_feature), dtype=torch.int8)
        )

        # Scale and zero-point buffers for quantization
        self.register_buffer("scales", None)
        self.register_buffer("zero_points", None)

        # Optional bias
        if bias:
            self.register_buffer("bias", torch.randn((1, output_feature), dtype=dtype))
        else:
            self.bias = None

    def forward(self, input):
        return int8(self.int8_weights,input,self.scales, self.bias)