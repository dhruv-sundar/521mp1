import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal


"""
A convolution kernel that you need to implement.

Parameters:
    X: the input tensor
    W: the weights of the convolution filters.
    bias: the biases of the convolution filters.

expect: X.shape == [batch_size, in_channels, input_height, input_width]
expect: W.shape == [out_channels, in_channels, filter_height, filter_width]
expect: bias.shape == [out_channels]
expect: filter_height == filter_width
expect: input_channels % 128 == 0
expect: output_channels % 128 == 0

out_height = input_height - filter_height + 1
out_width = input_width - filter_width + 1

out_pool_height = out_height
out_pool_width = out_width

The shape of the output should be [batch_size, out_channels, out_pool_height, out_pool_width]

"""

@nki.jit
def conv2d(X, W, bias):

    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]

    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    out_pool_height = out_height
    out_pool_width = out_width
    
    # Can assume multiple of 128 to avoid using mask
    assert in_channels % 128 == 0

    # Can assume one PSUM bank can at least fit one row of the pixels
    assert nl.tile_size.gemm_moving_fmax >= out_width

    # Initialize output array
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    # Various tiling dimensions (You may want to define more of them)
    c_in_pmax = nl.tile_size.pmax
    c_out_pmax = nl.tile_size.pmax
    n_tiles_c_in = in_channels // c_in_pmax
    n_tiles_c_out = out_channels // c_out_pmax

    # Initialize output array
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    # Tiling constants for matrix multiplication
    TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
    TILE_N = nl.tile_size.gemm_moving_fmax      # 512
    TILE_K = nl.tile_size.pmax                  # 128

    # Process each image in the batch
    for b in nl.affine_range(batch_size):
        # Process output channels in tiles
        for c_out_start in nl.affine_range(0, out_channels, TILE_M):
            # Initialize accumulator for this output tile
            acc = nl.zeros(
                (TILE_M, out_height * out_width),
                dtype=X.dtype,
                buffer=nl.psum
            )
            
            # Iterate over filter positions
            for h in nl.affine_range(filter_height):
                for w in nl.affine_range(filter_width):
                    # Process input channels in tiles
                    for c_in_start in nl.affine_range(0, in_channels, TILE_K):
                        # Extract input window for current position
                        input_window = X[b, 
                                       c_in_start:c_in_start + TILE_K,
                                       h:h + out_height,
                                       w:w + out_width]
                        
                        # Reshape input window for matrix multiplication
                        input_reshaped = np.reshape(input_window, 
                            (TILE_K, out_height * out_width))

                        # Get corresponding filter weights
                        weights = W[c_out_start:c_out_start + TILE_M,
                                  c_in_start:c_in_start + TILE_K,
                                  h, w]
                        
                        # Perform matrix multiplication and accumulate
                        nl.matmul(weights, input_reshaped, acc_out=acc)

            # Add bias after all accumulations for this output tile
            bias_slice = bias[c_out_start:c_out_start + TILE_M]
            bias_reshaped = np.reshape(bias_slice, (TILE_M, 1))
            
            # Add bias and reshape result
            result = np.reshape(acc + bias_reshaped, 
                (TILE_M, out_height, out_width))
            
            # Store result to output
            X_out[b, c_out_start:c_out_start + TILE_M] = result

    return X_out
