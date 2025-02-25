import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal


def nki_matmul_tiled_(lhsT, rhs, result):
  """NKI kernel to compute a matrix multiplication operation in a tiled manner"""

  K, M = lhsT.shape
  K_, N = rhs.shape
  assert K == K_, "lhsT and rhs must have the same contraction dimension"

  # Maximum free dimension of the stationary operand of general matrix multiplication on tensor engine
  TILE_M = nl.tile_size.gemm_stationary_fmax  # 128

  # Maximum partition dimension of a tile
  TILE_K = nl.tile_size.pmax  # 128

  # Maximum free dimension of the moving operand of general matrix multiplication on tensor engine
  TILE_N = nl.tile_size.gemm_moving_fmax  # 512

  # Use affine_range to loop over tiles
  for m in nl.affine_range(M // TILE_M):
    for n in nl.affine_range(N // TILE_N):
      # Allocate a tensor in PSUM
      res_psum = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum)

      for k in nl.affine_range(K // TILE_K):
        # Declare the tiles on SBUF
        lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
        rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)

        # Load tiles from lhsT and rhs
        lhsT_tile[...] = nl.load(lhsT[k * TILE_K:(k + 1) * TILE_K,
                                      m * TILE_M:(m + 1) * TILE_M])
        rhs_tile[...] = nl.load(rhs[k * TILE_K:(k + 1) * TILE_K,
                                    n * TILE_N:(n + 1) * TILE_N])

        # Accumulate partial-sums into PSUM
        res_psum += nl.matmul(lhsT_tile[...], rhs_tile[...], transpose_x=True)

      # Copy the result from PSUM back to SBUF, and cast to expected output data-type
      res_sb = nl.copy(res_psum, dtype=result.dtype)
      nl.store(result[m * TILE_M:(m + 1) * TILE_M, n * TILE_N:(n + 1) * TILE_N],
               value=res_sb)

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
    #resize weights
    W = W.reshape((out_channels // 128, 128, in_channels_, filter_height, filter_width))

    W_sbuf = nl.ndarray(
            shape=(out_channels // 128, nl.par_dim(128), in_channels_, filter_height, filter_width),
            dtype=W.dtype,
            buffer=nl.sbuf
        )
    w = nl.ndarray(
        shape=(filter_height, filter_width, out_channels // 128, nl.par_dim(128), in_channels_),
        dtype=W.dtype,
        buffer=nl.sbuf
    )

    for k in nl.affine_range(out_channels // 128):
        W_sbuf[k] = nl.load(W[k])

    for i in nl.affine_range(filter_height):
            for j in nl.affine_range(filter_width):
                for k in nl.affine_range(out_channels // 128):
                    w[i,j,k,:,:] = nl.copy(W_sbuf[k,:,:,i,j])

    # Various tiling dimensions (You may want to define more of them)
    # K = in_channels   
    max_in_channels = nl.tile_size.pmax
    num_in_channel_tiles = in_channels // max_in_channels

    # M = out_channels
    max_out_channels = nl.tile_size.gemm_stationary_fmax
    num_out_channel_tiles = out_channels // max_out_channels


    # N = out_width
    print('Reorganized, beginning multiplication')

    # Process the images in batches
    for b in nl.affine_range(batch_size):
        for h in nl.affine_range(out_height):
            for k in nl.affine_range(out_channels // 128):
                start_idx_out_channel = k * 128
                end_idx_out_channel = (k + 1) * 128

                res = nl.zeros((max_out_channels, out_width), nl.float32, buffer=nl.psum)
                # tiling dims
                for m in nl.affine_range(num_in_channel_tiles): 
                    for i in nl.affine_range(filter_height):
                        for j in nl.affine_range(filter_width):
                            image_slice = nl.ndarray((max_in_channels, out_width), dtype=X.dtype, buffer=nl.sbuf)
        
                            start_idx_in_channel = m * max_in_channels
                            end_idx_in_channel = (m + 1) * max_in_channels

                            image_slice[...] = nl.load(X[b, start_idx_in_channel:end_idx_in_channel, h + i, j:j + out_width])

                            res += nl.matmul(w[i, j, k, :, start_idx_in_channel:end_idx_in_channel], image_slice[...])

                res_sb = nl.copy(res, dtype=res.dtype)

                bias_slice = nl.load(bias[start_idx_out_channel:end_idx_out_channel])
                # for t in nl.affine_range(out_width):
                #     res_sb[:, t] = bias_slice

                final = nl.add(res_sb, bias_slice)

                nl.store(X_out[b, start_idx_out_channel:end_idx_out_channel, h, :], value=final)

    return X_out

"""
optimized vector addition example
def vector_add_tiled(a_vec, b_vec):
    
    # Allocate space for the output vector in HBM
    out = nl.ndarray(shape=a_vec.shape, dtype=a_vec.dtype, buffer=nl.hbm)

    # Get the total number of vector rows
    M = a_vec.shape[0]
    
    ROW_CHUNK = 1

    # Loop over the total number of chunks, we can use affine_range
    # because there are no loop-carried dependencies
    for m in nl.affine_range((M // ROW_CHUNK)):

        # Allocate row-chunk sized tiles for the input vectors
        a_tile = nl.ndarray((ROW_CHUNK, 1), dtype=a_vec.dtype, buffer=nl.sbuf)
        b_tile = nl.ndarray((ROW_CHUNK, 1), dtype=b_vec.dtype, buffer=nl.sbuf)
        
        # Load a chunk of rows
        a_tile[...] = nl.load(a_vec[m * ROW_CHUNK : (m + 1) * ROW_CHUNK])
        b_tile[...] = nl.load(b_vec[m * ROW_CHUNK : (m + 1) * ROW_CHUNK])

        # Add the row chunks together
        res = nl.add(a_tile, b_tile)

        # Store the result chunk into HBM
        nl.store(out[m * ROW_CHUNK : (m + 1) * ROW_CHUNK], value=res)
    
    return out

vector addition example
# Allocate space for the output vector in HBM
    # it is the same dimensions as a_vec, with the same element type as a_vec
    out = nl.ndarray(shape=a_vec.shape, dtype=a_vec.dtype, buffer=nl.hbm)

    # Load the input vectors from HBM into variables stored in SBUF 
    a = nl.load(a_vec)
    b = nl.load(b_vec)

    # Add the input vectors
    res = nl.add(a, b)

    # Store the result into HBM
    nl.store(out, value=res)
"""
