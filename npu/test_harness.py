import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal
from neuronxcc.nki import benchmark

from conv_npu import conv2d

from conv_numpy import conv2d_cpu_torch
import logging
import argparse
import io
import sys

import subprocess

logging.disable(logging.OFF)


def save_trace(profile_name, neff_file_name):
    """Run the profiler and save the NEFF and NTFF files with the specified name."""
    subprocess.run(
        [
            "neuron-profile",
            "capture",
            "-n",
            neff_file_name,
            "-s",
            profile_name + ".ntff",
        ],
        check=True,
    )

    subprocess.run(["mv", neff_file_name, profile_name + ".neff"], check=True)

    print(
        f"\n\nNEFF / NTFF files generated with names: {profile_name + '.neff'}, {profile_name + '.ntff'}"
    )


def test_correctness_conv2d_kernel(
    kernel,
    use_cpu_impl=False,
    use_larger_images=False,
    use_bias=False,
):
    ref_impl = conv2d_cpu_torch

    input_channels_list = [128, 256]
    output_channels_list = [128, 256]
    kernel_size_list = [3]
    batch_size_list = [4]
    image_dims_list = [(32, 16)]

    if use_larger_images:
        input_channels_list = [256]
        output_channels_list = [256]
        image_dims_list = [(224, 224)]

    for input_channels in input_channels_list:
        for output_channels in output_channels_list:
            for kernel_size in kernel_size_list:
                for batch_size in batch_size_list:
                    for image_dims in image_dims_list:
                        X = np.random.rand(
                            batch_size, input_channels, image_dims[0], image_dims[1]
                        ).astype(np.float32)
                        W = np.random.rand(
                            output_channels, input_channels, kernel_size, kernel_size
                        ).astype(np.float32)
                        bias = (
                            np.zeros(output_channels).astype(np.float32)
                            if not use_bias
                            else np.random.rand(output_channels).astype(np.float32)
                        )

                        args = [X, W, bias]

                        out = kernel(*args)
                        out_ref = ref_impl(*args)

                        #print(out)
                        #print(out_ref)

                        if not np.allclose(out, out_ref):
                            print(
                                f"Output mismatch for input_channels: {input_channels}, \
                        output_channels: {output_channels}, kernel_size: {kernel_size}, batch_size: {batch_size},\
                         image_dims: {image_dims}, use_bias: {use_bias}"
                            )
                            return False

    return True


def test_performance_conv2d_kernel(
    kernel,
    dtype=np.float32,
    batch_size=1,
    in_channels=256,
    out_channels=256,
    image_height=224,
    image_width=224,
    kernel_height=3,
    kernel_width=3,
):

    performance_requirements_by_dtype = {
        np.float32: 4300,
        np.float16: 1300
    }

    X = np.random.rand(batch_size, in_channels, image_height, image_width).astype(dtype)
    W = np.random.rand(out_channels, in_channels, kernel_height, kernel_width).astype(
        dtype
    )
    bias = np.random.rand(out_channels).astype(dtype)

    args = [X, W, bias]

    dtype_str = "float32" if dtype == np.float32 else "float16"

    bench_func = nki.benchmark(
        warmup=5, iters=20, save_neff_name=f"file_{dtype_str}.neff"
    )(kernel)
    text_trap = io.StringIO()
    sys.stdout = text_trap
    bench_func(*args)
    sys.stdout = sys.__stdout__
    p99_us_student = bench_func.benchmark_result.nc_latency.get_latency_percentile(99)
    print(f"\n\nExecution Time for student implementation: {p99_us_student} μs")

    if p99_us_student > performance_requirements_by_dtype[dtype]:
        print(f"Performance requirement not met: need to be under {performance_requirements_by_dtype[dtype]} μs")
        return False

    return True

def test_small_conv2d(kernel):
    """
    Test conv2d with small inputs for easier debugging
    """
    # Small input dimensions
    batch_size = 1
    in_channels = 128  # Keep multiple of 128 as per requirement
    input_height = 4
    input_width = 4
    
    # Filter dimensions
    out_channels = 128  # Keep multiple of 128 as per requirement
    filter_height = 2
    filter_width = 2

    # Create input tensor with a simple pattern
    X = np.array([
        [  # One batch
            [  # First channel (others will be similar)
                [1, 0, 1, 0],
                [2, 1, 2, 1],
                [1, 0, 1, 0],
                [2, 1, 2, 2]
            ]
        ] * in_channels  # Repeat for all input channels
    ], dtype=np.float32).transpose(0, 1, 2, 3)

    # Create weight tensor - simple pattern for verification
    W = np.array([
        [  # First output channel (others will be similar)
            [  # First input channel (others will be similar)
                [0, 1],
                [0.2, 0]
            ]
        ] * in_channels # Repeat for all input channels
    ] * out_channels, dtype=np.float32)

    # Create bias - simple pattern
    bias = np.zeros(out_channels, dtype=np.float32)

    # Run your kernel
    out = kernel(X, W, bias)
    
    # Run reference implementation
    out_ref = conv2d_cpu_torch(X, W, bias)
    
    # Print shapes for verification
    print("Input shape:", X.shape)
    print("Weight shape:", W.shape)
    print("Output shape:", out.shape)
    
    # Print first channel of first batch for verification
    print("\nFirst channel of input:")
    print(X[0, 0])
    print("\nFirst channel of output:")
    print(out[0, 0])
    print("\nFirst channel of reference output:")
    print(out_ref[0, 0])

    print("\nLast channel of input:")
    print(X[0, -1])
    print("\nLast channel of output:")
    print(out[0, -1])
    print("\nLast channel of reference output:")
    print(out_ref[0, -1])
    
    # Check if output matches reference
    outputs_match = np.allclose(out, out_ref, rtol=1e-3, atol=1e-3)
    print("\nOutputs match reference:", outputs_match)

    return outputs_match

# write a function g which when passed a function f, returns a new function that when called with some *args and **kwargs, calls
# nki.simulate_kernel(f, *args, **kwargs) and returns the result
def simulate_kernel_wrapper(kernel):
    def temp_func(*args, **kwargs):
        return nki.simulate_kernel(kernel, *args, **kwargs)

    return temp_func


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--profile", type=str, default=None, help="File to save the neff file"
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Use nki.simulate_kernel to run student implementation",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for random number generation"
    )

    args = parser.parse_args()

    np.random.seed(args.seed)

    if args.simulate:
        conv2d = simulate_kernel_wrapper(conv2d)
    
    # print("Running small test for conv2d kernel...", end="", flush=True)
    # small_test_result = test_small_conv2d(conv2d)
    # if small_test_result:
    #     print("Small test passed! 🎉")
    # else:
    #     print("Small test failed 😢")
    #     sys.exit(1)


    # running correctness tests
    print(
        "Running correctness test for conv2d kernel with smaller images...",
        end="",
        flush=True,
    )
    test_result = test_correctness_conv2d_kernel(conv2d, use_larger_images=False)
    if test_result:
        print("Passed 😎")
    else:
        print("Failed 😢")
        sys.exit(1)

    print(
        "Running correctness test for conv2d kernel with larger images...",
        end="",
        flush=True,
    )
    test_result = test_correctness_conv2d_kernel(conv2d, use_larger_images=True)
    if test_result:
        print("Passed 😇")
    else:
        print("Failed 😢")

    print(
        "Running correctness test for conv2d kernel with larger images + bias...",
        end="",
        flush=True,
    )
    test_result = test_correctness_conv2d_kernel(
        conv2d, use_bias=True, use_larger_images=True
    )
    if test_result:
        print("Passed 😍")
    else:
        print("Failed 😢")

    print("Comparing performance with reference kernel (float32)...")
    test_result = test_performance_conv2d_kernel(conv2d, dtype = np.float32)
    if test_result:
        print("Performance test passed 😍")
    else:
        print("Performance test failed 😢")

    if args.profile is not None:
        save_trace(args.profile + "_float32", "file_float32.neff")
    
    print("Comparing performance with reference kernel (float16)...")
    test_result = test_performance_conv2d_kernel(conv2d, dtype = np.float16)
    if test_result:
        print("Performance test passed 😍")
    else:
        print("Performance test failed 😢")

    if args.profile is not None:
        save_trace(args.profile + "_float16", "file_float16.neff")
