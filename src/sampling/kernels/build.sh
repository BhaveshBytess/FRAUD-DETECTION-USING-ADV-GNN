#!/bin/bash
# src/sampling/kernels/build.sh
# Build script for CUDA kernels per §PHASE_B compilation

set -e

echo "Building G-SAMPLER CUDA kernels..."

# Check for CUDA installation
if ! command -v nvcc &> /dev/null; then
    echo "CUDA compiler (nvcc) not found. Please install CUDA toolkit."
    exit 1
fi

# Build configuration
KERNEL_SRC="kernels.cu"
OUTPUT_LIB="libgsampler_kernels.so"
COMPUTE_CAPABILITY="75"  # RTX 20xx/30xx series, adjust as needed

# Compiler flags
NVCC_FLAGS="-O3 -Xcompiler -fPIC -shared"
CUDA_ARCH="-arch=sm_${COMPUTE_CAPABILITY}"
INCLUDE_DIRS="-I/usr/local/cuda/include"

# Build the shared library
echo "Compiling CUDA kernels with compute capability ${COMPUTE_CAPABILITY}..."
nvcc ${NVCC_FLAGS} ${CUDA_ARCH} ${INCLUDE_DIRS} -o ${OUTPUT_LIB} ${KERNEL_SRC}

if [ $? -eq 0 ]; then
    echo "✓ CUDA kernels compiled successfully: ${OUTPUT_LIB}"
    
    # Check library symbols
    echo "Library symbols:"
    nm -D ${OUTPUT_LIB} | grep -E "(launch_|time_window|sample_k|compact|parallel_map)" || echo "No kernel symbols found"
    
    # Check CUDA runtime
    echo "CUDA runtime info:"
    nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"
    
else
    echo "✗ CUDA kernel compilation failed"
    exit 1
fi

echo "G-SAMPLER CUDA kernels build complete!"
