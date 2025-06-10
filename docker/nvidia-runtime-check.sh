#!/bin/bash

# NVIDIA Runtime Check Script
# Validates NVIDIA Docker runtime availability and GPU accessibility

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REQUIRED_CUDA_VERSION="11.8"
REQUIRED_NVIDIA_DRIVER="520.0"
REQUIRED_GPU_MEMORY_GB=20

echo -e "${BLUE}=== NVIDIA Runtime & GPU Accessibility Check ===${NC}"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo

# Function to print status
print_status() {
    local status=$1
    local message=$2
    if [ "$status" = "PASS" ]; then
        echo -e "${GREEN}✓ PASS${NC}: $message"
    elif [ "$status" = "WARN" ]; then
        echo -e "${YELLOW}⚠ WARN${NC}: $message"
    else
        echo -e "${RED}✗ FAIL${NC}: $message"
    fi
}

# Check if running in Docker container
check_docker_environment() {
    echo -e "${BLUE}1. Docker Environment Check${NC}"
    if [ -f /.dockerenv ]; then
        print_status "PASS" "Running inside Docker container"
        return 0
    else
        print_status "WARN" "Not running in Docker container"
        return 1
    fi
}

# Check NVIDIA Docker runtime
check_nvidia_docker() {
    echo -e "${BLUE}2. NVIDIA Docker Runtime Check${NC}"
    
    # Check if nvidia-smi is available
    if command -v nvidia-smi &> /dev/null; then
        print_status "PASS" "nvidia-smi command available"
    else
        print_status "FAIL" "nvidia-smi not found - NVIDIA drivers not installed"
        return 1
    fi
    
    # Check NVIDIA runtime
    if [ -n "$NVIDIA_VISIBLE_DEVICES" ]; then
        print_status "PASS" "NVIDIA_VISIBLE_DEVICES set: $NVIDIA_VISIBLE_DEVICES"
    else
        print_status "FAIL" "NVIDIA_VISIBLE_DEVICES not set"
        return 1
    fi
    
    return 0
}

# Check GPU accessibility
check_gpu_access() {
    echo -e "${BLUE}3. GPU Accessibility Check${NC}"
    
    # Run nvidia-smi and capture output
    if nvidia_output=$(nvidia-smi --query-gpu=index,name,memory.total,driver_version,cuda_version --format=csv,noheader,nounits 2>/dev/null); then
        print_status "PASS" "nvidia-smi executed successfully"
        
        # Parse GPU information
        while IFS=',' read -r gpu_index gpu_name memory_total driver_version cuda_version; do
            # Clean up whitespace
            gpu_index=$(echo "$gpu_index" | xargs)
            gpu_name=$(echo "$gpu_name" | xargs)
            memory_total=$(echo "$memory_total" | xargs)
            driver_version=$(echo "$driver_version" | xargs)
            cuda_version=$(echo "$cuda_version" | xargs)
            
            echo "  GPU $gpu_index: $gpu_name"
            echo "    Memory: ${memory_total}MB"
            echo "    Driver: $driver_version"
            echo "    CUDA: $cuda_version"
            
            # Check memory requirements
            memory_gb=$((memory_total / 1024))
            if [ "$memory_gb" -ge "$REQUIRED_GPU_MEMORY_GB" ]; then
                print_status "PASS" "GPU $gpu_index has sufficient memory: ${memory_gb}GB >= ${REQUIRED_GPU_MEMORY_GB}GB"
            else
                print_status "WARN" "GPU $gpu_index may have insufficient memory: ${memory_gb}GB < ${REQUIRED_GPU_MEMORY_GB}GB"
            fi
            
        done <<< "$nvidia_output"
        
    else
        print_status "FAIL" "Failed to execute nvidia-smi"
        return 1
    fi
    
    return 0
}

# Check CUDA installation
check_cuda_installation() {
    echo -e "${BLUE}4. CUDA Installation Check${NC}"
    
    # Check nvcc
    if command -v nvcc &> /dev/null; then
        cuda_version=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9.]*\).*/\1/p')
        print_status "PASS" "CUDA compiler available: $cuda_version"
        
        # Version comparison
        if command -v python3 &> /dev/null; then
            version_check=$(python3 -c "
import sys
current = [int(x) for x in '$cuda_version'.split('.')]
required = [int(x) for x in '$REQUIRED_CUDA_VERSION'.split('.')]
print('ok' if current >= required else 'fail')
")
            if [ "$version_check" = "ok" ]; then
                print_status "PASS" "CUDA version meets requirements: $cuda_version >= $REQUIRED_CUDA_VERSION"
            else
                print_status "WARN" "CUDA version may be insufficient: $cuda_version < $REQUIRED_CUDA_VERSION"
            fi
        fi
    else
        print_status "WARN" "CUDA compiler (nvcc) not found"
    fi
    
    # Check CUDA libraries
    if ldconfig -p | grep -q libcuda; then
        print_status "PASS" "CUDA libraries found in system"
    else
        print_status "WARN" "CUDA libraries not found in ldconfig"
    fi
    
    return 0
}

# Check Python CUDA packages
check_python_cuda() {
    echo -e "${BLUE}5. Python CUDA Packages Check${NC}"
    
    if command -v python3 &> /dev/null; then
        # Check PyTorch CUDA
        if python3 -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}')" 2>/dev/null; then
            if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
                device_count=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
                print_status "PASS" "PyTorch CUDA available with $device_count device(s)"
            else
                print_status "FAIL" "PyTorch CUDA not available"
            fi
        else
            print_status "WARN" "PyTorch not installed or CUDA not available"
        fi
        
        # Check ONNX Runtime GPU
        if python3 -c "import onnxruntime as ort; providers = ort.get_available_providers(); print('CUDAExecutionProvider' in providers)" 2>/dev/null | grep -q True; then
            print_status "PASS" "ONNX Runtime CUDA provider available"
        else
            print_status "WARN" "ONNX Runtime CUDA provider not available"
        fi
    else
        print_status "FAIL" "Python3 not available"
        return 1
    fi
    
    return 0
}

# GPU memory stress test
gpu_memory_test() {
    echo -e "${BLUE}6. GPU Memory Allocation Test${NC}"
    
    if command -v python3 &> /dev/null; then
        # Simple GPU memory allocation test
        memory_test_result=$(python3 -c "
import torch
try:
    if torch.cuda.is_available():
        # Allocate 1GB tensor
        x = torch.randn(256, 1024, 1024, device='cuda')
        print(f'Successfully allocated 1GB on GPU: {x.device}')
        del x
        torch.cuda.empty_cache()
        print('Memory freed successfully')
        print('PASS')
    else:
        print('CUDA not available')
        print('FAIL')
except Exception as e:
    print(f'Error: {e}')
    print('FAIL')
" 2>/dev/null)
        
        if echo "$memory_test_result" | grep -q "PASS"; then
            print_status "PASS" "GPU memory allocation test successful"
        else
            print_status "FAIL" "GPU memory allocation test failed"
            echo "$memory_test_result"
        fi
    else
        print_status "SKIP" "Python3 not available for memory test"
    fi
}

# Main execution
main() {
    local exit_code=0
    
    check_docker_environment || exit_code=1
    check_nvidia_docker || exit_code=1
    check_gpu_access || exit_code=1
    check_cuda_installation || exit_code=1
    check_python_cuda || exit_code=1
    gpu_memory_test || exit_code=1
    
    echo
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}=== All checks passed! NVIDIA runtime is ready for Avatar Service ===${NC}"
    else
        echo -e "${RED}=== Some checks failed. Please resolve issues before running Avatar Service ===${NC}"
    fi
    
    echo
    echo "Next steps:"
    echo "1. If all checks passed, you can start the Avatar Service"
    echo "2. If checks failed, install missing components or fix configuration"
    echo "3. Re-run this script to verify fixes"
    
    return $exit_code
}

# Run main function
main "$@" 