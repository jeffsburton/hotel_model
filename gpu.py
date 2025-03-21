import torch

# Check PyTorch version
print("PyTorch version:", torch.__version__)

# Check if CUDA (GPU support) is available
print("\nIs CUDA available:", torch.cuda.is_available())

# Number of GPUs available
print("\nNumber of GPUs Available:", torch.cuda.device_count())

# List GPU details (if available)
if torch.cuda.is_available() :
    print("\nGPU Details:")
    for i in range(torch.cuda.device_count()):
        print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("\nNo GPUs found.")

# Set device (use GPU if available, otherwise fallback to CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nDevice selected for computation:", device)

# Tensor operations (simple matrix multiplication)
a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device)
b = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device=device)
c = torch.matmul(a, b)  # Perform matrix multiplication on the selected device
print("\nTensor Operation Result:")
print(c)

# Get device memory details
if torch.cuda.is_available():
    print("\nMemory Details:")
    print(f"  Allocated memory: {torch.cuda.memory_allocated()} bytes")
    print(f"  Cached memory: {torch.cuda.memory_reserved()} bytes")
else:
    print("\nGPU memory information not available or no GPU found.")
