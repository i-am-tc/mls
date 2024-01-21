import torch
import time

# Function to check for a dedicated GPU
def has_dedicated_gpu():
    for i in range(torch.cuda.device_count()):
        gpu_properties = torch.cuda.get_device_properties(i)
        if gpu_properties.total_memory > 4e9:  # Assuming dedicated GPUs have more than 4GB of memory
            print(gpu_properties)
            return True
    return False

# Check for a dedicated GPU
if not has_dedicated_gpu():
    print("No dedicated GPU found.")
else:
    print("Dedicated GPU found, starting stress test.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    size = (10000, 10000)  # Size of the tensors (modify if needed)

    tensor1 = torch.randn(size, device=device)
    tensor2 = torch.randn(size, device=device)

    with torch.no_grad():  # Warm-up run
        warmup = torch.matmul(tensor1, tensor2)

    start_time = time.time()

    with torch.no_grad():
        for _ in range(999999):  # Repeat to increase stress on the GPU
            result = torch.matmul(tensor1, tensor2)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time for 10 matrix multiplications: {elapsed_time:.2f} seconds")
