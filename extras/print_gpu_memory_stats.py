import torch


def print_gpu_memory_stats():
    if not torch.cuda.is_available():
        print("No GPU available")
        return

    for i in range(torch.cuda.device_count()):
        device_name = torch.cuda.get_device_name(i)
        # Convert to GB
        total_memory = torch.cuda.get_device_properties(i).total_memory / (1024
                                                                           **3)
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        reserved = torch.cuda.memory_reserved(i) / (1024**3)
        max_allocated = torch.cuda.max_memory_allocated(i) / (1024**3)
        max_reserved = torch.cuda.max_memory_reserved(i) / (1024**3)
        free_memory = reserved - allocated

        print(f"Device {i}: {device_name}")
        print(f"  Total Memory:       {total_memory:.2f} GB")
        print(f"  Allocated Memory:   {allocated:.2f} GB")
        print(f"  Reserved Memory:    {reserved:.2f} GB")
        print(f"  Free Memory:        {free_memory:.2f} GB")
        print(f"  Max Allocated:      {max_allocated:.2f} GB")
        print(f"  Max Reserved:       {max_reserved:.2f} GB")
        print("-" * 40)


print_gpu_memory_stats()
