import torch
import torch.nn.functional as F

def same_shape(shape1, shape2):
    return torch.allclose(shape1, shape2, atol=1e-08)

def shape_diff(shape1, shape2):
    total_size = shape1.numel()
    return torch.sum(torch.abs(shape2 - shape1)) / total_size

def one_hot_encode(index, size):
    if not isinstance(index, torch.Tensor):
        index = torch.tensor(index)
        
    return F.one_hot(index, num_classes=size).float()

def one_hot_to_index(one_hot_tensor):
    return torch.argmax(one_hot_tensor, dim=-1)

def extract_patches(image, patch_size=10):
    patches = []
    for i in range(0, image.shape[2], patch_size):
        for j in range(0, image.shape[3], patch_size):
            patch = image[:, :, i:i + patch_size, j:j + patch_size]
            patches.append(patch)
    return torch.stack(patches, dim=1)

def get_device():
    if not torch.cuda.is_available():
        return "cpu"
    
    # Get the number of GPUs
    num_gpus = torch.cuda.device_count()
    best_gpu = None
    max_memory = 0
    
    for i in range(num_gpus):
        gpu_memory = torch.cuda.get_device_properties(i).total_memory
        gpu_free_memory = torch.cuda.memory_reserved(i) - torch.cuda.memory_allocated(i)
        available_memory = gpu_memory - gpu_free_memory
        
        if available_memory > max_memory:
            max_memory = available_memory
            best_gpu = i
    
    if best_gpu is None:
        raise RuntimeError("No suitable GPU found")
    
    return f"cuda:{best_gpu}"

def get_unique_images(batches):
    unique_images = []
    for batch in batches:
        for image in batch:
            # Check if the image already exists in unique_images
            if not any(same_shape(image, unique_image) for unique_image in unique_images):
                
                unique_images.append(image)

    return unique_images