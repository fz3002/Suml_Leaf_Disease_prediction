import torch


def image_to__tensor():
    pass


def path_to_tensor():
    pass


@torch.no_grad()
def predict_image(model, images: torch.Tensor, device: torch.device):
    images = images.to(device, non_blocking=True).float()
    logits = model(images)
    return logits.argmax(dim=1).cpu()