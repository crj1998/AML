
import torch
import torch.nn.functional as F

@torch.enable_grad()
def rpgd(model, images, labels, steps, step_size, epsilon):
    model.eval()
    delta = 0.005 * torch.randn_like(images)
    delta = torch.clamp(images + delta, min=0, max=1) - images

    for _ in range(steps):
        delta.requires_grad_(True)
        loss = F.cross_entropy(model(images + delta), labels)
        grad = torch.autograd.grad(loss, [delta])[0].detach()

        delta = delta.detach() + step_size * grad.sign()
        delta = torch.clamp(delta, - epsilon, + epsilon)
        delta = torch.clamp(images + delta, 0.0, 1.0) - images

    return delta

@torch.enable_grad()
def trades(model, images, steps, step_size, epsilon):
    model.eval()
    delta = 0.005 * torch.randn_like(images)
    delta = torch.clamp(images + delta, min=0, max=1) - images
    
    with torch.no_grad():
        probs = F.softmax(model(images), dim=-1)
        
    for _ in range(steps):
        delta.requires_grad_(True)
        loss = F.kl_div(F.log_softmax(model(images + delta), dim=-1), probs, reduction="batchmean")
        grad = torch.autograd.grad(loss, [delta])[0].detach()

        delta = delta.detach() + step_size * grad.sign()
        delta = torch.clamp(delta, - epsilon, + epsilon)
        delta = torch.clamp(images + delta, 0.0, 1.0) - images

    return delta