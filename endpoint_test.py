import torch

def endpoint_error(gt_flow, pred_flow):
    """
    Calculate the endpoint error between ground truth and predicted flows across a batch of images.
    """
    # Calculate the Euclidean distance between the ground truth and predicted flows
    distance = torch.norm(gt_flow - pred_flow, p=2, dim=1)
    print(distance.size())
    # Calculate the endpoint error as the average distance over all pixels in the image
    epe = torch.mean(distance, (1, 2))
    print(epe.size())
    epe = torch.mean(epe)
    
    return epe

# Example usage
gt_flow = torch.randn(8, 2, 448, 1024)  # ground truth flows
pred_flow = torch.randn(8, 2, 448, 1024)  # predicted flows
epe = endpoint_error(gt_flow, pred_flow)
print(f"Endpoint error: {epe.item()}")