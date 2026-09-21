
import sys
sys.path.append('BANet/banet-2d')
sys.path.append('BANet/banet-2d/core')


def load_banet(device):
    import torch
    from core.BANet import BANet
    print("Imports OK")
    banet = torch.nn.DataParallel(BANet(None))
    # checkpoint = torch.load("weights/banet_sceneflow.pth")
    checkpoint = torch.load("weights/banet_kitti.pth")
    banet.load_state_dict(checkpoint, strict=True)
    print("Loaded weights")
    banet = banet.module
    banet.to(device)
    banet.eval()
    return banet


