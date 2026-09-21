from rfdetr import RFDETRBase, RFDETRSegPreview

#import torch.multiprocessing
#torch.multiprocessing.set_sharing_strategy('file_system')

model = RFDETRSegPreview()
#model = RFDETRBase()

batch_size = 1
batch_full = 16

model.train(
    #dataset_dir="/home/quant/datasets/cityscapes/coco_coarse",
    dataset_dir="/home/quant/datasets/cityscapes/coco",
    #dataset_dir="/home/quant/datasets/cityscapes/coco_excluded",
    output_dir="trained_models/att5",
    resume="trained_models/att5/checkpoint.pth",
    epochs=120,
    lr_drop=32,
    batch_size=batch_size,
    batch_size_test=8,
    grad_accum_steps=batch_full/batch_size,
    lr=1e-4,
    use_ema=True,
    # aux_device="cpu",
    # criterion_device="cpu",
)
