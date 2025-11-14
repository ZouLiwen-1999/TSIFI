# -*- coding: utf-8 -*-
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
join=os.path.join
ld=os.listdir
import cv2 
import logging
import sys
import torch
from PIL import Image
import time
from multiprocessing import Pool
import monai
# import segmentation_models_pytorch as smp
from tqdm import tqdm
import tempfile
from torch.utils.data import DataLoader
from monai.data import create_test_image_2d, list_data_collate, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    AddChanneld,
    AsDiscrete,
    Compose,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ScaleIntensityd,
    EnsureTyped,
    EnsureType,
    Resized
)
import warnings
from skimage import metrics
warnings.filterwarnings('ignore')
""
root='./data/ABFM'
method='before'
model_name = 'unetr'#模型名称   
plans_fname = join(root,'splits_final.pkl')
plans = load_pickle(plans_fname)
img_path = join(root,'processed_images')#这里是图像路径
gt_path = join(root,'processed_labels')#这里是标注路径
result_path=join(root,method,'output/seg',model_name)
vis_path=join(root,method,'output/vis',model_name)
for path in [result_path,vis_path]:
    if not os.path.exists(path):
        os.makedirs(path)
# define transforms for image and segmentation
train_transforms = Compose(
    [   
        LoadImaged(keys=["img", "seg"]),
        AddChanneld(keys=["img", "seg"]),
        ScaleIntensityd(keys="img"),
        # Resized(keys=["img"],spatial_size=[512,512],mode="linear"),
        # Resized(keys=["seg"],spatial_size=[512,512],mode="nearest"),
        # RandCropByPosNegLabeld(
        #     keys=["img", "seg"], label_key="seg", spatial_size=[96, 96], pos=1, neg=1, num_samples=4
        # ),
        RandRotate90d(keys=["img", "seg"], prob=0.5, spatial_axes=[0, 1]),
        EnsureTyped(keys=["img", "seg"]),
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["img", "seg"]),
        AddChanneld(keys=["img", "seg"]),
        # Resized(keys=["img"],spatial_size=[512,512],mode="linear"),
        # Resized(keys=["seg"],spatial_size=[512,512],mode="nearest"),
        ScaleIntensityd(keys="img"),
        EnsureTyped(keys=["img", "seg"]),
    ]
)

val_interval = 2#训练多少轮验证一次
epoch_num = 200#训练轮次
loss_function = monai.losses.DiceCELoss(sigmoid=True)#损失函数
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

""
def train(test_fold): 
    save_path =join(root,method,'models',model_name,'fold'+str(test_fold))#训练好的模型参数存放路径
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    #选择模型
    model  = monai.networks.nets.UNETR(in_channels=1, out_channels=1, img_size=512, feature_size=32, norm_name='batch', spatial_dims=2).cuda()
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)#优化器
#     model_path = join(save_path, "best.pth")
#     model.load_state_dict(torch.load(model_path))
    
    if test_fold==4:
        val_fold=0
    else:
        val_fold=test_fold+1
    train_fold=[0,1,2,3,4]
    train_fold.remove(test_fold)
    train_fold.remove(val_fold)
    
    train_images=[]
    train_masks=[]
    for fold in train_fold:
        for file in plans[fold]['val']:
            train_images.append(join(img_path,file+'.png'))
            train_masks.append(join(gt_path,file+'.png'))
            
    val_images=[]
    val_masks=[]
    for file in plans[val_fold]['val']:
        val_images.append(join(img_path,file+'.png'))
        val_masks.append(join(gt_path,file+'.png'))

    train_files = [{"img": img, "seg": seg} for img, seg in zip(train_images,train_masks)]
    val_files = [{"img": img, "seg": seg} for img, seg in zip(val_images, val_masks)]

    #%% create a training data loader
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    train_loader = DataLoader(
        train_ds,
        batch_size=4,#batch_size选取
        shuffle=True,
        num_workers=0,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )
    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=2, num_workers=0, collate_fn=list_data_collate)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    post_trans = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold_values=True)])
    #%% create UNet, DiceLoss and Adam optimizer
    
    # start a typical PyTorch training
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    
    for epoch in range(epoch_num):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epoch_num}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in tqdm(train_loader):
            step += 1
            inputs, labels = batch_data["img"].to(device), batch_data["seg"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            # outputs = torch.sigmoid(val_outputs) # sigmoid activation function is in the loss
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None
                for val_data in val_loader:
                    val_images, val_labels = val_data["img"].to(device), val_data["seg"].to(device)
                    val_outputs = model(val_images)
                    val_outputs = torch.sigmoid(val_outputs)
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs>0.5, y=val_labels)
                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()
                metric_values.append(metric)
    #             #隔几轮保存
    #             if metric > best_metric:
    #                 best_metric = metric
    #             best_metric_epoch = epoch + 1
    #             torch.save(model.state_dict(), join(save_path, str(epoch)+"_"+str(metric)+".pth"))

                #保存最佳
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), join(save_path, "best_ep200.pth"))
                    print("saved new best metric model")

                print(
                    "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")

def predict_whole_img(test_img_dir,name,model):
    img = cv2.imread(join(test_img_dir, name),0)
    # print('img.shape: ', img.shape)
    img_resize = cv2.resize(img, (512,512), interpolation=cv2.INTER_CUBIC)
#     img_resize = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
    data_one_tensor = torch.from_numpy(img_resize/255).unsqueeze(0).unsqueeze(0).float().cuda()
#     print('##',data_one_tensor.shape)
    predict_one_tensor = model(data_one_tensor)
    predict_one_tensor = torch.sigmoid(predict_one_tensor)
    predict_one_array = predict_one_tensor.cpu().squeeze(0).squeeze(0).detach().numpy()
    img_pred = np.uint8(predict_one_array>0.5)*255
    # img_pred_post = morphology.remove_small_objects(ndimage.binary_fill_holes(img_pred))
    seg_resize = cv2.resize(np.uint8(img_pred), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
#     print('seg_resize shape:', seg_resize.shape)
    cv2.imwrite(join(join(result_path, name)), seg_resize)


def dice_equation(mask1, mask2):
    intersection = (mask1 * mask2).sum()
    union = (mask1 + mask2).sum()
    fp=mask1.sum()-intersection.sum()
    fn=mask2.sum()-intersection.sum()
    if union != 0:
        dices = float((2 * intersection) / union+1e-8)
        fpsr=fp/(union+1e-8)
        fnsr=fn/(union+1e-8)
    else:
        dices = 0
    
    return dices,fpsr,fnsr

def evaluation():
    print('start to evaluation...')
    DSC,FPSR,FNSR,HD=0,0,0,0
    n=0
    for file in tqdm(ld(result_path)):
        if file[-3:]=='png':
            n+=1
            y_pred=np.array(Image.open(join(result_path,file))).astype(np.uint8)/255
#             print(np.unique(y_pred))
            y_true=np.array(Image.open(join(gt_path,file))).astype(np.uint8)
#             print(np.unique(y_true))
            dice,fpsr,fnsr=dice_equation(y_pred, y_true)
            if np.sum(y_pred)>0:
                    hd=metrics.hausdorff_distance(y_pred, y_true)
            else:
                hd=100
            # print(file,dice)
            DSC+=dice
            FPSR+=fpsr
            FNSR+=fnsr
            HD+=hd
    print('Mean DSC:',DSC/n)
    print('Mean FPSR:',FPSR/n)
    print('Mean FNSR:',FNSR/n)
    print('Mean HD:',HD/n)

def union_image_mask(name,vis_path):
    img = cv2.imread(join(img_path,name)) 
    pre = cv2.imread(join(result_path,name),0) 
    gt = cv2.imread(join(gt_path,name),0) 
    gt=np.uint8(gt*255) 
    _, thresh_gt = cv2.threshold(gt,127,255,cv2.THRESH_BINARY)
    contours_gt, _ = cv2.findContours(thresh_gt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours_gt, -1, (0, 255, 0), 1)#绿色的是标注
    
    _, thresh_pre = cv2.threshold(pre,127,255,cv2.THRESH_BINARY)
    contours_pre, _ = cv2.findContours(thresh_pre, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours_pre, -1, (0, 255, 255),1)#黄色的是预测
    cv2.imwrite(join(vis_path, name), img)

def vis():
    print('start to visulization...')
    for name in tqdm(ld(result_path)):
        if name[-3:]=='png':
            union_image_mask(name,vis_path)


def test(test_fold):
    print('start to predict...')
    model_path = join(root,method, 'models',model_name,'fold'+str(test_fold),'best_ep200.pth')
    model  = monai.networks.nets.UNETR(in_channels=1, out_channels=1, img_size=512, feature_size=32, norm_name='batch', spatial_dims=2).cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        for file in plans[test_fold]['val']:
                predict_whole_img(img_path,file+'.png',model)
    print('finished!')

def main():
#     test(0)
    for test_fold in [0,1,2,3,4]:
        print('---------------Start fold ',test_fold,'training---------------')
        train(test_fold)
        test(test_fold)
    evaluation()
    vis()

if __name__ == '__main__':
    main()
