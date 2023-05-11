import os
import math
import torch
import cv2
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import logging
from torchvision import transforms
from logging.handlers import RotatingFileHandler
from PIL import Image
from torchsummary import summary
from datetime import datetime

from model.PWCNet import PWCNet

log = logging.getLogger("pwcnet")
if not log.handlers:
    log.setLevel(logging.DEBUG)
    fmt_str = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f'logs/pwcnet_{date}.log'
    formatter = logging.Formatter(fmt_str)

    # Setup file handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh = RotatingFileHandler(log_path, maxBytes=1000000, backupCount=4)
    fh.setFormatter(formatter)

    # Setup stream handler
    # ch = logging.StreamHandler()
    # ch.setLevel(logging.INFO)
    # ch.setFormatter(formatter)

    # Add handlers
    log.addHandler(fh)
    # _log.addHandler(ch)

# log = logging.getLogger("pwcnet")


class SintelDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, flow_path, transforms=None):
        super().__init__()
        log.info("Building dataset")
        self.img_path = img_path
        self.flow_path = flow_path

        self.img_filenames = []
        self.flow_filenames = []

        self.img_filenames_dict = {}
        self.flow_filenames_dict = {}
        self.img_sets = []
        
        # TEST_LIMIT = 1
        # i = 0
        for dir in os.listdir(self.img_path):
            # if i == TEST_LIMIT:
            #     break
            # else:
            #     i += 1
            if os.path.isdir(os.path.join(self.img_path, dir)):
                # If directory, loop through all images inside
                for f in os.listdir(os.path.join(self.img_path, dir)):
                    if dir not in self.img_filenames_dict:
                        self.img_filenames_dict[dir] = []
                    self.img_filenames_dict[dir].append(f)

        # i = 0
        for dir in os.listdir(self.flow_path):
            # if i == TEST_LIMIT:
            #     break
            # else:
            #     i += 1
            if os.path.isdir(os.path.join(self.flow_path, dir)):
                # If directory, loop through all images inside
                for f in os.listdir(os.path.join(self.flow_path, dir)):
                    if dir not in self.flow_filenames_dict:
                        self.flow_filenames_dict[dir] = []
                    self.flow_filenames_dict[dir].append(f)

        for dir in self.img_filenames_dict:
            log.info(f"Getting images from {dir}")
            for i in range(len(self.img_filenames_dict[dir]) - 1):
                img_set = {}
                img1 = self.open_img(os.path.join(self.img_path, dir, self.img_filenames_dict[dir][i]))
                img2 = self.open_img(os.path.join(self.img_path, dir, self.img_filenames_dict[dir][i+1]))
                img_set['imgs'] = torch.cat((img1, img2), 0)
                img_set['flow'] = self.load_flow(os.path.join(self.flow_path, dir, self.flow_filenames_dict[dir][i]))
                self.img_sets.append(img_set)


    def open_img(self, path):
        divisor = 64
        img = Image.open(path).convert('RGB')
        img = np.array(img)
        img = img[:, :, :3]
        H = img.shape[0]
        W = img.shape[1]
        H_ = int(math.ceil(H/divisor) * divisor)
        W_ = int(math.ceil(W/divisor) * divisor)
        img = cv2.resize(img, (W_, H_))
        img = img[:, :, ::-1]
        img = 1.0 * img/255
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img)
        # img = img.expand(1, img.size()[0], img.size()[1], img.size()[2])
        img = img.float()
        return img
    
        
    def load_flow(self, path):
        divisor = 64
        with open(path, 'rb') as f:
            magic = float(np.fromfile(f, np.float32, count = 1)[0])
            if magic == 202021.25:
                w, h = np.fromfile(f, np.int32, count = 1)[0], np.fromfile(f, np.int32, count = 1)[0]
                flow = np.fromfile(f, np.float32, count = h*w*2)
                flow.resize((h, w, 2))
                H = flow.shape[0]
                W = flow.shape[1]
                H_ = int(math.ceil(H/divisor) * divisor)
                W_ = int(math.ceil(W/divisor) * divisor)
                flow = cv2.resize(flow, (W_, H_))
                flow = flow[:, :, ::-1]
                flow = 1.0 * flow/255
                flow = np.transpose(flow, (2, 0, 1))
                flow = torch.from_numpy(flow)
                # flow = flow.expand(1, flow.size()[0], flow.size()[1], flow.size()[2])
                flow = flow.float()
                return flow
            return None


    def __getitem__(self, i):
        # img1 = self.img_sets[i]['img1']
        # img2 = self.img_sets[i]['img2']
        imgs = self.img_sets[i]['imgs']
        flow = self.img_sets[i]['flow']
        return imgs, flow
    

    def __len__(self):
        return len(self.img_sets)
    

def loss_fn(flows, ground_flow, model=None):
    # Notes:
    #   - Get ground truth flow (supervision signal) at each level by scaling down original ground truth flow
    #   - Upscale flow2 to match size of ground truth flow?
    
    a = [0.005, 0.01, 0.02, 0.08, 0.32]
    gamma = 0.0004

    gt_flows = []

    ground_flow = ground_flow/20
    
    gt2 = nn.functional.interpolate(ground_flow, size=(112, 256), mode='bilinear')
    gt3 = nn.functional.interpolate(ground_flow, size=(56, 128), mode='bilinear')
    gt4 = nn.functional.interpolate(ground_flow, size=(28, 64), mode='bilinear')
    gt5 = nn.functional.interpolate(ground_flow, size=(14, 32), mode='bilinear')
    gt6 = nn.functional.interpolate(ground_flow, size=(7, 16), mode='bilinear')

    # with torch.no_grad():
    #     # theta = [p for p in model.parameters() if p.requires_grad]
    #     # theta = torch.tensor(torch.stack(theta))
    #     theta = nn.utils.parameters_to_vector([p for p in model.parameters() if p.requires_grad])
    #     reg = gamma * torch.norm(theta)
    #     # reg = 1


    f2_l2 = torch.norm(gt2 - flows[0], dim=1) # L2 norm across flow (1st) dim (dims (8, 2, h, w))
    # print(f"shape1: {f2_l2.size()}")
    # f2_l2 = torch.sum(f2_l2, dim=0) # Sum across images in batch (0th) dim (dims (8, 1, h, w))
    f2_l2 = torch.sum(f2_l2, dim=(1, 2)) # Sum across h, w dims (dims (8, h, w))
    # print(f"shape2: {f2_l2.size()}")
    f2_l2 = torch.mean(f2_l2) # Mean across batch
    # print(f"shape3: {f2_l2.size()}")

    f3_l2 = torch.norm(gt3 - flows[1], dim=1) # L2 norm across flow (1st) dim (dims (8, 2, h, w))
    f3_l2 = torch.sum(f3_l2, dim=(1, 2)) # Sum across h, w dims (dims (8, 1, h, w))
    f3_l2 = torch.mean(f3_l2)

    f4_l2 = torch.norm(gt4 - flows[2], dim=1) # L2 norm across flow (1st) dim (dims (8, 2, h, w))
    f4_l2 = torch.sum(f4_l2, dim=(1, 2)) # Sum across h, w dims (dims (8, 1, h, w))
    f4_l2 = torch.mean(f4_l2)

    f5_l2 = torch.norm(gt5 - flows[3], dim=1) # L2 norm across flow (1st) dim (dims (8, 2, h, w))
    f5_l2 = torch.sum(f5_l2, dim=(1, 2)) # Sum across h, w dims (dims (8, 1, h, w))
    f5_l2 = torch.mean(f5_l2)

    f6_l2 = torch.norm(gt6 - flows[4], dim=1) # L2 norm across flow (1st) dim (dims (8, 2, h, w))
    f6_l2 = torch.sum(f6_l2, dim=(1, 2)) # Sum across h, w dims (dims (8, 1, h, w))
    f6_l2 = torch.mean(f6_l2)
    
    # l2 = flows[0] - gt2
    # l2 = torch.norm(l2)
    # # TODO: add norm of trainable params here? (theta in pwcnet paper) (model.params ?)
    # l2 = torch.sum(l2) + reg

    # l3 = flows[1] - gt3
    # l3 = torch.norm(l3)
    # # TODO: add norm of trainable params here? (theta in pwcnet paper) (model.params ?)
    # l3 = torch.sum(l3) + reg

    # l4 = flows[2] - gt4
    # l4 = torch.norm(l4)
    # # TODO: add norm of trainable params here? (theta in pwcnet paper) (model.params ?)
    # l4 = torch.sum(l4) + reg

    # l5 = flows[3] - gt5
    # l5 = torch.norm(l5)
    # # TODO: add norm of trainable params here? (theta in pwcnet paper) (model.params ?)
    # l5 = torch.sum(l5) + reg

    # l6 = flows[4] - gt6
    # l6 = torch.norm(l6)
    # # TODO: add norm of trainable params here? (theta in pwcnet paper) (model.params ?)
    # l6 = torch.sum(l6) + reg

    # loss = (l2*a[0]) + (l3*a[1]) + (l4*a[2]) + (l5*a[3]) + (l6*a[4])

    loss = (f2_l2*a[0]) + (f3_l2*a[1]) + (f4_l2*a[2]) + (f5_l2*a[3]) + (f6_l2*a[4])
    return loss
    

def endpiont_error(flow_pred, flow_gt):
    # print(f"flow_pred max: {torch.max(flow_pred)} flow_pred min: {torch.min(flow_pred)} flow_gt max: {torch.max(flow_gt)} flow_gt min: {torch.min(flow_gt)}")
    distance = torch.norm(flow_gt - flow_pred, p=2, dim=1)
    # print(f"distance max: {torch.max(distance)} distance min: {torch.min(distance)}")
    e = torch.mean(distance, (1, 2))
    e = torch.mean(e)
    # print(f"e: {e}")
    # e = torch.norm(flow_pred - flow_gt)
    return e


def match_size(target, img):
    # print(f"target: {target.size()}")
    # print(f"img: {img.size()}")
    H = target.size()[-2]
    W = target.size()[-1]
    img = transforms.Resize((H, W), antialias=True)(img)
    # print(f"type: {type(img)}")
    # print(f"img after: {img.size()}")
    return img

    



def train(model, iterator, optimizer, device, scheduler=None):
    log.info("    Training...")
    epoch_loss = 0
    acc_epe = 0

    model.train()
    
    for imgs, ground_flow in iterator:
        cpu_device = 'cpu'

        imgs = imgs.to(device)
        ground_flow = ground_flow.to(device)

        optimizer.zero_grad()
        flow2, flow3, flow4, flow5, flow6 = model(imgs)
        flows = [flow2, flow3, flow4, flow5, flow6]

        # To free memory on gpu
        imgs = imgs.to(cpu_device)

        # loss = loss_fn(flows, ground_flow, model)
        # print(f"0 - ground max: {torch.max(ground_flow)} ground min: {torch.min(ground_flow)} flow max: {torch.max(flow2)} flow min: {torch.min(flow2)}")
        loss = loss_fn(flows, ground_flow)

        flow3, flow4, flow5, flow6 = flow3.to('cpu'), flow4.to('cpu'), flow5.to('cpu'), flow6.to('cpu')

        loss.backward()
        

        optimizer.step()

        loss = loss.to('cpu')

        # if scheduler is not None:
        #     scheduler.step()
            # TODO: save learning rate here to plot at the end (turn this whole thing into a class to access learning_rate_hist variable?)

        # TODO: calculate accuracy here ?

        # print(f"flow2: {flow2.size()}")
        # print(f"ground_flow: {ground_flow.size()}")
        # flow_resize = flow_resize.to('cpu')
        # flow2, flow3, flow4, flow5, flow6 = flow2.to('cpu'), flow3.to('cpu'), flow4.to('cpu'), flow5.to('cpu'), flow6.to('cpu')
        
        ground_flow = ground_flow.to('cpu')
        flow2 = flow2.to('cpu')
        flow2 = match_size(ground_flow, flow2)
        flow2 = torch.mul(flow2, 20)
        # print(f"1 - ground max: {torch.max(ground_flow)} ground min: {torch.min(ground_flow)} flow max: {torch.max(flow2)} flow min: {torch.min(flow2)}")
        # epe = endpiont_error(flow_resize, ground_flow*20)
        epe = endpiont_error(flow2, ground_flow)
        # ground_flow = ground_flow.to('cpu')
        # flow2 = flow2.to('cpu')
        # flow_resize = flow_resize.to('cpu')
        epe = epe.to('cpu')
        # log.info(f"acc_epe: {type(acc_epe)}")
        # log.info(f"epe: {type(epe)}")
        acc_epe += epe.item()
        

        # print("loss: {}".format(loss.item()))
        epoch_loss += loss.item()
        # print(f"len: {len(iterator)}")
        # print(f"acc_epe: {acc_epe}")
        # print(f"2 - ground max: {torch.max(ground_flow)} ground min: {torch.min(ground_flow)} flow max: {torch.max(flow2)} flow min: {torch.min(flow2)}")
        # return 0, 0
    log.info(f"    acc_epe: {acc_epe} , # of batches: {len(iterator)}")
    avg_epe = acc_epe / len(iterator)
    avg_loss = epoch_loss / len(iterator)
    return avg_loss, avg_epe


def evaluate(model, iterator, device):
    log.info("    Evaluating...")
    epoch_loss = 0
    acc_epe = 0

    model.eval()
    
    with torch.no_grad():
        for imgs, ground_flow in iterator:
            cpu_device = 'cpu'

            imgs = imgs.to(device)
            ground_flow = ground_flow.to(device)

            flow2, flow3, flow4, flow5, flow6 = model(imgs)
            flows = [flow2, flow3, flow4, flow5, flow6]

            # To free memory on gpu
            imgs = imgs.to(cpu_device)

            loss = loss_fn(flows, ground_flow, model)

            # TODO: calculate accuracy here ?

            ground_flow = ground_flow.to('cpu')
            # flow_resize = flow_resize.to('cpu')
            flow2, flow3, flow4, flow5, flow6 = flow2.to('cpu'), flow3.to('cpu'), flow4.to('cpu'), flow5.to('cpu'), flow6.to('cpu')

            flow2 = match_size(ground_flow, flow2)
            flow2 = flow2*20
            epe = endpiont_error(flow2, ground_flow/20)
            epe = epe.to('cpu')

            acc_epe += epe

            epoch_loss += loss.item()
            # return 0, 0
    avg_epe = acc_epe / len(iterator)
    avg_loss = epoch_loss / len(iterator)

    return avg_loss, avg_epe





if __name__ == "__main__":
    log.info("Starting pwc.py")

    # Determine device
    if (torch.cuda.is_available() == True):
        log.info("Using device: cuda")
        device = 'cuda'
    else:
        log.info("Using device: cpu")
        device = 'cpu'

    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    log.info(f"Starting at: {now}")

    # Params
    batch_size = 8
    EPOCHS = 100
    SAVE_PATH = now + "_model.pt"

    # Build dataset
    img_path = ".data\\MPI-Sintel-complete\\training\\clean"
    flow_path = ".data\\MPI-Sintel-complete\\training\\flow"
    sintel_data = SintelDataset(img_path, flow_path)
    log.info(f"Dataset length: {len(sintel_data)}")

    train_data, val_data = torch.utils.data.random_split(sintel_data, [0.8, 0.2])
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)

    # Define model
    model = PWCNet()
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=0.0005)
    optimizer = torch.optim.Adam(params, lr=0.0001)
    # scheduler = None
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    # model_sum = summary(model, (6, 448, 1024), batch_size=1, device='cuda')

    train_losses = []
    val_losses = []
    
    best_val_loss = None

    for epoch in range(EPOCHS):
        log.info("TRAINING - epoch {}".format(epoch))
        # train_loss = train(model, train_dataloader, optimizer, device, scheduler)
        train_loss, train_epe = train(model, train_dataloader, optimizer, device)
        val_loss, val_epe = evaluate(model, val_dataloader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if scheduler is not None:
            scheduler.step()

        # TODO: save model here if val loss is best
        if (best_val_loss is None) or (val_loss < best_val_loss):
            log.info("    Saving model")
            best_val_loss = val_loss
            torch.save(model.state_dict(), SAVE_PATH)

        log.info("    epoch: {} // train loss: {} // train epe: {} // val loss: {} // val epe: {}".format(epoch, train_loss, train_epe, val_loss, val_epe))


    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    log.info(f"Ending at: {now}")

    log.info("pwc - end")
