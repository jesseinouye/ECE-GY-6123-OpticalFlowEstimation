import os
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torchsummary import summary
os.environ['PYTHON_EGG_CACHE'] = 'tmp/' # a writable directory
from .correlation_package.correlation import Correlation
# from .correlation_package.correlation import Correlation



class PWCNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Maximum displacement for correlation
        max_disp = 4

        # Feature Extraction
        self.fe_block1 = self.fe_conv_block(3, 16)
        self.fe_block2 = self.fe_conv_block(16, 32)
        self.fe_block3 = self.fe_conv_block(32, 64)
        self.fe_block4 = self.fe_conv_block(64, 96)
        self.fe_block5 = self.fe_conv_block(96, 128)
        self.fe_block6 = self.fe_conv_block(128, 196)

        self.corr = Correlation(pad_size=max_disp, kernel_size=1, max_displacement=max_disp, stride1=1, stride2=1, corr_multiply=1)
        self.leakyReLU = nn.LeakyReLU(0.1)

        # self.corr = self.corr_block(max_disp)

        nd = (2*max_disp+1)**2
        dims = np.cumsum([128, 128, 96, 64, 32])

        in_dim = nd
        # self.ofe_block6 = self.ofe_block(in_dim)
        # self.ofe_block6 = self.build_ofe_conv(in_dim)
        self.pred_flow6 = nn.Conv2d(in_dim+dims[4], 2, kernel_size=3, stride=1, padding=1, bias=True)
        self.up_feat6 = nn.ConvTranspose2d(in_dim+dims[4], 2, kernel_size=4, stride=2, padding=1)

        self.ofe_block6_0 = self.conv_block(in_dim, 128, kernel_size=3, stride=1)
        self.ofe_block6_1 = self.conv_block(in_dim+dims[0], 128, kernel_size=3, stride=1)
        self.ofe_block6_2 = self.conv_block(in_dim+dims[1], 96, kernel_size=3, stride=1)
        self.ofe_block6_3 = self.conv_block(in_dim+dims[2], 64, kernel_size=3, stride=1)
        self.ofe_block6_4 = self.conv_block(in_dim+dims[3], 32, kernel_size=3, stride=1)

        in_dim = nd+128+4
        # self.ofe_block5 = self.ofe_block(in_dim)
        self.pred_flow5 = nn.Conv2d(in_dim+dims[4], 2, kernel_size=3, stride=1, padding=1, bias=True)
        self.up_feat5 = nn.ConvTranspose2d(in_dim+dims[4], 2, kernel_size=4, stride=2, padding=1)

        self.ofe_block5_0 = self.conv_block(in_dim, 128, kernel_size=3, stride=1)
        self.ofe_block5_1 = self.conv_block(in_dim+dims[0], 128, kernel_size=3, stride=1)
        self.ofe_block5_2 = self.conv_block(in_dim+dims[1], 96, kernel_size=3, stride=1)
        self.ofe_block5_3 = self.conv_block(in_dim+dims[2], 64, kernel_size=3, stride=1)
        self.ofe_block5_4 = self.conv_block(in_dim+dims[3], 32, kernel_size=3, stride=1)

        in_dim = nd+96+4
        # self.ofe_block4 = self.ofe_block(in_dim)
        self.pred_flow4 = nn.Conv2d(in_dim+dims[4], 2, kernel_size=3, stride=1, padding=1, bias=True)
        self.up_feat4 = nn.ConvTranspose2d(in_dim+dims[4], 2, kernel_size=4, stride=2, padding=1)

        self.ofe_block4_0 = self.conv_block(in_dim, 128, kernel_size=3, stride=1)
        self.ofe_block4_1 = self.conv_block(in_dim+dims[0], 128, kernel_size=3, stride=1)
        self.ofe_block4_2 = self.conv_block(in_dim+dims[1], 96, kernel_size=3, stride=1)
        self.ofe_block4_3 = self.conv_block(in_dim+dims[2], 64, kernel_size=3, stride=1)
        self.ofe_block4_4 = self.conv_block(in_dim+dims[3], 32, kernel_size=3, stride=1)

        in_dim = nd+64+4
        # self.ofe_block3 = self.ofe_block(in_dim)
        self.pred_flow3 = nn.Conv2d(in_dim+dims[4], 2, kernel_size=3, stride=1, padding=1, bias=True)
        self.up_feat3 = nn.ConvTranspose2d(in_dim+dims[4], 2, kernel_size=4, stride=2, padding=1)

        self.ofe_block3_0 = self.conv_block(in_dim, 128, kernel_size=3, stride=1)
        self.ofe_block3_1 = self.conv_block(in_dim+dims[0], 128, kernel_size=3, stride=1)
        self.ofe_block3_2 = self.conv_block(in_dim+dims[1], 96, kernel_size=3, stride=1)
        self.ofe_block3_3 = self.conv_block(in_dim+dims[2], 64, kernel_size=3, stride=1)
        self.ofe_block3_4 = self.conv_block(in_dim+dims[3], 32, kernel_size=3, stride=1)

        in_dim = nd+32+4
        # self.ofe_block2 = self.ofe_block(in_dim)
        self.pred_flow2 = nn.Conv2d(in_dim+dims[4], 2, kernel_size=3, stride=1, padding=1, bias=True)
        
        self.ofe_block2_0 = self.conv_block(in_dim, 128, kernel_size=3, stride=1)
        self.ofe_block2_1 = self.conv_block(in_dim+dims[0], 128, kernel_size=3, stride=1)
        self.ofe_block2_2 = self.conv_block(in_dim+dims[1], 96, kernel_size=3, stride=1)
        self.ofe_block2_3 = self.conv_block(in_dim+dims[2], 64, kernel_size=3, stride=1)
        self.ofe_block2_4 = self.conv_block(in_dim+dims[3], 32, kernel_size=3, stride=1)

        self.deconv = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1)

        self.context_conv1 = self.conv_block(in_dim+dims[4], 128, kernel_size=3, stride=1, padding=1, dilation=1)
        self.context_conv2 = self.conv_block(128, 128, kernel_size=3, stride=1, padding=1, dilation=1)
        self.context_conv3 = self.conv_block(128, 128, kernel_size=3, stride=1, padding=1, dilation=1)
        self.context_conv4 = self.conv_block(128, 96, kernel_size=3, stride=1, padding=1, dilation=1)
        self.context_conv5 = self.conv_block(96, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.context_conv6 = self.conv_block(64, 32, kernel_size=3, stride=1, padding=1, dilation=1)
        self.context_conv7 = nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    # def corr_block(self, max_disp):
    #     block = nn.Sequential(
    #         Correlation(pad_size=max_disp, kernel_size=1, max_displacement=max_disp, stride1=1, stride2=2, corr_multiply=1),
    #         nn.LeakyReLU(0.1)
    #     )
    #     return block


    def conv_block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=True),
            nn.LeakyReLU(0.1)
        )
        return block

    # Build feature extraction block
    def fe_conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.LeakyReLU(0.1),
        )
        return block

    def ofe_block(self, input_dim):
        dims = np.cumsum([128, 128, 96, 64, 32])

        block = nn.Sequential(
            nn.Conv2d(input_dim, 128, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(input_dim + dims[0], 128, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(input_dim + dims[1], 96, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(input_dim + dims[2], 64, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(input_dim + dims[3], 32, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.LeakyReLU(0.1),
        )
        return block

    def build_ofe_conv(self, input_dim):
        dims = [0, 128, 128, 96, 64, 32]
        convs = []
        for i in range(len(dims)-1):
            block = nn.Sequential(
                nn.Conv2d(input_dim + dims[i], dims[i+1], kernel_size=3, stride=1, padding=1, dilation=1),
                nn.ReLU(0.1)
            )
            convs.append(block)

        return convs


    # NOTE: copied from original PWCNet implementation
    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1] 
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)        
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)

        # if W==128:
            # np.save('mask.npy', mask.cpu().data.numpy())
            # np.save('warp.npy', output.cpu().data.numpy())
        
        mask[mask<0.9999] = 0
        mask[mask>0] = 1
        
        return output*mask
    
    def forward(self, x):
        im1 = x[:,:3,:,:]
        im2 = x[:,3:,:,:]

        fe1_im1 = self.fe_block1(im1)
        fe1_im2 = self.fe_block1(im2)
        fe2_im1 = self.fe_block2(fe1_im1)
        fe2_im2 = self.fe_block2(fe1_im2)
        fe3_im1 = self.fe_block3(fe2_im1)
        fe3_im2 = self.fe_block3(fe2_im2)
        fe4_im1 = self.fe_block4(fe3_im1)
        fe4_im2 = self.fe_block4(fe3_im2)
        fe5_im1 = self.fe_block5(fe4_im1)
        fe5_im2 = self.fe_block5(fe4_im2)
        fe6_im1 = self.fe_block6(fe5_im1)
        fe6_im2 = self.fe_block6(fe5_im2)

        corr6 = self.corr(fe6_im1, fe6_im2)
        corr6 = self.leakyReLU(corr6)
        # x1 = self.ofe_block6[0](corr6)
        x1 = self.ofe_block6_0(corr6)
        x1 = torch.cat((x1, corr6), 1)
        x2 = self.ofe_block6_1(x1)
        x2 = torch.cat((x2, x1), 1)
        x3 = self.ofe_block6_2(x2)
        x3 = torch.cat((x3, x2), 1)
        x4 = self.ofe_block6_3(x3)
        x4 = torch.cat((x4, x3), 1)
        x5 = self.ofe_block6_4(x4)
        x5 = torch.cat((x5, x4), 1)
        flow6 = self.pred_flow6(x5)
        upflow6 = self.deconv(flow6)
        upfeat6 = self.up_feat6(x5)

        warp5 = self.warp(fe5_im2, upflow6*0.625)
        corr5 = self.corr(fe5_im1, warp5)
        corr5 = self.leakyReLU(corr5)
        x0 = torch.cat((corr5, fe5_im1, upflow6, upfeat6), 1)
        x1 = self.ofe_block5_0(x0)
        x1 = torch.cat((x1, x0), 1)
        x2 = self.ofe_block5_1(x1)
        x2 = torch.cat((x2, x1), 1)
        x3 = self.ofe_block5_2(x2)
        x3 = torch.cat((x3, x2), 1)
        x4 = self.ofe_block5_3(x3)
        x4 = torch.cat((x4, x3), 1)
        x5 = self.ofe_block5_4(x4)
        x5 = torch.cat((x5, x4), 1)
        flow5 = self.pred_flow5(x5)
        upflow5 = self.deconv(flow5)
        upfeat5 = self.up_feat5(x5)

        warp4 = self.warp(fe4_im2, upflow5*1.25)
        corr4 = self.corr(fe4_im1, warp4)
        corr4 = self.leakyReLU(corr4)
        x0 = torch.cat((corr4, fe4_im1, upflow5, upfeat5), 1)
        x1 = self.ofe_block4_0(x0)
        x1 = torch.cat((x1, x0), 1)
        x2 = self.ofe_block4_1(x1)
        x2 = torch.cat((x2, x1), 1)
        x3 = self.ofe_block4_2(x2)
        x3 = torch.cat((x3, x2), 1)
        x4 = self.ofe_block4_3(x3)
        x4 = torch.cat((x4, x3), 1)
        x5 = self.ofe_block4_4(x4)
        x5 = torch.cat((x5, x4), 1)
        flow4 = self.pred_flow4(x5)
        upflow4 = self.deconv(flow4)
        upfeat4 = self.up_feat4(x5)

        warp3 = self.warp(fe3_im2, upflow4*2.5)
        corr3 = self.corr(fe3_im1, warp3)
        corr3 = self.leakyReLU(corr3)
        x0 = torch.cat((corr3, fe3_im1, upflow4, upfeat4), 1)
        x1 = self.ofe_block3_0(x0)
        x1 = torch.cat((x1, x0), 1)
        x2 = self.ofe_block3_1(x1)
        x2 = torch.cat((x2, x1), 1)
        x3 = self.ofe_block3_2(x2)
        x3 = torch.cat((x3, x2), 1)
        x4 = self.ofe_block3_3(x3)
        x4 = torch.cat((x4, x3), 1)
        x5 = self.ofe_block3_4(x4)
        x5 = torch.cat((x5, x4), 1)
        flow3 = self.pred_flow3(x5)
        upflow3 = self.deconv(flow3)
        upfeat3 = self.up_feat3(x5)
        
        warp2 = self.warp(fe2_im2, upflow3*5.0)
        corr2 = self.corr(fe2_im1, warp2)
        corr2 = self.leakyReLU(corr2)
        x0 = torch.cat((corr2, fe2_im1, upflow3, upfeat3), 1)
        x1 = self.ofe_block2_0(x0)
        x1 = torch.cat((x1, x0), 1)
        x2 = self.ofe_block2_1(x1)
        x2 = torch.cat((x2, x1), 1)
        x3 = self.ofe_block2_2(x2)
        x3 = torch.cat((x3, x2), 1)
        x4 = self.ofe_block2_3(x3)
        x4 = torch.cat((x4, x3), 1)
        x5 = self.ofe_block2_4(x4)
        x5 = torch.cat((x5, x4), 1)
        flow2 = self.pred_flow2(x5)

        y = self.context_conv1(x5)
        y = self.context_conv2(y)
        y = self.context_conv3(y)
        y = self.context_conv4(y)
        y = self.context_conv5(y)
        y = self.context_conv6(y)
        y = flow2 + self.context_conv7(y)

        # if self.training:
        #     return y, flow3, flow4, flow5, flow6
        # else:
        #     return y
        
        return y, flow3, flow4, flow5, flow6
    


if __name__ == "__main__":

    if (torch.cuda.is_available() == True):
        print("  Using device: cuda")
        device = 'cuda'
    else:
        print("  Using device: cpu")
        device = 'cpu'

    model = PWCNet()
    model = model.to(device)
    model_sum = summary(model, (6, 448, 1024), batch_size=1, device='cuda')
