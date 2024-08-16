"""
 Splitting trained single-encoder-multi-decoder into three networks
 Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 If you use this code, please cite the following paper:
 Mahmoud Afifi and Michael S Brown. Deep White-Balance Editing. In CVPR, 2020.
"""
__author__ = "Mahmoud Afifi"
__credits__ = ["Mahmoud Afifi"]

import copy

from arch import deep_wb_single_task


def splitNetworks(net):
    # create instances from deepWBnet (the single task model)
    net_awb = deep_wb_single_task.deepWBnet()
    net_t = deep_wb_single_task.deepWBnet()
    net_s = deep_wb_single_task.deepWBnet()
    # copy AWB weights
    net_awb.encoder_inc = copy.deepcopy(net.encoder_inc)
    net_awb.encoder_down1 = copy.deepcopy(net.encoder_down1)
    net_awb.encoder_down2 = copy.deepcopy(net.encoder_down2)
    net_awb.encoder_down3 = copy.deepcopy(net.encoder_down3)
    net_awb.encoder_bridge_down = copy.deepcopy(net.encoder_bridge_down)
    net_awb.decoder_bridge_up = copy.deepcopy(net.awb_decoder_bridge_up)
    net_awb.decoder_up1 = copy.deepcopy(net.awb_decoder_up1)
    net_awb.decoder_up2 = copy.deepcopy(net.awb_decoder_up2)
    net_awb.decoder_up3 = copy.deepcopy(net.awb_decoder_up3)
    net_awb.decoder_out = copy.deepcopy(net.awb_decoder_out)
    
    return net_awb
