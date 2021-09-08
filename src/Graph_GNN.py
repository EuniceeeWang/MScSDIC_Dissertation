import torch
import torch.nn as nn

from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.simplelayers import SkipConnection
from monai.utils import alias, export
from torch_geometric.nn import SAGEConv
from typing import Sequence, Tuple, Union


# These two model classes are modified based on monai project unet model 
# https://docs.monai.io/en/latest/_modules/monai/networks/nets/unet.html
class UNet_GNN_pass(nn.Module):
    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        act: Union[Tuple, str] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout=0.0,
    ) -> None:
        super().__init__()
        delta = len(strides) - (len(channels) - 1)
        self.dimensions = dimensions
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.downs = []
        self.ups = []
        self.downs = []
        
        
        #graph sage conv
        self.sageconv1 = SAGEConv(in_channels = 256, out_channels = 256)
        self.relu = nn.ReLU(inplace=True)
        self.sageconv2 = SAGEConv(in_channels = 256, out_channels = 256)
        
        # this function using recurssive method to create encoder and decoder convolution layers
        def _create_block(
            inc: int, outc: int, channels: Sequence[int], strides: Sequence[int], is_top: bool
        ) -> nn.Sequential:
            c = channels[0]
            s = strides[0]

            subblock: nn.Module
            
            # if not in the bottom, recurss
            if len(channels) > 2:
                _create_block(c, c, channels[1:], strides[1:], False)
                upc = c * 2
                
            # get the bottom layer    
            else:
                self.subblock = self._get_bottom_layer(c, channels[1])
                upc = c + channels[1]

            self.downs.append(self._get_down_layer(inc, c, s, is_top))
            self.ups.append(self._get_up_layer(upc, outc, s, is_top))
        
        _create_block(in_channels, out_channels, self.channels, self.strides, True)
        self.up1, self.up2, self.up3, self.up4 = self.ups
        del self.ups
        self.down1, self.down2, self.down3, self.down4 = self.downs
        del self.downs

       
    def _get_down_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        # this functions is used to get encoding layers
        if self.num_res_units > 0:
            return ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
            )
        return Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
        )

    def _get_bottom_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        # this functions is used to get bottom layers
        return self._get_down_layer(in_channels, out_channels, 1, False)

    def _get_up_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        # this function is used to get decoding layers
        conv: Union[Convolution, nn.Sequential]

        conv = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            conv_only=is_top and self.num_res_units == 0,
            is_transposed=True,
        )

        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions,
                out_channels,
                out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                last_conv_only=is_top,
            )
            conv = nn.Sequential(conv, ru)

        return conv

    def forward(self, x: torch.Tensor, device, edges) -> torch.Tensor:
        edges = edges.to(device) # put edges to device
        xs = []        
        for m in [self.down4, self.down3, self.down2, self.down1]:
            x = m(x)
            #print(x.shape)
            xs.append(x)
        
        x = self.subblock(x)   
        #print(x.shape)
        x = x.view(x.shape[0], 256, -1).permute(0, 2, 1)
        #print(x.shape)
        
        # pass method will pass the embeddings right through the gnn layers and pass the output to the first decoding layers
        x = self.sageconv1(x=x, edge_index=edges)
        x = self.relu(x)
        x = self.sageconv2(x=x, edge_index=edges)
        x = self.relu(x)
        x = x.permute(0, 2, 1).view(x.shape[0], 256, 25, 25).float()
        #print(x.shape)
        for m, cat in zip([self.up1, self.up2, self.up3, self.up4], xs[::-1]):
            x = torch.cat([cat, x], dim=1)
            x = m(x)
            #print(x.shape)

        return x

# most of the code of UNet_GNN_linear is the same with the Unet_GNN_pass
# The only difference is that we used a linear layer to convert the concatenation of gnn dense features and resnet dense features to the 
# first decoding input
class UNet_GNN_linear(nn.Module):
    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        act: Union[Tuple, str] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout=0.0,
    ) -> None:
        super().__init__()
        delta = len(strides) - (len(channels) - 1)
        self.dimensions = dimensions
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.downs = []
        self.ups = []
        self.downs = []
        
        
        #graph sage conv
        self.sageconv1 = SAGEConv(in_channels = 256, out_channels = 256)
        self.relu = nn.ReLU(inplace=True)
        self.sageconv2 = SAGEConv(in_channels = 256, out_channels = 256)
        self.linear =  nn.Linear(512, 256)
        def _create_block(
            inc: int, outc: int, channels: Sequence[int], strides: Sequence[int], is_top: bool
        ) -> nn.Sequential:
            c = channels[0]
            s = strides[0]

            subblock: nn.Module

            if len(channels) > 2:
                _create_block(c, c, channels[1:], strides[1:], False)
                upc = c * 2
            else:
                self.subblock = self._get_bottom_layer(c, channels[1])
                upc = c + channels[1]

            self.downs.append(self._get_down_layer(inc, c, s, is_top))
            self.ups.append(self._get_up_layer(upc, outc, s, is_top))
        
        _create_block(in_channels, out_channels, self.channels, self.strides, True)
        print(len(self.ups), len(self.downs))
        self.up1, self.up2, self.up3, self.up4 = self.ups
        del self.ups
        self.down1, self.down2, self.down3, self.down4 = self.downs
        del self.downs

    def _get_down_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        if self.num_res_units > 0:
            return ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
            )
        return Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
        )

    def _get_bottom_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        return self._get_down_layer(in_channels, out_channels, 1, False)

    def _get_up_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        conv: Union[Convolution, nn.Sequential]

        conv = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            conv_only=is_top and self.num_res_units == 0,
            is_transposed=True,
        )

        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions,
                out_channels,
                out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                last_conv_only=is_top,
            )
            conv = nn.Sequential(conv, ru)

        return conv

    def forward(self, x: torch.Tensor, device, edges) -> torch.Tensor:
        edges = edges.to(device)
        xs = []        
        for m in [self.down4, self.down3, self.down2, self.down1]:
            x = m(x)
            #print(x.shape)
            xs.append(x)
        
        x = self.subblock(x)   
        #print(x.shape)
        graph_x = x.view(x.shape[0], 256, -1).permute(0, 2, 1)
        #print(x.shape)
        graph_x = self.sageconv1(x=graph_x, edge_index=edges)
        graph_x = self.relu(graph_x)
        graph_x = self.sageconv2(x=graph_x, edge_index=edges)
        graph_x = self.relu(graph_x)
        graph_x = graph_x.permute(0, 2, 1).view(graph_x.shape[0], 256, 25, 25).float()
        
        x = torch.cat([x, graph_x], dim=1).permute(0, 2, 3, 1)
        x = self.linear(x)
        x = self.relu(x)
        x = x.permute(0, 3, 1, 2)
        #print(x.shape)
        for m, cat in zip([self.up1, self.up2, self.up3, self.up4], xs[::-1]):
            x = torch.cat([cat, x], dim=1)
            x = m(x)
            #print(x.shape)

        return x