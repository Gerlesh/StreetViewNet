from typing import Any, Callable, List, Optional, Type, Union, Tuple
import os
import random

import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch import Tensor
import torchvision.transforms as transforms

from PIL import Image
from tqdm import tqdm

DATA_FOLDER = "resized"
IMAGE_SIZE = 256
NUM_CLASSES = 22
NUM_CONFIGS = 100
EPOCHS = 50
BATCH_SIZE = 32


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                "BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = NUM_CLASSES,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.softmax = nn.Softmax(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    # type: ignore[arg-type]
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    # type: ignore[arg-type]
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.softmax(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    **kwargs: Any,
) -> ResNet:

    model = ResNet(block, layers, **kwargs)

    return model

# def resnet101(*,  progress: bool = False, **kwargs: Any) -> ResNet:
#     """ResNet-101 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`__.
#     .. note::
#        The bottleneck of TorchVision places the stride for downsampling to the second 3x3
#        convolution while the original paper places it to the first 1x1 convolution.
#        This variant improves the accuracy and is known as `ResNet V1.5
#        <https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch>`_.
#     Args:
#         weights (:class:`~torchvision.models.ResNet101_Weights`, optional): The
#             pretrained weights to use. See
#             :class:`~torchvision.models.ResNet101_Weights` below for
#             more details, and possible values. By default, no pre-trained
#             weights are used.
#         progress (bool, optional): If True, displays a progress bar of the
#             download to stderr. Default is True.
#         **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
#             base class. Please refer to the `source code
#             <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
#             for more details about this class.
#     .. autoclass:: torchvision.models.ResNet101_Weights
#         :members:
#     """

#     return _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)


def get_batches(batch_size: int, data_folder: str) -> Tuple[Tuple[str]]:
    files = os.listdir(data_folder)
    batches = []
    for i in range(1, len(files)//batch_size):
        batches.append(tuple(files[(i-1)*batch_size:i*batch_size]))

    if len(batches[-1]) < batch_size:
        batches.pop(-1)

    return tuple(batches)


def load_images(batch: Tuple[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    data = torch.zeros((len(batch), 3, IMAGE_SIZE, IMAGE_SIZE))
    labels = torch.zeros((len(batch), NUM_CLASSES))
    transform = transforms.ToTensor()
    for i, im in enumerate(batch):
        image = Image.open(os.path.join(DATA_FOLDER, im))
        data[i, :, :, :] = transform(image)
        labels[i, int(im[:-4].split()[2])] = 1

    return data, labels


def train_test_split(batches: Tuple[Tuple[str]], train_ratio: float = 0.9) -> Tuple[Tuple[Tuple[str]], Tuple[Tuple[str]]]:
    train_set = tuple(random.choices(batches, k=int(train_ratio*len(batches))))
    test_set = tuple([b for b in batches if b not in train_set])

    return train_set, test_set


def train_epoch(model, train_set, valid_set, optimizer, epoch):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
    total_loss = 0
    total_correct = 0
    total_pred = 0
    pbar = tqdm(enumerate(train_set), total=len(train_set))
    for i, batch in pbar:
        data, labels = load_images(batch)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            pred = model(data.cuda())
            loss = loss_fn(pred, labels.cuda())
            correct = torch.sum(torch.argmax(pred, dim=1) ==
                                torch.argmax(labels.cuda(), dim=1))

        if loss != loss:
            print(data)
            print(pred)
        total = labels.size(0)

        total_loss += loss.item()
        total_correct += correct.item()
        total_pred += total

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        pbar.set_description(
            f'Epoch {epoch}, Loss {loss.item():.4f}({total_loss / (i + 1):.4f}), Accuracy {correct.item()/total:.4f}({total_correct / total_pred:.4f})')

    total_loss = 0
    total_correct = 0
    total_pred = 0
    pbar = tqdm(enumerate(valid_set), total=len(valid_set))
    for i, batch in pbar:
        data, labels = load_images(batch)

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                pred = model(data.cuda())
                loss = loss_fn(pred, labels.cuda())
                correct = torch.sum(torch.argmax(pred, dim=1) ==
                                    torch.argmax(labels.cuda(), dim=1))

        total = labels.size(0)

        total_loss += loss.item()
        total_correct += correct.item()
        total_pred += total

        pbar.set_description(
            f'Epoch {epoch} Validation, Loss {loss.item():.4f}({total_loss / (i + 1):.4f}), Accuracy {correct.item()/total:.4f}({total_correct / total_pred:.4f})')

    return total_loss/len(valid_set)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    torch.random.manual_seed(1)
    torch.cuda.manual_seed(1)
    random.seed(1)

    batches = get_batches(BATCH_SIZE, DATA_FOLDER)
    train_set, valid_set = train_test_split(batches)

    struct_list = [Bottleneck, BasicBlock]

    # Hyperparameter Tuning Loop
    hyperparameters = []
    losses = []
    for i in range(NUM_CONFIGS):
        print("Model", i)
        var1 = random.randint(2, 8)
        var2 = random.randint(3, 10)
        var3 = random.randint(12, 30)
        var4 = random.randint(2, 8)
        var5 = random.randint(0, 1)
        var6 = random.randint(0, 3)
        var7 = pow(10, (-3 * random.random() - 1.5))
        var8 = 1 - pow(10, (-2 * random.random() - 0.5))
        var9 = 1 - pow(10, (-2 * random.random() - 1))
        var10 = random.random()
        print((var1, var2, var3, var4, var5, var6, var7, var8, var9, var10))

        net = _resnet(struct_list[var5], [var1, var2, var3, var4]).cuda()
        if (var6 == 0):
            optimizer = torch.optim.SGD(net.parameters(), lr=var7)
        elif (var6 == 1):
            optimizer = torch.optim.SGD(
                net.parameters(), lr=var7, momentum=var10)
        elif (var6 == 2):
            optimizer = torch.optim.Adam(
                net.parameters(), lr=var7, betas=(var8, var9))
        else:
            optimizer = torch.optim.AdamW(
                net.parameters(), lr=var7, betas=(var8, var9))

        min_loss = float('inf')
        count = 0
        for j in range(EPOCHS):
            valid_loss = train_epoch(net, train_set, valid_set, optimizer, j)

            if (valid_loss < min_loss):
                min_loss = valid_loss
                count = 0
            count += 1
            if (count == 5):
                break  # if 5 epochs without better validation_performance, finish training

        loss_fn = nn.CrossEntropyLoss()
        total_loss = 0
        total_correct = 0
        total_pred = 0
        pbar = tqdm(enumerate(valid_set), total=len(valid_set))
        for i, batch in pbar:
            data, labels = load_images(batch)

            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    pred = net(data.cuda())
                    loss = loss_fn(pred, labels.cuda())
                    correct = torch.sum(torch.argmax(pred, dim=1) ==
                                        torch.argmax(labels.cuda(), dim=1))

            total = labels.size(0)

            total_loss += loss.item()
            total_correct += correct.item()
            total_pred += total

            pbar.set_description(
                f'Model {i} Validation, Loss {loss.item():.4f}({total_loss:.4f}), Accuracy {correct.item()/total:.4f}({total_correct / total_pred:.4f})')

        if not losses or total_loss <= min(losses):
            torch.save(net.state_dict(), "net.pt")

        losses.append(total_loss)
        hyperparameters.append(
            (var1, var2, var3, var4, var5, var6, var7, var8, var9, var10)
        )

    print(hyperparameters[torch.argmin(torch.Tensor(losses))])

    # var1, var2, var3, var4, var5, var6, var7, var8, var9, var10 = (
    #     8, 10, 30, 8, 1, 3, 1e-04, 0.995, 0.925, 0.175)
    # net = _resnet(struct_list[var5], [var1, var2, var3, var4]).cuda()
    # if (var6 == 0):
    #     optimizer = torch.optim.SGD(net.parameters(), lr=var7)
    # elif (var6 == 1):
    #     optimizer = torch.optim.SGD(
    #         net.parameters(), lr=var7, momentum=var10)
    # elif (var6 == 2):
    #     optimizer = torch.optim.Adam(
    #         net.parameters(), lr=var7, betas=(var8, var9))
    # else:
    #     optimizer = torch.optim.AdamW(
    #         net.parameters(), lr=var7, betas=(var8, var9))
    #
    # min_loss = float('inf')
    # count = 0
    # try:
    #     for j in range(EPOCHS):
    #         for batch in tqdm(train_set, desc="Epoch "+str(j)):
    #             optimizer.zero_grad()
    #             data, labels = load_images(batch)
    #             output = net(data.cuda())
    #             loss = loss_func(output, labels.cuda())
    #             loss.backward()
    #             optimizer.step()
    #
    #         total = 0
    #         correct = 0
    #         valid_loss = 0
    #         for batch in valid_set:
    #             data, valid_labels = load_images(batch)
    #             with torch.no_grad():
    #                 valid_output = net(data.cuda())
    #                 valid_loss += loss_func(valid_output,
    #                                         valid_labels.cuda()).item()
    #
    #             correct += torch.sum(torch.argmax(valid_output, dim=1) ==
    #                                  torch.argmax(valid_labels.cuda(), dim=1)).item()
    #             total += valid_labels.size(0)
    #
    #         valid_loss /= len(valid_set)
    #         success_rate = correct / total
    #         print("Epoch", j, "Validation loss:",
    #               valid_loss, "Accuracy:", success_rate)
    #
    #         if (valid_loss < min_loss):
    #             min_loss = valid_loss
    #             count = 0
    #         count += 1
    #         if (count == 5):
    #             break  # if 5 epochs without better validation_performance, finish training
    # except KeyboardInterrupt:
    #     pass
    #
    # total = 0
    # correct = 0
    # valid_loss = 0
    # for batch in valid_set:
    #     data, valid_labels = load_images(batch)
    #     with torch.no_grad():
    #         valid_output = net(data.cuda())
    #         valid_loss += loss_func(valid_output,
    #                                 valid_labels.cuda()).item()
    #
    #     correct += torch.sum(torch.argmax(valid_output, dim=1) ==
    #                          torch.argmax(valid_labels.cuda(), dim=1)).item()
    #     total += valid_labels.size(0)
    #
    # valid_loss /= len(valid_set)
    #
    # success_rate = correct / total
    # print("The success rate for this iteration is", success_rate)
    # print("The validation loss for this iteration is", valid_loss)

    # net = _resnet(Bottleneck, [3, 4, 23, 3]).cuda()
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    #
    # for i in range(10):
    #     optimizer.zero_grad()
    #     output = net(input)
    #     loss = loss_func(output, labels)
    #     loss.backward()
    #     optimizer.step()
