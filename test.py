import torch
import random
from torch import nn
from tqdm import tqdm
from ResNet_Model import load_images, train_test_split, get_batches, _resnet, Bottleneck, BasicBlock


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        res = []
        top = output.topk(max(topk), dim=1).indices
        t = target.argmax(dim=1).unsqueeze(1).expand(*top.size())
        for k in topk:
            res.append(torch.sum((t == top)[:, :k]))
        # print(top)
        # print(t)
        # print(res)
        # input()

        return res


DATA_FOLDER = "resized"
IMAGE_SIZE = 256
NUM_CLASSES = 22
NUM_CONFIGS = 100
EPOCHS = 50
BATCH_SIZE = 32

torch.random.manual_seed(1)
torch.cuda.manual_seed(1)
random.seed(1)

batches = get_batches(BATCH_SIZE, DATA_FOLDER)
train_set, valid_set = train_test_split(batches)
struct_list = [Bottleneck, BasicBlock]

var1, var2, var3, var4, var5, var6, var7, var8, var9, var10 = (
    6, 5, 20, 5, 1, 2, 4.228304269036996e-05, 0.995757784592188, 0.9877921655451408, 0.3975623291097461)
net = _resnet(struct_list[var5], [var1, var2, var3, var4]).cuda()
net.load_state_dict(torch.load("net.pt"))
net.eval()
loss_fn = nn.CrossEntropyLoss()
total_loss = 0
total_correct = 0
total_correct3 = 0
total_correct5 = 0
total_pred = 0
pbar = tqdm(enumerate(valid_set), total=len(valid_set))
for i, batch in pbar:
    data, labels = load_images(batch)

    with torch.cuda.amp.autocast():
        with torch.no_grad():
            pred = net(data.cuda())
            loss = loss_fn(pred, labels.cuda())
            correct, correct3, correct5 = accuracy(
                pred, labels.cuda(), (1, 3, 5))

    total = labels.size(0)

    total_loss += loss.item()
    total_correct += correct.item()
    total_correct3 += correct3.item()
    total_correct5 += correct5.item()
    total_pred += total

    pbar.set_description(
        f'Model Validation, Loss {loss.item():.4f}({total_loss/(i+1):.4f}), Accuracy Top1: {total_correct / total_pred:.4f} Top3: {total_correct3 / total_pred:.4f} Top5: {total_correct5 / total_pred:.4f}')
