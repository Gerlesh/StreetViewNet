import torch
import random
import numpy as np
from tqdm import tqdm
from ResNet_Model import load_images, get_batches, _resnet, Bottleneck, BasicBlock


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        res = []
        top = output.topk(max(topk), dim=1).indices
        t = target.argmax(dim=1).unsqueeze(1).expand(*top.size())
        for k in topk:
            res.append(torch.sum((t == top)[:, :k]))

        return res, top[:,0], t[:,0]

DATA_FOLDER = "resized"
NUM_CLASSES = 22
BATCH_SIZE = 32

torch.random.manual_seed(1)
#torch.cuda.manual_seed(1)
random.seed(1)

test_set = get_batches(BATCH_SIZE, DATA_FOLDER)
struct_list = [Bottleneck, BasicBlock]

var1, var2, var3, var4, var5, var6, var7, var8, var9, var10 = (
    6, 5, 20, 5, 1, 2, 4.228304269036996e-05, 0.995757784592188, 0.9877921655451408, 0.3975623291097461)
net = _resnet(struct_list[var5], [var1, var2, var3, var4]) # .cuda()
net.load_state_dict(torch.load("net.pt", map_location=torch.device('cpu')))
net.eval()

top1_array = torch.zeros(NUM_CLASSES)
pred_array = torch.zeros(NUM_CLASSES)
most_confident = torch.zeros(NUM_CLASSES)
most_confident_imgs = np.array(np.zeros(NUM_CLASSES), dtype=object)
total_correct = 0
total_correct3 = 0
total_correct5 = 0
total_pred = 0
pbar = tqdm(enumerate(test_set), total=len(test_set))
for i, batch in pbar:
    data, labels = load_images(batch)

    #with torch.cuda.amp.autocast():
    with torch.no_grad():
        pred = net(data) #.cuda()
        (correct,correct3,correct5), top1, t1 = accuracy(
            pred, labels, (1, 3, 5)) #.cuda()

    total = labels.size(0)

    concat = torch.concat((most_confident.view(1,-1), pred))
    for i in range(NUM_CLASSES):
        max_idx = torch.argmax(concat[:, i])
        if max_idx == 0:
            continue
        most_confident[i] = concat[max_idx, i]
        most_confident_imgs[i] = batch[max_idx-1]

    nonzero = torch.nonzero(top1 == t1)
    if nonzero.size(0) > 0:
        top1_array += torch.bincount(top1[nonzero.flatten()], minlength=22)
    pred_array += torch.bincount(t1, minlength=22)

    total_correct += correct.item()
    total_correct3 += correct3.item()
    total_correct5 += correct5.item()
    total_pred += total

    pbar.set_description(
        f'Model Test, Accuracy Top1: {total_correct / total_pred:.4f} Top3: {total_correct3 / total_pred:.4f} Top5: {total_correct5 / total_pred:.4f}')
print("Top 1 Accuracy for each class: ", top1_array / pred_array)
print("Most confident predictions for each class: ")
for i in range(NUM_CLASSES):
    print(f"{i}: {most_confident[i]}")
    print(most_confident_imgs[i])