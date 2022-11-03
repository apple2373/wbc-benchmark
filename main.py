import os
import time
from datetime import datetime
import argparse
from tqdm.auto import tqdm

import utils
from utils import accuracy
from utils import AverageMeter
from utils import MyImageFolder

import timm
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy

import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

utils.make_deterministic(123)


def setup_args():
    parser = argparse.ArgumentParser(description="WBC Classification")

    parser.add_argument('--train-img-root', type=str,
                        default="./data/RaabinWBC/Train", help="training image root")
    parser.add_argument('--test-img-root', type=str,
                        default="./data/LISCCropped", help="testing image root")

    parser.add_argument('--workers', type=int, default=8,
                        help="workers for torch.utils.data.DataLoader")
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help="gpu id. -1 means cpu. -2 means use CUDA_VISIBLE_DEVICES one")

    parser.add_argument('--batch', type=int, default=128, help="batch size")
    parser.add_argument('--lr', type=float, default=0.0001,
                        help="initial learning rate")
    parser.add_argument('--decay', type=float,
                        default=0.005, help="weight decay")
    parser.add_argument('--cosine-warmup', type=int, default=10,
                        help="number of warmup epochs under cosine lr decay")

    parser.add_argument('--epochs', type=int, default=100,
                        help="maximum number of epochs. if 0, evaluation only mode")
    parser.add_argument('--eval-freq', type=int, default=1,
                        help="evaluation frequnecy in epochs")

    parser.add_argument('--backbone', type=str, choices=['resnet50', 'vgg16', 'vgg16_bn', 'resnet50_gn', 'vit_b_16'],
                        default="resnet50_gn", help="backbone cnns")
    parser.add_argument('--resume', type=str, default=None,
                        help="checkpoint to resume")

    parser.add_argument('--saveroot',  default="./experiments/default/",
                        help='Root directory to make the output directory')
    parser.add_argument('--saveprefix',  default="log",
                        help='prefix to append to the name of log directory')
    parser.add_argument('--saveargs', default=["backbone", "addfc", 'freezebn', 'last16only', 'mixup', 'seed'],
                        nargs='+', help='args to append to the name of log directory')

    parser.add_argument('--addfc', type=int, default=0, choices=[0, 1],
                        help="add VGG style FC to ResNet50 or not")
    parser.add_argument('--freezebn', type=int, default=0, choices=[0, 1],
                        help="Freeze BatchNorm layers or not")
    parser.add_argument('--last16only', type=int, default=0, choices=[0, 1],
                        help="finetune last16only or not , legacy option")

    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')

    parser.add_argument('--seed', default=1, type=int,
                        help='seed. -1 means random from time. -2 is SATOSHI_SEED env')

    return parser.parse_args()


def setup_dataset(args):
    # setup dataset as pandas data frame
    df_dict = {}
    df_dict["train"] = args.train_img_root
    df_dict["test"] = args.test_img_root

    dataset_dict = {}
    # key is train/val/test and the value is corresponding pytorch dataset
    for split, df in df_dict.items():
        # target_transform is mapping from category name to category idx start from 0
        if split == "train":
            tforms = []
            tforms.append(transforms.Resize(256))
            tforms.append(transforms.RandomCrop((224, 224)))
            tforms.append(transforms.ToTensor())
            tforms.append(transforms.RandomHorizontalFlip())
            tforms.append(transforms.RandomVerticalFlip())
            tforms.append(transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
            transform = transforms.Compose(tforms)
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                     0.229, 0.224, 0.225])
            ])

        ds = MyImageFolder(df, transform=transform)

        class_to_idx = {'Basophil': 0,
                        'Eosinophil': 1,
                        'Lymphocyte': 2,
                        'Monocyte': 3,
                        'Neutrophil': 4}
        if len(ds.class_to_idx) != len(class_to_idx):
            # RaabinWBC Test-B has this problem
            print("Not all WBC categories are included, so reindex labels")
            ds = fix_incomplete_ds(ds, class_to_idx)

        dataset_dict[split] = ds

    return dataset_dict


def fix_incomplete_ds(ds, class_to_idx):
    classes_old = ds.classes
    ds.samples = [(path, class_to_idx[classes_old[idx]])
                  for path, idx in ds.samples]
    ds.targets = [s[1] for s in ds.samples]
    ds.class_to_idx = class_to_idx
    ds.classes = sorted(class_to_idx.keys())
    return ds


def setup_dataloader(args, dataset_dic):
    dataloader_dict = {}
    for split, dataset in dataset_dic.items():
        batch_size = args.batch
        dataloader_dict[split] = DataLoader(dataset,
                                            batch_size=batch_size,
                                            shuffle=split == "train",
                                            num_workers=args.workers,
                                            )
    return dataloader_dict


def setup_backbone(name, num_classes=5):
    # create backbone cnn instance
    if name == 'resnet50_gn':
        model = timm.create_model('resnetv2_50d_gn', pretrained=True)
        print("called resnet50-group-normalization with imagenet pretrained weights")
    else:
        model = getattr(torchvision.models, name)(weights='IMAGENET1K_V1')
        print("called torchvision.models.%s with imagenet pretrained weights")

    # change the last layer for WBC classification
    if name.startswith("vgg"):
        num_features = int(model.classifier[6].in_features)
        model.classifier[6] = nn.Linear(num_features, num_classes)
    elif name == 'resnet50':
        num_features = int(model.fc.in_features)
        model.fc = nn.Linear(num_features, num_classes)
    elif name == 'resnet50_gn':
        num_features = int(model.head.fc.in_channels)
        model.head.fc = nn.Conv2d(
            num_features, num_classes, kernel_size=(1, 1), stride=(1, 1))
    elif name.startswith("vit"):
        num_features = int(model.heads.head.in_features)
        model.heads.head = nn.Linear(num_features, num_classes)
    else:
        raise NotImplementedError("Undefined backbone name %s" % name)

    return model


def get_lrs_wramup_then_cosine(lr, warmup, epochs):
    # just make fake optimizer for using CosLE
    optimizer = torch.optim.SGD([nn.Parameter(torch.ones(1))], lr=lr)
    scheduler_cos = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, epochs-warmup)
    # remove first and last as they are zero, and start rate
    lrs = torch.linspace(0, lr, warmup+2)[1:-1].tolist()
    for i in range(epochs-warmup+1):
        optimizer.zero_grad()
        optimizer.step()
        lrs += scheduler_cos.get_last_lr()
        scheduler_cos.step()
    assert lrs[-1] == 0  # the last lr should be zero, which we don't use
    lrs = lrs[0:-1]
    assert len(lrs) == epochs
    return lrs


def train_one_epoch(args, dataloader, model, criterion, optimizer, accuracy=accuracy, device=None, mixup_fn=None):
    since = time.time()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()  # Set model to training mode
    if args.freezebn:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    losses = AverageMeter()
    accs = AverageMeter()

    for i, data in enumerate(tqdm(dataloader)):
        inputs = data["input"]
        labels = data["label"]

        if mixup_fn is not None:
            if len(inputs) % 2 == 1:
                # for some reasons, this mixup needs to have even numbers as batch size...
                inputs = inputs[0:-1]
                labels = labels[0:-1]
            inputs, labels = mixup_fn(inputs, labels)

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        if mixup_fn is not None:
            labels = labels.argmax(dim=1)
        acc = accuracy(outputs, labels)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), outputs.size(0))
        accs.update(acc.item(), outputs.size(0))

        print_txt = "current loss: %0.5f " % loss.item()
        print_txt += "acc %0.5f " % acc.item()
        print_txt += "| running average loss %0.5f " % losses.avg
        print_txt += "acc %0.5f " % accs.avg
        if i % 50 == 0:
            print(i, print_txt)

    time_elapsed = time.time() - since
    print(print_txt)
    print('this epoch took {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    return float(losses.avg), float(accs.avg)


def evaluate(dataloader, model, criterion, accuracy, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    losses = AverageMeter()
    accs = AverageMeter()
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader)):
            inputs = data["input"].to(device)
            labels = data["label"].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            acc = accuracy(outputs, labels)
            losses.update(loss.item(), outputs.size(0))
            accs.update(acc.item(), outputs.size(0))

    print("eval loss %0.5f acc %0.5f " % (losses.avg, accs.avg))
    return float(losses.avg), float(accs.avg)


def main(args):
    since = time.time()
    print(args)
    # set seed
    args.seed = utils.setup_seed(args.seed)
    utils.make_deterministic(args.seed)
    # setup the directory to save the experiment log and trained models
    log_dir = utils.setup_savedir(prefix=args.saveprefix, basedir=args.saveroot, args=args,
                                  append_args=args.saveargs)
    log = {}
    log["git"] = utils.check_gitstatus()
    log["timestamp"] = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_save_path = os.path.join(log_dir, "log.json")
    utils.save_json(log, log_save_path)
    # save args
    utils.save_args(log_dir, args)
    # setup device
    device = utils.setup_device(args.gpu)

    # setup dataset and dataloaders
    dataset_dict = setup_dataset(args)
    dataloader_dict = setup_dataloader(args, dataset_dict)

    # setup backbone cnn
    model = setup_backbone(args.backbone)
    if args.resume is not None:
        model = utils.resume_model(model, args.resume, state_dict_key="model")

    if args.addfc:
        assert args.backbone == 'resnet50'
        print('Add VGG style fc to ResNet50')
        # a vgg, but still using global average pooling
        in_features = model.fc.in_features
        num_classes = model.fc.out_features
        model.fc = nn.Sequential(*[nn.Linear(in_features, in_features),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout(),
                                   nn.Linear(in_features, in_features),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout(),
                                   nn.Linear(in_features, num_classes),
                                   ])

    if args.last16only:
        assert args.backbone == 'resnet50'
        print('Train only the last 16 layers of ResNet50')
        # fine tune last 16 layers only
        for p in model.parameters():
            p.requires_grad = False
        # find last 16 conv layers...
        mlist = []
        for m1 in model.children():
            if isinstance(m1, nn.Sequential):
                for m2 in m1.children():
                    mlist.append(m2)
            else:
                mlist.append(m1)
        counter = 0
        for m1 in mlist[-7:]:
            for p in m1.parameters():
                p.requires_grad = True
            # sanity check
            for m in m1.modules():
                if isinstance(m, nn.Conv2d):
                    print(m)
                    counter += 1
        assert counter == 16

    # setup mixup
    # https://github.com/facebookresearch/mae/blob/be47fef7a727943547afb0c670cf1b26034c3c89/main_finetune.py
    mixup_fn = None
    if args.mixup > 0:
        print("Mixup enabled with %0.2f!" % args.mixup)
        mixup_fn = Mixup(mixup_alpha=args.mixup, mode='batch',
                         label_smoothing=0.1, num_classes=5)

    # setup loss
    criterion = torch.nn.CrossEntropyLoss().to(device)
    criterion_train = criterion
    if mixup_fn is not None:
        criterion_train = SoftTargetCrossEntropy().to(device)

    # training
    if args.epochs > 0:
        # setup optimizer
        # https://www.fast.ai/posts/2018-07-02-adam-weight-decay.html
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.decay)

        if args.cosine_warmup > args.epochs:
            print("warmup is larger than the total epochs. setting warmup to zero.")
            args.cosine_warmup = 0

        # first arg should be 1 because LambdaLR *multiplies* into the lr of the optimizer
        lrs = get_lrs_wramup_then_cosine(1, args.cosine_warmup, args.epochs)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: lrs[epoch-1], verbose=True)
        log["train"] = []
        log["lr"] = []
        print(model)
    for epoch in range(args.epochs):
        print("epoch: %d --starts from 0 and ends at %d" %
              (epoch, args.epochs-1))
        loss, acc = train_one_epoch(args, dataloader_dict["train"], model, criterion_train,
                                    optimizer, accuracy=accuracy, device=device, mixup_fn=mixup_fn)
        log["train"].append({'epoch': epoch, "loss": loss, "acc": acc})
        utils.save_json(log, log_save_path)
        max_lr_now = max([group['lr'] for group in optimizer.param_groups])
        log["lr"].append(max_lr_now)
        lr_scheduler.step()
    if args.epochs > 0:
        save_path = os.path.join(log_dir, "model_%d.pth" % epoch)
        utils.save_checkpoint(save_path, model, key="model")

    # testing
    print("test started")
    loss, acc = evaluate(
        dataloader_dict["test"], model, criterion, accuracy=accuracy, device=device)
    log["test"] = {"loss": loss, "acc": acc}

    # save the final log
    time_elapsed = time.time() - since
    log["time_elapsed"] = time_elapsed
    utils.save_json(log, log_save_path)


if __name__ == '__main__':
    args = setup_args()
    main(args)
