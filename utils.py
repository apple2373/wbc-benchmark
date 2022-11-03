import time
import os
import json
import random
import numpy as np
import torch
import torchvision


class MyImageFolder(torchvision.datasets.ImageFolder):
    '''
    Just making the ImageFolder to output dict instead of tuple
    '''

    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)

    def __getitem__(self, index: int):
        img, label = super().__getitem__(index)
        return {"input": img, "label": label}

# taken from pytorch imagenet example


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# taken from pytorch imagenet example


def acc_topk(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res


def accuracy(output, target): return acc_topk(output, target)[0]


def setup_device(gpu_id):
    # set up GPUS
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if int(gpu_id) == -2 and os.getenv('CUDA_VISIBLE_DEVICES') is not None:
        gpu_id = os.getenv('CUDA_VISIBLE_DEVICES')
    elif int(gpu_id) >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        print("set CUDA_VISIBLE_DEVICES=", gpu_id)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using device %s" % device)
    return device


def setup_seed(seed):
    if seed < 0:
        if os.getenv('SATOSHI_SEED') is not None and seed == -2:
            seed = int(os.getenv('SATOSHI_SEED'))
            print("env seed used")
        else:
            import math
            seed = int(10**4*math.modf(time.time())[0])
            seed = seed
    print("random seed", seed)
    return seed


def setup_savedir(prefix="", basedir="./experiments", args=None, append_args=[], add_time=True):
    savedir = prefix
    if len(append_args) > 0 and args is not None:
        for arg_opt in append_args:
            arg_value = getattr(args, arg_opt)
            savedir += "_"+arg_opt+"-"+str(arg_value)
    else:
        savedir += "exp"

    savedir = savedir.replace(" ", "").replace("'", "").replace('"', '')
    savedir = os.path.join(basedir, savedir)

    if add_time:
        now = time.localtime()
        d = time.strftime('%Y%m%d%H%M%S', now)
        savedir = savedir + "_" + str(d)

    # if exists, append _num-[num]
    i = 1
    savedir_ori = savedir
    while True:
        try:
            os.makedirs(savedir)
            break
        except FileExistsError:
            savedir = savedir_ori+"_num-%d" % i
            i += 1

    print("made the log directory", savedir)
    return savedir


def save_args(savedir, args, name="args.json"):
    # save args as "args.json" in the savedir
    path = os.path.join(savedir, name)
    with open(path, 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)
    print("args saved as %s" % path)


def save_json(dict, path):
    with open(path, 'w') as f:
        json.dump(dict, f, sort_keys=True, indent=4)
        print("log saved at %s" % path)


def resume_model(model, resume, state_dict_key="model"):
    '''
    model:pytorch model
    resume: path to the resume file
    state_dict_key: dict key
    '''
    print("resuming trained weights from %s" % resume)

    checkpoint = torch.load(resume, map_location='cpu')
    if state_dict_key is not None:
        pretrained_dict = checkpoint[state_dict_key]
    else:
        pretrained_dict = checkpoint

    try:
        model.load_state_dict(pretrained_dict)
    except RuntimeError as e:
        print(e)
        print("can't load the all weights due to error above, trying to load part of them!")
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict_use = {}
        pretrained_dict_ignored = {}
        for k, v in pretrained_dict.items():
            if k in model_dict:
                pretrained_dict_use[k] = v
            else:
                pretrained_dict_ignored[k] = v
        pretrained_dict = pretrained_dict_use
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
        print("resumed only", pretrained_dict.keys())
        print("ignored:", pretrained_dict_ignored.keys())

    return model


def advanced_load_state_dict(model, pretrained_dict, remove_key_prefix=""):
    model_dict = model.state_dict()
    # c.f. https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/32
    pretrained_dict_use = {}
    pretrained_dict_ignored = {}
    for k, v in pretrained_dict.items():
        # remove prefix if specified
        if len(remove_key_prefix) > 0 and k.startswith(remove_key_prefix):
            k = k[len(remove_key_prefix):]
        # use only if 1) the pretrained model has the same layer as the current model,
        #  and 2) the shape of tensors matches
        if k in model_dict and v.shape == model_dict[k].shape:
            pretrained_dict_use[k] = v
        else:
            pretrained_dict_ignored[k] = v
    pretrained_dict = pretrained_dict_use
    # 2overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # load the new state dict
    model.load_state_dict(model_dict)
    print("resumed:", pretrained_dict.keys())
    print("ignored:", pretrained_dict_ignored.keys())
    return model


def save_checkpoint(path, model, key="model"):
    # save model state dict
    checkpoint = {}
    checkpoint[key] = model.state_dict()
    torch.save(checkpoint, path)
    print("checkpoint saved at", path)


def make_deterministic(seed, strict=False, loose=False):
    '''
    scrict: very strict reproducability required.
    loose: speed is optimized at the cost of reproducability.
    '''

    # https://github.com/pytorch/pytorch/issues/7068#issuecomment-487907668
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    if loose:
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if strict:
        # https://github.com/pytorch/pytorch/issues/7068#issuecomment-515728600
        torch.backends.cudnn.enabled = False
        print("strict reproducability required! cudnn disabled. make sure to set num_workers=0 too!")


def check_gitstatus():
    import subprocess
    from subprocess import PIPE
    changed = "gitpython N/A"
    sha = None
    changed = None
    status = None
    # from https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script
    # from https://stackoverflow.com/questions/31540449/how-to-check-if-a-git-repo-has-uncommitted-changes-using-python
    # from https://stackoverflow.com/questions/33733453/get-changed-files-using-gitpython/42792158
    try:
        import git
        try:
            repo = git.Repo(search_parent_directories=True)
            sha = repo.head.object.hexsha
            changed = [item.a_path for item in repo.index.diff(None)]
        except Exception as e:
            print(e)
    except ImportError:
        print("cannot import gitpython ; try pip install gitpython")

    if sha is None:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], stdout=PIPE,
                             stderr=subprocess.PIPE).stdout.decode("utf-8").strip()
    print("git hash", sha)

    if status is None:
        status = subprocess.run(["git", "status"], stdout=PIPE,
                                stderr=subprocess.PIPE).stdout.decode("utf-8").strip()
    print("git status", status)

    return {"hash": sha, "changed": changed, "status": status}
