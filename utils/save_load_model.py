import os
import torch


def save_model(args, model, optimizer, current_epoch, exp, name, index):
    out = os.path.join(args.model_path, "{}/{}/checkpoint_{}_{}.tar".format(exp, name, name, index))
    path = os.path.join(args.model_path, "{}/{}".format(exp, name))
    if not os.path.exists(path):
        os.makedirs(path)
    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': current_epoch}
    torch.save(state, out)


def load_model(args, model):
    model_fp = os.path.join(args.model_path, args.reload_model)
    checkpoint = torch.load(model_fp)
    status = model.load_state_dict(checkpoint['net'], strict=False)
    print("load:", status)
    if args.start_epoch != 0:
        args.start_epoch = checkpoint['epoch'] + 1
