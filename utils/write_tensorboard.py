from tensorboardX import SummaryWriter


def get_summarywriter(path):
    return SummaryWriter(path)


def write_tensorboard_log(writer, eval_info, loss_info, epoch):
    for info in eval_info:
        writer.add_scalar(
            "eval_info:/{}:".format(info[1]),
            info[0],
            epoch
        )
    for info in loss_info:
        writer.add_scalar(
            "loss_info:/{}:".format(info[1]),
            info[0],
            epoch
        )
    writer.flush()