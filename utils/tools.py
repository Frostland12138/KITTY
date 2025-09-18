

def write_config(args):
    print(args.experiment + ":" + args.name + "_" + args.info)
    dic = vars(args)
    for i in dic.items():
        print(i)

