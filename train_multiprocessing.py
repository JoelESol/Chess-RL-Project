from alpha_net import ChessNet, train
import os
import pickle
import numpy as np
import torch
import torch.multiprocessing as mp

if __name__=="__main__":
    net_to_train="current_net_trained_iter1.pth.tar"; save_as="current_net_trained_iter2.pth.tar"
    data_path ="./datasets/iter0/"; model_path ="./model_data/"
    datasets = []
    for idx, file in enumerate(os.listdir(data_path)):
        filename = os.path.join(data_path,file)
        with open(filename, 'rb') as fo:
            datasets.extend(pickle.load(fo, encoding='bytes'))
    datasets = np.array(datasets)

    mp.set_start_method("spawn", force=True)
    net = ChessNet()
    cuda = torch.cuda.is_available()
    if cuda:
        net.cuda()
    net.share_memory()
    net.train()

    current_net_filename = os.path.join(model_path, net_to_train)
    checkpoint = torch.load(current_net_filename)
    net.load_state_dict(checkpoint['state_dict'])
    print("Loaded nets: Starting training")

    processes = []
    for i in range(8):
        p = mp.Process(target=train, args=(net, datasets, 400, 3, i))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    torch.save({'state_dict': net.state_dict()}, os.path.join(model_path, save_as))
    print("Model Saved")
