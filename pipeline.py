from alpha_net import ChessNet, train
from MCTS import MCTS_self_play
import os
import pickle
import numpy as np
import torch
import torch.multiprocessing as mp

if __name__=="__main__":
    for iteration in range(10):
        net_to_play="current_net_trained_iter"+str(iteration+1)+".pth.tar"
        mp.set_start_method("spawn", force=True)
        net = ChessNet()
        cuda=torch.cuda.is_available()
        if cuda:
            net.cuda()
        net.share_memory()
        net.eval()
        current_net_filename = os.path.join("./model_data/", net_to_play)
        checkpoint = torch.load(current_net_filename)
        net.load_state_dict(checkpoint['state_dict'])
        processes1 = []
        for i in range(36):
            p1= mp.Process(target=MCTS_self_play, args=(net, 5, i))
            p1.start()
            processes1.append(p1)
        for p1 in processes1:
            p1.join()

        #Runs Net Training
        net_to_train = "current_net_trained_iter"+str(iteration+1)+".pth.tar"; save_as="current_net_trained_iter"+str(iteration+2)+".pth.tar"
        #gather data
        data_path = "./datasets/iter5/"
        datasets=[]
        for idx, file, in enumerate(os.listdir(data_path)):
            filename = os.path.join(data_path, file)
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
        current_net_filename = os.path.join("./model_data/", net_to_train)
        checkpoint = torch.load(current_net_filename)
        net.load_state_dict(checkpoint['state_dict'])

        processes2 = []
        for i in range(8):
            p2 = mp.Process(target=train, args=(net, datasets, 0, 15, i))
            p2.start()
            processes2.append(p2)
        for p2 in processes2:
            p2.join()
        # save results
        torch.save({'state_dict': net.state_dict()}, os.path.join("./model_data/", save_as))
