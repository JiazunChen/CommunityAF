import argparse
import hashlib
import os
import datetime
import random
from conv import GraphConv
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
from time import time
from multiprocessing import Pool, cpu_count
from utils import load_data,preprocess_nodefeats,PretrainComDataset,get_augment,get_data,make_single_node_encoding,make_nodes_encoding
from model import CSAFModel
from eval import save_model,generate_batch,calu

def margin_loss(x1,x2,target,margin,reduce):
    if reduce =='mean':
        return torch.mean(torch.sqrt(torch.relu(((x2-x1+1)**2-(1-margin)**2))))
    elif reduce=='sum':
        return torch.mean(torch.sqrt(torch.relu(((x2 - x1 + 1) ** 2 - (1 - margin) ** 2))))

def train_epoch(args,device,epoch_cnt,model,train_dataloader,optimizer,roll_mapper,pre_dataset,attr):
    t_start = time()
    batch_losses = []
    model.train()
    batch_cnt = 0
    for i_batch, batch_data in (enumerate(train_dataloader)):

        print('batch_data', batch_data[:3])
        batch_data = get_data(batch_data.tolist(),args,roll_mapper,pre_dataset)
        optimizer.zero_grad()
        batch_cnt += 1
        batch_result, ln_var = model(attr, batch_data)
        out_z, out_logdet = batch_result['out_z'], batch_result['out_logdet']
        loss = model.log_prob(out_z, out_logdet)
        if args.rankingloss and epoch_cnt >= args.cnt_rank:
            stop_score = batch_result['stop_score']
            pospair = batch_result['pospair']
            negpair = batch_result['negpair']
            target = [1] * len(pospair)
            target = torch.LongTensor(target).to(device)
            if args.squrank:
                rank_loss = margin_loss(stop_score[pospair], stop_score[negpair], target,margin=args.margin,reduce=args.reduce)
            else:
                rank_loss = torch.nn.functional.margin_ranking_loss(stop_score[pospair], stop_score[negpair], target,margin=args.margin,reduce='sum')
            loss += rank_loss*args.gamma
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
        if args.suffle == True:
            break
    epoch_loss = sum(batch_losses) / len(batch_losses)

    print('Epoch: {: d}, loss {:5.5f}, train time {:.2f} '.format(epoch_cnt, epoch_loss, time() - t_start, ))
    return epoch_loss



def preprocess(idx, seed, args, path=None):
    try:
        if path!=None:
            print(f'idx {idx} begin')
            st = time()
        com = train_comms[idx]
        labels = [0] * len(com)
        labels[-1] = 1
        # seed =  com[0]
        left = set(com) - set([seed])
        bfs_com = [seed]
        bfs_com_set = set(bfs_com)
        neighbor = []
        next_nodes = []
        batch_now_com = [[seed]]
        neighbor.append(graph.neighbors[seed])
        while left:
            union = neighbor[-1] & left
            assert union != set()
            left -= union
            union = list(union)
            while union:
                if False and args.rollouts != 1:
                    random.shuffle(union)
                top = union.pop()
                bfs_com.append(top)
                bfs_com_set.add(top)
                batch_now_com.append(batch_now_com[-1] + [top])
                neighbor.append((graph.neighbors[top] | neighbor[-1]) - bfs_com_set)
                next_nodes.append([top])

        next_nodes.append([-1])
        for i in range(len(neighbor)):
            next_node = next_nodes[i]

            if (len(neighbor[i]) > args.max_neighbor):
                neighbor[i] = np.random.choice(list(neighbor[i] - set(next_node)),
                                             args.max_neighbor - len(next_node)).tolist() + next_node


            neighbor[i] = sorted(list(neighbor[i]))

        neighbor_map = [{node: i for i, node in enumerate(ner)} for ner in neighbor]
        next_onehot = np.zeros((len(next_nodes), args.max_neighbor + 1))
        for i in range(len(next_nodes) - 1):
            tmpnext =   next_nodes[i]
            for j in tmpnext:
                next_onehot[i][neighbor_map[i][j]] = 1

        next_onehot[-1][args.max_neighbor] = 1
        batch_z_seed = []
        batch_z_node = []
        z_seed = make_single_node_encoding([seed], graph,conv)

        z_node = make_nodes_encoding([[seed]] + [node for node in next_nodes[:-1]], graph,conv)
        st = time()
        for i in range(1, len(next_nodes)):
            z_node[i] += z_node[i - 1]
        for i in range(len(next_nodes)):
            batch_z_seed.append(torch.from_numpy(z_seed[0, (batch_now_com[i]) + neighbor[i]].todense()))
            batch_z_node.append(torch.from_numpy(z_node[i, (batch_now_com[i]) + neighbor[i]].todense()))
            assert batch_z_node[-1].shape[1] == len(batch_now_com[i]) + len(neighbor[i]) and batch_z_seed[-1].shape[1] == len(
                batch_now_com[i]) + len(neighbor[i])
        assert len(neighbor) == len(batch_now_com)
        result = {"neighbor": neighbor,
                  "next_onehot": next_onehot,
                  "now_com": batch_now_com,
                  "z_seed": batch_z_seed,
                  "z_node": batch_z_node,
                  "label": labels
                  }
        if args.augment:
            batch_z_augment = []
            z_augment = get_augment([seed],graph,nxg,conv)
            for i in range(len(next_nodes)):
                batch_z_augment.append(torch.from_numpy(z_augment[0, (batch_now_com[i]) + neighbor[i]].todense()))
            result["z_augment"] = batch_z_augment
        if args.rankingloss:
            neg_neighbor = neighbor[-1][:]
            if -1 in neg_neighbor:
                neg_neighbor.remove(-1)
            if len(neg_neighbor) < args.neg_num:
                neg_neighbor += np.random.choice(list(graph.nodes - set(com)), args.neg_num - len(neg_neighbor)).tolist()
            batch_neg = np.random.choice(neg_neighbor, args.neg_num).tolist()
            assert len(batch_neg) >= args.neg_num
            neg_z_node = make_nodes_encoding([[node] for node in batch_neg], graph,conv)
            for i in range(neg_z_node.shape[0]):
                neg_z_node[i] += z_node[-1]
            batch_neg_z_node = []
            batch_neg_z_seed = []
            neg_neighbors = []
            neg_batch_z_augment = []
            for i in range(len(batch_neg)):
                neg = batch_neg[i]
                neg_neighbor = list(set(neg_neighbor) | graph.neighbors[neg] - set(com + [neg]))
                if (len(neg_neighbor) > args.max_neighbor):
                    neg_neighbor = np.random.choice(neg_neighbor, args.max_neighbor).tolist()

                batch_neg_z_seed.append(torch.from_numpy(z_seed[0, com + [neg] + sorted(
                    neg_neighbor)].todense()))
                batch_neg_z_node.append(torch.from_numpy(neg_z_node[i, com + [neg] + sorted(neg_neighbor)].todense()))
                neg_neighbors.append(neg_neighbor)
                if args.augment:
                    neg_batch_z_augment.append(torch.from_numpy(z_augment[0, com + [neg] + sorted(neg_neighbor)].todense()))

            result["neg"] = batch_neg
            result["neg_neighbors"] = neg_neighbors
            result["neg_z_seed"] = batch_neg_z_seed
            result["neg_z_node"] = batch_neg_z_node
            result["neg_z_augment"] = neg_batch_z_augment

        if path is not None:
            np.save(path + f'_{idx}_{seed}.npy', result)
            print(idx, seed, "time: %0.2f" % (time() - st))
        else:
            return result
    except Exception as ex:
        print(ex)
        print('try to use parameters --multiprocessing to close multiprocessing!')





def load_pre_dataset(args,train_comms):

    pre_dataset = []
    roll_mapper = {}
    arg_dict = vars(args)
    pretrain_related_args = ['dataset', 'train_size', 'community_min', 'augment',
                             'max_neighbor', 'rollouts',
                             'rankingloss', 'neg_num']
    code = ' '.join([str(arg_dict[k]) for k in pretrain_related_args])
    code = hashlib.md5(code.encode('utf-8')).hexdigest().upper()
    path = f'./preprocess/{code}'
    name = path + '.npy'
    if os.path.exists(name) and args.cache == True:
        print(f'{name} exists.' )
        pre_dataset = np.load(name, allow_pickle=True).tolist()
    else:
        if os.path.exists('./preprocess')==False:
            os.mkdir('./preprocess')
        print(f'{name}  not exists.' )
        st = time()
        if args.multiprocessing:
            po = Pool(cpu_count())
            print('cpu count',cpu_count())
            for i in range(len(train_comms)):
                for j in range(args.rollouts):
                    if len(train_comms[i]) > j:
                        seed = train_comms[i][j]
                        po.apply_async(preprocess, (i, seed,args, path))
            po.close()
            po.join()
            for i in tqdm(range(len(train_comms))):
                roll_mapper[i] = []
                for j in range(args.rollouts):
                    if len(train_comms[i]) > j:
                        seed = train_comms[i][j]
                        result = np.load(path + f'_{i}_{seed}.npy', allow_pickle=True).tolist()
                        os.remove(path + f'_{i}_{seed}.npy')
                        roll_mapper[i].append(len(pre_dataset))
                        pre_dataset.append(result)
            pre_dataset.append(roll_mapper)
            np.save(name, pre_dataset, allow_pickle=True)

        else:
            print('cpu count', 1)
            for i in tqdm(range(len(train_comms))):
                roll_mapper[i] = []
                for j in range(args.rollouts):
                    if len(train_comms[i]) > j:
                        seed = train_comms[i][j]
                        result = preprocess(i, seed,args)
                        roll_mapper[i].append(len(pre_dataset))
                        pre_dataset.append(result)
            pre_dataset.append(roll_mapper)
            np.save(name, pre_dataset, allow_pickle=True)
        print('end preprocess', time() - st)
    return pre_dataset[:-1], pre_dataset[-1]


def set_seed(seed, cuda=False):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    print('set seed for random numpy and torch')



def main(args):

    global train_comms
    global graph
    global nxg
    global conv
    args.cuda = torch.cuda.is_available()
    set_seed(args.random_seed, args.cuda)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph, train_comms, valid_comms, test_comms, eval_seeds, nodefeats, ds_name, nxg = load_data(args)
    conv = GraphConv(graph)

    if nodefeats == None or args.with_feature == False:
        attr = None
    else:
        attr = preprocess_nodefeats(conv, nodefeats, args.nfeat)
        attr = torch.from_numpy(attr).to(device).float()

    pre_dataset, roll_mapper = load_pre_dataset(args,train_comms)
    now = datetime.datetime.now()
    save_path = './result/' + now.strftime("%Y%m%d%H%M%S")
    if os.path.exists(save_path) == False:
        os.mkdir(save_path)

    model = CSAFModel(args=args,
                      device=device,
                      ).to(device)
    train_dataloader = DataLoader(PretrainComDataset(roll_mapper, len(train_comms)),
                                  batch_size=args.batch_size,
                                  shuffle=args.suffle,
                                  num_workers=0)
    with open(args.log, 'a+') as f:
        print(f'save_path ={save_path}',args, file=f)
        print(f'save_path ={save_path}')

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=0)
    start_epoch = 0


    total_loss = []
    start_epoch = start_epoch
    print('start fitting.')
    best_f = -1
    best_e = 0
    for epoch in (range(args.epoch)):
        epoch_loss = train_epoch(args,device,epoch,model,train_dataloader,optimizer,roll_mapper,pre_dataset,attr)
        total_loss.append(epoch_loss)
        if epoch > args.cnt_rank:
            st = time()
            coms = generate_batch(model, args, graph, conv, attr, eval_seeds[:args.valid_size], nxg,neribor_max=args.max_neighbor,
                                  stop=args.stopmode, decline=args.m_s)
            st = time() - st
            avglen = sum([len(com) for com in coms]) / len(coms)
            rf, rj = calu(coms, valid_comms[:args.valid_size])  # train_comms

            f, j = calu(valid_comms + train_comms, coms)
            print(f'ff:{f:.4f} bf:{rf:.4f} f:{(f + rf) / 2:.4f} avglen:{avglen:.2f} sample time {st:.2f}')


            if (rf + f) / 2 > best_f:
                with open(args.log, 'a+') as file:
                    print(f' dataset = {args.dataset} ', epoch, args.dataset, f, rf, (rf + f) / 2, file=file)
                best_f = (rf + f) / 2
                best_e = epoch
                print('save')
                var_list = {
                            'best_f': best_f,
                            'args': args,
                            }
                save_model(model, optimizer, args, var_list, save_path, epoch=epoch + start_epoch)

    with open(args.result, 'a') as f:
        print(args,file=f)
        print(f'save_path ={save_path} best_f,p,r,e\n', best_f, best_e, file=f)
        print('best_f,p,r,e', best_f, best_e)



def eval(checkpoint):

    args = checkpoint['args']
    print('args', args)
    global train_comms
    global graph
    global nxg
    global conv
    args.cuda = torch.cuda.is_available()
    set_seed(args.random_seed, args.cuda)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph, train_comms, valid_comms, test_comms, eval_seeds, nodefeats, ds_name, nxg = load_data(args)
    conv = GraphConv(graph)

    if nodefeats == None or args.with_feature == False:
        attr = None
    else:
        attr = preprocess_nodefeats(conv, nodefeats, args.nfeat)
        attr = torch.from_numpy(attr).to(device).float()



    model = CSAFModel(args=args,
                      device=device,
                      ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    st = time()
    coms = generate_batch(model, args, graph, conv, attr, eval_seeds, nxg,
                          neribor_max=args.max_neighbor,
                          stop=args.stopmode, decline=args.m_s)
    st = time() - st
    avglen = sum([len(com) for com in coms]) / len(coms)
    rf, rj = calu(coms, valid_comms + train_comms)
    f, j = calu(valid_comms + train_comms, coms)
    print(f'ff:{f:.4f} bf:{rf:.4f} f:{(f + rf) / 2:.4f} avglen:{avglen:.2f} time {st:.2f}')


if __name__ == '__main__':

    add_hour = datetime.datetime.now() + datetime.timedelta(hours=1)
    now_hour = add_hour.strftime('%Y-%m-%d-%H')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='facebook')
    parser.add_argument('--root', type=str, default='datasets')
    parser.add_argument('--train_size', type=int, default=450)
    parser.add_argument('--valid_size', type=int, default=200)
    parser.add_argument('--community_min', type=int, default=3)
    parser.add_argument('--max_neighbor', type=int, default=200)
    parser.add_argument('--eval_path', type=str, default="")
    parser.add_argument('--eval_seed_mode', type=str, default="max")
    parser.add_argument('--random_seed', type=int, default=1)
    #########
    parser.add_argument('--cnt_rank', type=int, default=0)
    parser.add_argument('--num_flow_layer', type=int, default=8)
    parser.add_argument('--dropout', type=int, default=0)
    parser.add_argument('--suffle', action='store_false', default=True)
    parser.add_argument('--rollouts', type=int, default=1)
    parser.add_argument('--augment', action='store_true', default=False)
    parser.add_argument('--neg_num', type=int, default=10)
    #########
    parser.add_argument('--nfeat', type=int, default=128)
    parser.add_argument('--nhid', type=int, default=128)
    parser.add_argument('--nout', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=80)
    parser.add_argument('--single_mlp', action='store_false', default=True)
    parser.add_argument('--is_batchNorm', action='store_true', default=False)
    parser.add_argument('--with_feature', action='store_true', default=False)
    parser.add_argument('--normalization', type=str, default=None)
    parser.add_argument('--is_bn_before', action='store_true', default=False)
    parser.add_argument('--att_mode', type=int, default=1)
    parser.add_argument('--init_emb', action='store_true', default=False)
    parser.add_argument('--com_feature', action='store_false', default=True)
    parser.add_argument('--rankingloss', action='store_false', default=True)
    parser.add_argument('--best_score', action='store_false', default=True)
    parser.add_argument('--m_e', type=int, default=1)
    parser.add_argument('--m_s', type=int, default=2)
    parser.add_argument('--gamma', type=int, default=1)
    parser.add_argument('--squrank', action='store_true', default=False)
    parser.add_argument('--margin', type=int, default=0)
    parser.add_argument('--reduce', type=str, default="sum")
    ##
    parser.add_argument('--log', type=str, default='log.txt')
    parser.add_argument('--result', type=str, default='result.txt')
    parser.add_argument('--cache',action='store_false', default=True)
    parser.add_argument('--rule', action='store_true', default=False)
    parser.add_argument('--multiprocessing', action='store_false', default=True)
    parser.add_argument('--stopmode', type=int, default=1)
    parser.add_argument('--chunk_size', type=int, default=2000)
    #
    parser.add_argument('--default_parameters', action='store_true', default=False)

    args = parser.parse_args( )  # facebook
    if os.path.exists('./result') == False:
        os.mkdir('result')
    if args.eval_path=="":
        if args.default_parameters:
            datasets = ['facebook', 'dblp', 'amazon', 'twitter', 'youtube']
            cnt_rank = [0, 40, 40, 0, 20]
            num_flow_layers = [4, 16, 8, 8, 4]
            dropouts = [0.5, 0, 0, 0, 0.2]
            suffle = [True, False, False, True, True]
            rollout = [2, 1, 1, 1, 1]
            augment = [False, True, False, False, False]
            negnum = [10, 10, 5, 20, 10]
            m_e_list = [5, 3, 5, 3, 1]
            m_s_list = [2, 5, 2, 2, 1]
            for dataset, args.cnt_rank, args.num_flow_layer, args.dropout, args.suffle, args.rollouts, args.augment, args.neg_num,args.m_s,args.m_e in \
                    zip(datasets, cnt_rank, num_flow_layers, dropouts, suffle, rollout, augment, negnum,m_s_list,m_e_list):
                if args.dataset == 'facebook':
                    args.train_size = 28
                    args.valid_size = 2
                else:
                    args.train_size = 450
                    args.valid_size = 50
                if args.dataset == dataset:
                    break
        print(args)
        main(args)
    else:
        print('eval')
        if os.path.exists(args.eval_path):
            checkpoint = torch.load(args.eval_path)
            eval(checkpoint)
        else:
            print(f'{args.eval_path} not exists!')
