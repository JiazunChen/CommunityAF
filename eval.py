import json
import os
import numpy as np
import networkx as nx
import torch
from time import time
from utils import  get_augment,make_single_node_encoding,make_nodes_encoding

def save_model(model, optimizer, args, var_list, save_path, epoch=None):
    argparse_dict = vars(args)
    with open(os.path.join(save_path, 'config.json'), 'w') as f:
        json.dump(argparse_dict, f)

    epoch = str(epoch) if epoch is not None else ''
    latest_save_path = os.path.join(save_path, 'checkpoint')
    final_save_path = os.path.join(save_path, 'checkpoint%s' % epoch)
    torch.save({
        **var_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        final_save_path
    )


    torch.save({
        **var_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        latest_save_path
    )



def initialize_from_checkpoint(init_checkpoint,model):
    checkpoint = torch.load(init_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    return checkpoint['args']

def generate_batch(model, args, graph, conv, attr, seeds, nxg,temperature=0.75, neribor_max=100, debug=0, stop=0, decline=1):
    old_seeds = seeds
    n = len(old_seeds)
    model.eval()
    batch_coms = []
    chunk_size = args.chunk_size

    for k in range((n // chunk_size) + 1):
        seeds = old_seeds[k * chunk_size:(k + 1) * chunk_size]
        seeds = [s for s in seeds for _ in range(args.m_e)]
        n_seed = len(seeds)
        if n_seed == 0:
            continue
        with torch.no_grad():
            prior_node_dist = [torch.distributions.normal.Normal(torch.zeros([(neribor_max + 1)]),
                                                                 temperature * torch.ones((neribor_max + 1))) for i in
                               range(n_seed)]

            batch_scores = [[-1] * decline for i in range(n_seed)]
            batch_nowcom = [[seed] for seed in seeds]
            batch_neribor = [list(graph.neighbors[seed]) for seed in seeds]
            batch_bestcom = [None for i in range(n_seed)]
            bacth_bests = [-2 for i in range(n_seed)]
            z_seeds = make_single_node_encoding([seed for seed in seeds], graph,conv)
            z_nodes = make_single_node_encoding([seed for seed in seeds], graph,conv)
            batch_z_seed = [0] * n_seed
            batch_z_node = [0] * n_seed
            if args.augment:
                batch_z_augment = [0] * n_seed
                z_augment = get_augment(seeds,graph,nxg,conv)
            batch_ok = [0] * n_seed
            ok = 0
            latent_nodes = [0] * n_seed
            for size in (range(args.max_size)):
                if (ok == n_seed):
                    break
                for i in (range(n_seed)):
                    if batch_ok[i] == 1:
                        continue
                    latent_nodes[i] = prior_node_dist[i].sample().view(1, -1)
                    if len(batch_neribor[i]) > neribor_max:
                        batch_neribor[i] = np.random.choice(batch_neribor[i], neribor_max)

                    batch_neribor[i] = sorted(batch_neribor[i])
                    batch_z_seed[i] = torch.from_numpy(z_seeds[i, (batch_nowcom[i]) + batch_neribor[i]].todense())
                    batch_z_node[i] = torch.from_numpy(z_nodes[i, (batch_nowcom[i]) + batch_neribor[i]].todense())
                    if args.augment:
                        batch_z_augment[i] = torch.from_numpy(
                            z_augment[i, (batch_nowcom[i]) + batch_neribor[i]].todense())

                latent_node = torch.stack(latent_nodes).view(n_seed, -1)
                batch_data = {"batch_com": batch_nowcom,
                              "batch_neighbor": batch_neribor,
                              "batch_deq": latent_node,
                              "batch_z_seed": batch_z_seed,
                              "batch_z_node": batch_z_node
                              }
                if args.augment:
                    batch_data['batch_z_augment'] = batch_z_augment

                result = model.flow_core.batch_revser(attr, batch_data)
                out_z = result['out_z']
                newnodes = []
                for i in range(n_seed):
                    if batch_ok[i] == 1:
                        newnodes.append(None)
                        continue
                    latent_node = out_z[i]
                    neribor_num = len(batch_neribor[i])


                    if stop == 1:
                        score = result['stop_score'][i].item()
                        if args.rule:
                            score = 1 - nx.conductance(nxg, batch_nowcom[i])
                        if score > bacth_bests[i]:
                            bacth_bests[i] = score
                            batch_bestcom[i] = batch_nowcom[i]
                        flag = 0
                        for d in range(decline):
                            if score < batch_scores[i][-1 - d]:
                                flag += 1
                        if flag == decline and len(batch_nowcom[i]) >= args.community_min:
                            newnodes.append(None)
                            batch_ok[i] = 1
                            ok += 1
                        elif neribor_num == 0:
                            newnodes.append(None)
                            batch_ok[i] = 1
                            ok += 1
                        elif flag < decline:

                            index = torch.argmax(latent_node[:neribor_num]).item()
                            newnode = batch_neribor[i][index]
                            batch_nowcom[i].append(newnode)
                            batch_neribor[i].extend(list(graph.neighbors[newnode]))
                            batch_neribor[i] = list(set(batch_neribor[i]) - set(batch_nowcom[i]))
                            batch_scores[i].append(score)
                            newnodes.append(newnode)
                        else:
                            batch_scores[i].append(score)
                            newnodes.append(None)
                    else:
                        if neribor_num != 0:
                            index = torch.argmax(latent_node[:neribor_num]).item()
                            newnode = batch_neribor[i][index]
                        else:
                            index = None
                        if index == None or (
                                latent_node[-1] > latent_node[index] and len(batch_nowcom[i]) >= args.community_min):
                            newnodes.append(None)
                            batch_ok[i] = 1
                            ok += 1
                        elif latent_node[-1] <= latent_node[index]:
                            batch_nowcom[i].append(newnode)
                            batch_neribor[i].extend(list(graph.neighbors[newnode]))
                            batch_neribor[i] = list(set(batch_neribor[i]) - set(batch_nowcom[i]))
                            newnodes.append(newnode)
                        else:
                            newnodes.append(None)

                z_nodes += make_single_node_encoding(newnodes, graph,conv)
            coms = []
            for i in range(0, len(batch_nowcom), args.m_e):
                if stop == 1:
                    bestc = None
                    bests = -1
                    for j in range(args.m_e):
                        com = batch_nowcom[i + j]
                        s = bacth_bests[i + j]
                        if args.best_score == True:
                            com = batch_bestcom[i + j]
                            s = bacth_bests[i + j]
                        if s > bests:
                            bestc = com
                    coms.append(bestc)
                else:
                    coms.append(batch_nowcom[i])
        batch_coms.extend(coms)
    return batch_coms

def calu(coms, valid):
    n_valid = len(valid)

    brf, brj = 0, 0
    i = 0
    for precom in valid:
        i += 1
        f, j = 0, 0
        for right_com in coms:
            right = len(set(right_com) & set(precom))
            nr = right / len(right_com)
            np = right / len(precom)
            if nr + np != 0:
                tmpf = 2 * (nr * np) / (nr + np)
                if tmpf > f:
                    f = tmpf
        brf += f
    return brf / n_valid, 0

