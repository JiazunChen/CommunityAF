import torch
import torch.nn as nn

class Rescale(nn.Module):
    def __init__(self):
        super(Rescale, self).__init__()
        self.weight = nn.Parameter(torch.zeros([1]))

    def forward(self, x):
        if torch.isnan(torch.exp(self.weight)).any():
            print(self.weight)
            raise RuntimeError('Rescale factor has NaN entries')

        x = torch.exp(self.weight) * x
        return x


class ST_Net_Exp(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim=64, num_layers=2, bias=True, scale_weight_norm=False,
                 sigmoid_shift=2.,
                 apply_batch_norm=False):
        super(ST_Net_Exp, self).__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.bias = bias
        self.apply_batch_norm = apply_batch_norm
        self.scale_weight_norm = scale_weight_norm
        self.sigmoid_shift = sigmoid_shift
        self.linear1 = nn.Linear(input_dim, hid_dim, bias=bias)
        self.linear2 = nn.Linear(hid_dim, output_dim * 2, bias=bias)
        if self.apply_batch_norm:
            self.bn1 = nn.BatchNorm1d(input_dim)
        if self.scale_weight_norm:
            self.rescale1 = nn.utils.weight_norm(Rescale())

        else:
            self.rescale1 = Rescale()
        self.tanh = nn.Tanh()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.constant_(self.linear2.weight, 1e-10)
        if self.bias:
            nn.init.constant_(self.linear1.bias, 0.)
            nn.init.constant_(self.linear2.bias, 0.)

    def forward(self, x):
        if self.apply_batch_norm:
            x = self.bn1(x)

        x = self.linear2(self.tanh(self.linear1(x)))
        s = x[:, :self.output_dim]
        t = x[:, self.output_dim:]
        s = self.rescale1(torch.tanh(s))
        return s, t


class SelfAttnPooling(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.score_layer = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x):
        w = self.score_layer(x)
        w = torch.softmax(w, -2)
        return (x * w).sum(-2, keepdim=True)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Sigmoid(nn.Module):
    def forward(self, x):
        return torch.sigmoid(x)


def make_linear_block(in_size, out_size, act_cls=None, norm_type=None, bias=True, residual=True, dropout=0.):
    return LinearBlock(in_size, out_size, act_cls, norm_type, bias, residual, dropout)


class LinearBlock(nn.Module):
    def __init__(self, in_size, out_size, act_cls=None, norm_type=None, bias=True, residual=True, dropout=0.):
        super().__init__()
        self.residual = residual and (in_size == out_size)
        layers = []
        if norm_type == 'batch_norm':
            layers.append(nn.BatchNorm1d(in_size))
        elif norm_type == 'layer_norm':
            layers.append(nn.LayerNorm(in_size))
        elif norm_type is not None:
            raise NotImplementedError
        if act_cls is not None:
            layers.append(act_cls())
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(in_size, out_size, bias))
        self.f = nn.Sequential(*layers)

    def forward(self, x):
        z = self.f(x)
        if self.residual:
            z += x
        return z

class coreModel(nn.Module):
    def __init__(self, args,device,neribor=120, num_flow_layer=12, nfeat=128, nhid=128, nout=128, is_batchNorm=False, gcn_layer=3,
                 dropout=0.0,
                 normalization='batch_norm', single_mlp=False, att_mode=0, init_emb=True):

        super(coreModel, self).__init__()
        self.device = device
        self.is_batchNorm = is_batchNorm
        self.init_emb = init_emb

        self.seed_embedding = nn.Linear(1, nfeat, bias=False)
        self.node_embedding = nn.Linear(1, nfeat, bias=False)
        self.args = args

        if self.args.augment:
            self.augment_embedding = nn.Linear(1, nfeat, bias=False)
        self.attr_embedding = nn.Linear(nfeat, nfeat)
        self.input_mapping = nn.Sequential(
            make_linear_block(nfeat, nout, Swish, normalization, dropout=dropout),
            make_linear_block(nout, nout, Swish, normalization, dropout=dropout))
        if self.args.rankingloss:
            self.stop_mlp = nn.Sequential(
                make_linear_block(nfeat * 2, nout, Swish, None),
                make_linear_block(nout, nout, Swish, None),
                make_linear_block(nout, 2, None, None),
            )

        self.emb_size = (neribor + 1) * nout
        self.nout = nout
        self.num_flow_layer = num_flow_layer
        self.gcn_layer = gcn_layer
        self.dropout = dropout
        self.normalization = normalization
        self.neribor = neribor
        self.nhid = nhid
        self.single_mlp = single_mlp
        self.att_mode = att_mode
        if self.is_batchNorm:
            self.batchNorm = nn.BatchNorm1d(nout)

        self.pool = SelfAttnPooling(nout)
        st_net_fn = ST_Net_Exp
        if self.single_mlp:
            if self.args.com_feature:
                self.node_st_net = nn.ModuleList(
                    [st_net_fn(self.nout * 2, 1, hid_dim=nhid, bias=True, apply_batch_norm=self.args.is_bn_before) for i in
                     range(self.num_flow_layer)])
            else:
                self.node_st_net = nn.ModuleList(
                    [st_net_fn(self.nout, 1, hid_dim=nhid, bias=True, apply_batch_norm=self.args.is_bn_before) for i in
                     range(self.num_flow_layer)])
        else:
            self.node_st_net = nn.ModuleList(
                [st_net_fn(self.emb_size, self.neribor + 1, hid_dim=nhid, bias=True) for i in
                 range(self.num_flow_layer)])

    def get_embs(self, x, adj):
        node_emb = self.gcn(x, adj)
        if self.is_batchNorm:
            node_emb = self.batchNorm(node_emb)
        return node_emb

    def forward(self, attr, batch_data):
        batch_com, batch_neighbor, batch_next_onehot, batch_z_seed, batch_z_node, batch_labels = batch_data["batch_com"], \
                                                                                               batch_data[
                                                                                                   "batch_neighbor"], \
                                                                                               batch_data[
                                                                                                   "batch_next_onehot"], \
                                                                                               batch_data[
                                                                                                   "batch_z_seed"], \
                                                                                               batch_data[
                                                                                                   "batch_z_node"], \
                                                                                               batch_data[
                                                                                                   "batch_labels"]
        if self.args.augment:
            batch_z_augment = batch_data["batch_z_augment"]
        x_deq = torch.tensor(batch_next_onehot, dtype=torch.float).to(self.device)
        repeat = x_deq.size(0)
        assert repeat == len(batch_z_seed)
        mlp_emb = torch.zeros((repeat * (self.neribor + 1), self.nout), dtype=torch.float).to(self.device)
        if self.args.com_feature:
            global_emb = torch.zeros((repeat * (self.neribor + 1), self.nout), dtype=torch.float).to(self.device)
        if self.args.rankingloss:
            labels = []
            coms = []
            negcoms = []
            embs = []
            com = []
            negcom = []
            batch_neg = batch_data["batch_neg"]
            batch_neg_neighbors = batch_data["batch_neg_neighbors"]
            batch_neg_z_seed = batch_data["batch_neg_z_seed"]
            batch_neg_z_node = batch_data["batch_neg_z_node"]
            if self.args.augment:
                batch_neg_z_augment = batch_data["batch_neg_z_augment"]
            neg_idx = 0
        for i in range(repeat):
            z_seed = batch_z_seed[i].to(self.device)
            z_node = batch_z_node[i].to(self.device)
            h = self.seed_embedding(z_seed.view(-1, 1)) + self.node_embedding(z_node.view(-1, 1))
            if self.args.augment:
                z_augment = batch_z_augment[i].to(self.device)
                h += self.augment_embedding(z_augment.view(-1, 1))
            assert h.shape[0] == len(batch_com[i]) + len(batch_neighbor[i])
            if attr != None:
                h += self.attr_embedding(attr[(batch_com[i]) + sorted(batch_neighbor[i])])
            now = self.input_mapping(h)
            if self.att_mode == 1:
                com_emb = self.pool(torch.unsqueeze(now, 0)).view(-1)
            elif self.att_mode == 2:
                com_emb = self.pool(torch.unsqueeze(now, 0)).view(-1)
            elif self.att_mode == 3:
                com_emb = now.mean(dim=0)
            elif self.att_mode == 0:
                com_emb = now[:len(batch_com[i])].mean(dim=0)
            elif self.att_mode == 4:
                com_emb = self.pool(torch.unsqueeze(now[:len(batch_com[i])][-self.args.xia:], 0)).view(-1)
            ner_emb = now[len(batch_com[i]):]
            plot = i * (self.neribor + 1)
            next_plot = (i + 1) * (self.neribor + 1)
            mlp_emb[plot:plot + len(batch_neighbor[i])] = ner_emb
            mlp_emb[next_plot - 1] = com_emb

            if self.args.com_feature:
                if self.att_mode == 2:
                    global_emb[plot:next_plot] = now.max(dim=0).repeat(self.neribor + 1, 1)
                else:
                    global_emb[plot:next_plot] = now.mean(dim=0).repeat(self.neribor + 1, 1)
            if self.args.rankingloss:

                com.append(len(embs))
                labels.append(batch_labels[i])
                embs.append(torch.cat((com_emb, now.mean(dim=0))))
                if batch_labels[i] == 1:
                    for j in range(len(batch_neg[neg_idx])):
                        labels.append(0)
                        neg_z_seed = batch_neg_z_seed[neg_idx][j].to(self.device)
                        neg_z_node = batch_neg_z_node[neg_idx][j].to(self.device)
                        h = self.seed_embedding(neg_z_seed.view(-1, 1)) + self.node_embedding(neg_z_node.view(-1, 1))
                        if self.args.augment:
                            neg_z_augment = batch_neg_z_augment[neg_idx][j].to(self.device)
                            h += self.augment_embedding(neg_z_augment.view(-1, 1))
                        if attr != None:
                            h += self.attr_embedding(attr[(batch_com[i] + [batch_neg[neg_idx][j]]) + sorted(
                                batch_neg_neighbors[neg_idx][j])])
                        neg_now = self.input_mapping(h)

                        negcom.append(len(embs))
                        embs.append(torch.cat(
                            (self.pool(torch.unsqueeze(neg_now, 0)).view(-1), neg_now.mean(dim=0))))  # 社区内部cat社区外部
                    negcoms.append(negcom)
                    negcom = []
                    coms.append(com)
                    com = []
                    neg_idx += 1

        if self.args.com_feature:
            mlp_emb = torch.cat((mlp_emb, global_emb), 1)
        for i in range(self.num_flow_layer):
            node_s, node_t = self.node_st_net[i](mlp_emb)
            node_s = node_s.view(repeat, -1).to(self.device)
            node_t = node_t.view(repeat, -1).to(self.device)
            node_s = node_s.exp()
            x_deq = (x_deq + node_t) * node_s
            if torch.isnan(x_deq).any():
                raise RuntimeError(
                    'x_deq has NaN entries after transformation at layer %d' % i)
            if i == 0:
                x_log_jacob = (torch.abs(node_s) + 1e-20).log()
            else:
                x_log_jacob += (torch.abs(node_s) + 1e-20).log()
        x_log_jacob = x_log_jacob.view(repeat, -1).sum(-1)  # (batch)
        result = {
            'out_z': x_deq,
            'out_logdet': x_log_jacob
        }
        if self.args.rankingloss:
            prediction = torch.nn.functional.softmax(self.stop_mlp(torch.stack(embs)), dim=1)
            stop_score = prediction[:, 1]
            pospair = []
            negpair = []
            assert len(coms) == len(negcoms)
            assert len(labels) == len(prediction)
            for com, neg in zip(coms, negcoms):
                for i in range(1, len(com)):
                    pospair.append(com[i])
                    negpair.append(com[i - 1])
                for i in range(len(neg)):
                    pospair.append(com[-1])
                    negpair.append(neg[i])

            result['stop_score'] = stop_score
            result['pospair'] = pospair
            result['negpair'] = negpair
            result['prediction'] = prediction
            result['label'] = labels
        return result

    def batch_revser(self, attr, batch_data):

        batch_com, batch_neighbor, x_deq, batch_z_seed, batch_z_node = batch_data["batch_com"], batch_data[
            "batch_neighbor"], \
                                                                     batch_data["batch_deq"], batch_data[
                                                                         "batch_z_seed"], \
                                                                     batch_data["batch_z_node"]

        if self.args.augment:
            batch_z_augment = batch_data["batch_z_augment"]
        repeat = x_deq.size(0)
        x_deq = x_deq.to(self.device)
        assert repeat == len(batch_z_seed)
        mlp_emb = torch.zeros((repeat * (self.neribor + 1), self.nout), dtype=torch.float).to(self.device)
        if self.args.com_feature:
            global_emb = torch.zeros((repeat * (self.neribor + 1), self.nout), dtype=torch.float).to(self.device)

        if self.args.rankingloss:
            embs = []
        for i in range(repeat):
            z_seed = batch_z_seed[i].to(self.device)
            z_node = batch_z_node[i].to(self.device)
            h = self.seed_embedding(z_seed.view(-1, 1)) + self.node_embedding(z_node.view(-1, 1))
            if self.args.augment:
                z_augment = batch_z_augment[i].to(self.device)
                h += self.augment_embedding(z_augment.view(-1, 1))
            assert h.shape[0] == len(batch_com[i]) + len(batch_neighbor[i])
            if attr != None:
                h += self.attr_embedding(attr[(batch_com[i]) + sorted(batch_neighbor[i])])  # 前面是当前社区 后面是邻居的特征
            now = self.input_mapping(h)

            if self.att_mode == 1:
                com_emb = self.pool(torch.unsqueeze(now, 0)).view(-1)
            elif self.att_mode == 2:
                com_emb = self.pool(torch.unsqueeze(now, 0)).view(-1)
            elif self.att_mode == 3:
                com_emb = now.mean(dim=0)
            elif self.att_mode == 0:
                com_emb = now[:len(batch_com[i])].mean(dim=0)
            elif self.att_mode == 4:
                com_emb = self.pool(torch.unsqueeze(now[:len(batch_com[i])][-self.args.xia:], 0)).view(-1)

            if self.args.rankingloss:

                embs.append(torch.cat((com_emb, now.mean(dim=0))))

            ner_emb = now[len(batch_com[i]):]
            plot = i * (self.neribor + 1)
            next_plot = (i + 1) * (self.neribor + 1)
            mlp_emb[plot:plot + len(batch_neighbor[i])] = ner_emb
            mlp_emb[next_plot - 1] = com_emb
            if self.args.com_feature:
                if self.att_mode == 2:
                    global_emb[plot:next_plot] = now.max(dim=0).repeat(self.neribor + 1, 1)
                else:
                    global_emb[plot:next_plot] = now.mean(dim=0).repeat(self.neribor + 1, 1)
        if self.args.com_feature:
            mlp_emb = torch.cat((mlp_emb, global_emb), 1)
        st_net = self.node_st_net
        for i in reversed(range(self.num_flow_layer)):
            s, t = st_net[i](mlp_emb)
            s = s.view(repeat, -1).to(self.device)
            t = t.view(repeat, -1).to(self.device)
            s = s.exp()
            x_deq = (x_deq / s) - t
            if torch.isnan(x_deq).any():
                raise RuntimeError(
                    'x_deq has NaN entries after transformation at layer %d' % i)

        result = {
            'out_z': x_deq,
        }
        if self.args.rankingloss:
            prediction = torch.nn.functional.softmax(self.stop_mlp(torch.stack(embs)), dim=1)
            stop_score = prediction[:, 1]
            result['stop_score'] = stop_score

        return result



class CSAFModel(nn.Module):
    def __init__(self, args,device):
        super(CSAFModel, self).__init__()
        self.args = args
        self.device = device
        self.neriborer = args.max_neighbor
        self.nfeat = args.nfeat
        self.nhid = args.nhid
        self.nout = args.nout
        self.num_flow_layer = args.num_flow_layer
        self.att_mode = args.att_mode
        assert (self.att_mode in [0, 1, 2, 3, 4])
        self.flow_core = coreModel(args = self.args,
                                       device = self.device,
                                        neribor=self.neriborer,
                                       nfeat=self.nfeat,
                                       nhid=self.nhid,
                                       nout=self.nout,
                                       num_flow_layer=self.num_flow_layer,
                                       dropout= args.dropout,
                                       single_mlp=args.single_mlp,
                                       is_batchNorm=args.is_batchNorm,
                                       normalization=args.normalization,
                                       att_mode=args.att_mode,
                                       init_emb=args.init_emb)
        self.prior_ln_var = nn.Parameter(torch.zeros([1]), requires_grad=False)
        self.constant_pi = nn.Parameter(torch.Tensor([3.1415926535]), requires_grad=False)

    def forward(self, attr, batch_data):
        batch_result = self.flow_core(attr, batch_data)

        return batch_result, self.prior_ln_var

    def log_prob(self, z, logdet):
        assert logdet.size(0) == z.size(0)
        latent_node_length = z.size(0) * (self.neriborer + 1)
        logdet = logdet - latent_node_length
        ll_node = -1 / 2 * (
                torch.log(2 * self.constant_pi) + self.prior_ln_var + torch.exp(-self.prior_ln_var) * (z ** 2))
        ll_node = ll_node.sum(-1)
        ll_node += logdet
        final = -(torch.mean(ll_node)) / (latent_node_length)

        return final
