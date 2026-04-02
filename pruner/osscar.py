import math
import time

import numpy as np
import torch
import torch.nn as nn
import transformers
from tqdm import tqdm

from datas import get_examples

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class OSSCAR_prune:

    def __init__(self, layer, nsamples, seqlen, update_iter=1, update_iter2=1, lambda2=1e-2, layername=None,
                 num_heads=32, local_out=False, local_fc=False, local_iter=20, local_test=10, fullseq=False):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.update_iter = update_iter
        self.update_iter2 = update_iter2
        self.lambda2 = lambda2
        self.nsamples = nsamples
        self.seqlen = seqlen
        self.layername = layername
        self.num_heads = num_heads
        self.equi_nsamples = self.nsamples * self.seqlen
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.local_out = local_out
        self.local_fc = local_fc
        self.local_iter = local_iter
        self.local_test = local_test
        self.fullseq = fullseq

        self.XtX = torch.zeros((self.columns, self.columns), device=self.dev)

        self.count = 0

        self.nsamples0 = 0

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

            if len(out.shape) == 3:
                out = out.reshape((-1, out.shape[-1]))
        out = out.t()
        if isinstance(self.layer, nn.Conv2d):
            print(inp.shape)
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        inp = inp.float()
        out = out.float()

        self.nsamples0 += tmp
        if self.fullseq:
            if self.nsamples0 % 2 == 0:
                self.XtX += (self.inp).matmul(inp.t()) / tmp
                self.inp = None
            else:
                self.inp = inp
        else:
            self.XtX += (inp).matmul(inp.t()) / tmp

    def get_XTY(self):

        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        dead = torch.diag(self.XtX) == 0
        B = W.t()
        B[dead, :] = 0

        self.XtX += torch.eye(B.shape[0]).to(device=self.XtX.device) * self.lambda2 * torch.mean(torch.diag(self.XtX))

        self.XtY = self.XtX @ B

        if isinstance(self.layer, transformers.Conv1D):
            self.layer.weight.data = B.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        else:
            self.layer.weight.data = B.t().reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        self.XtX = torch.zeros((self.columns, self.columns), device=self.dev)
        print("number of samples: ", self.nsamples0)

        self.nsamples0 = 0
        W = None
        B = None

    def prune(self, sp_fc, sp_out):

        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        # print("Layer:", self.layer, flush=True)

        st_time = time.time()

        dead = torch.diag(self.XtX) == 0
        B = W.t()
        B[dead, :] = 0

        self.XtX += torch.eye(B.shape[0]).to(device=self.XtX.device) * self.lambda2 * torch.mean(torch.diag(self.XtX))

        if not self.fullseq:
            self.XtY = (self.XtX @ B)
        # print("num of dead is", torch.sum(dead))

        pre_time = time.time() - st_time
        st_time = time.time()

        if self.layername != "self_attn.out_proj":
            num_cin = B.shape[0]
            sp = sp_fc
            upd_iter = self.update_iter
            local_swap = self.update_iter
            if self.local_fc:
                use_local = True
            else:
                use_local = False
        else:
            num_cin = self.num_heads
            sp = sp_out
            upd_iter = self.update_iter2
            local_swap = 5
            if self.local_out:
                use_local = True
            else:
                use_local = False

        # print("sp is {}, num_cin is {}".format(sp, num_cin))

        B_sol, B_obj = OSSCAR_fastprune(B.clone(), self.XtX, self.XtY, num_cin, int(num_cin * (1 - sp)), upd_iter)
        if use_local:
            B_sol, B_obj = OSSCAR_local_search(B_sol, self.XtX, self.XtY, num_cin, self.local_iter, local_swap)

        run_time = time.time() - st_time


        B = B_sol

        # print("num of zeros:", torch.sum(B == 0))
        #
        # print("pre-processing time: ", pre_time, "alg time: ", run_time)
        self.pre_time = pre_time
        self.run_time = run_time

        if isinstance(self.layer, transformers.Conv1D):
            self.layer.weight.data = B.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        else:
            self.layer.weight.data = B.t().reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        return

    def free(self):
        self.H = None
        self.Losses = None
        self.Trace = None
        self.X = None
        self.Y = None
        self.XtX = None
        self.YXt = None
        self.YtX = None
        self.XtY = None
        torch.cuda.empty_cache()


def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def prepare_calibration_input(model, layers, dataloader, device):
    model.seqlen = 2048
    use_cache = model.config.use_cache
    model.config.use_cache = False

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((len(dataloader), model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            if 'attention_mask' in kwargs and kwargs['attention_mask'] is not None:
                cache['attention_mask'] = kwargs['attention_mask']
            if 'position_ids' in kwargs and kwargs['position_ids'] is not None:
                cache['position_ids'] = kwargs['position_ids']
            if "position_embeddings" in kwargs and kwargs['position_embeddings'] is not None:
                cache['position_embeddings'] = kwargs['position_embeddings']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch.unsqueeze(0).to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    # outs = torch.zeros_like(inps)
    model.config.use_cache = use_cache
    del cache["i"]

    return inps, cache


def OSSCAR_fastprune(W, XTX, XTY, num_cin, num_sp, update_iter=1):
    DEV = W.device
    Wtype = W.dtype
    totp, num_cout = W.shape
    ksize = int(totp / num_cin)

    W = W.to(torch.float64)
    XTX = XTX.to(torch.float64)
    XTY = XTY.to(torch.float64)

    XTX_inv = torch.linalg.inv(XTX)

    num_prune = torch.sum(torch.abs(torch.sum(torch.sum(W.reshape(num_cin, ksize, num_cout), axis=2), axis=1)) <= 1e-12)
    prune_list = torch.abs(torch.sum(torch.sum(W.reshape(num_cin, ksize, num_cout), axis=2), axis=1)) <= 1e-12

    if num_prune:
        upd_idx = torch.cat([torch.arange(i * ksize, (i + 1) * ksize) for i in range(num_cin) if prune_list[i]])
        XTX_inv[upd_idx, :] = 0
        XTX_inv[:, upd_idx] = 0

    # W = XTX_inv@ XTY

    if int(num_cin - num_sp - num_prune) <= 0:
        upd_it = 0
    else:
        upd_it = int((num_cin - num_sp - num_prune) / update_iter)
        if upd_it == 0:
            upd_it = 1
        quo, rem = divmod(int(num_cin - num_sp - num_prune), int(upd_it))
        update_ten = torch.full((upd_it,), quo, dtype=torch.int).to(DEV)
        update_ten[:rem] += 1

    for i1 in range(upd_it):

        obj_mat = torch.zeros_like(W)
        if ksize > 1:
            for i2 in range(num_cin):
                if prune_list[i2]:
                    continue
                obj_mat[i2 * ksize:(i2 + 1) * ksize, :] = torch.linalg.inv(
                    XTX_inv[i2 * ksize:(i2 + 1) * ksize, i2 * ksize:(i2 + 1) * ksize]) @ W[i2 * ksize:(i2 + 1) * ksize,
                                                                                         :] / 2
        else:
            obj_mat = (1 / (prune_list + torch.diag(XTX_inv)))[:, None] * W / 2

        obj_cha = W * obj_mat
        obj_cha = obj_cha.reshape(num_cin, ksize, num_cout)
        obj_sum = torch.sum(torch.sum(obj_cha, axis=2), axis=1)

        idx = torch.argsort(obj_sum + 1e20 * (prune_list))

        upd_idx = torch.cat([torch.arange(idx[i] * ksize, (idx[i] + 1) * ksize) for i in range(update_ten[i1])])

        Xinv_tmp = torch.linalg.inv(XTX_inv[upd_idx[:, None], upd_idx])

        W -= XTX_inv[:, upd_idx] @ Xinv_tmp @ W[upd_idx, :]
        W = W.reshape(num_cin, ksize, num_cout)
        W[idx[:update_ten[i1]], :, :] = 0
        W = W.reshape(totp, num_cout)

        XTX_inv -= XTX_inv[:, upd_idx] @ Xinv_tmp @ XTX_inv[upd_idx, :]
        XTX_inv[upd_idx, :] = 0
        XTX_inv[:, upd_idx] = 0

        prune_list[idx[:update_ten[i1]]] = True

    W_sol = torch.zeros_like(W)
    nzi = torch.nonzero(W[:, 0], as_tuple=True)[0]
    W_sol[nzi, :] = torch.linalg.inv(XTX[nzi[:, None], nzi]) @ XTY[nzi, :]

    W_sol = W_sol.to(Wtype)
    XTY = XTY.to(Wtype)
    XTX = XTX.to(Wtype)

    return W_sol, torch.sum(-W_sol * XTY + (1 / 2) * W_sol * (XTX @ W_sol))


def OSSCAR_local_search(W, XTX, XTY, num_cin, max_iter=20, num_swap=100, switch_lb=1):
    DEV = W.device
    Wtype = W.dtype
    totp, num_cout = W.shape

    W = W.to(torch.float64)
    XTX = XTX.to(torch.float64)
    XTY = XTY.to(torch.float64)

    # num_swap = int(np.ceil(num_cin * switch_ratio))
    lb_swap = int(np.ceil(num_cin * switch_lb))

    ksize = int(totp / num_cin)

    prune_list = torch.abs(torch.sum(torch.sum(W.reshape(num_cin, ksize, num_cout), axis=2), axis=1)) <= 1e-12

    best_prune = torch.clone(prune_list)
    supp_idx = torch.cat([torch.arange(i * ksize, (i + 1) * ksize) for i in range(num_cin) if not prune_list[i]])

    XTX_inv = torch.zeros_like(XTX)
    XTX_inv[supp_idx[:, None], supp_idx] = torch.linalg.inv(XTX[supp_idx[:, None], supp_idx])

    obj_cur = torch.sum(-W * XTY + (1 / 2) * W * (XTX @ W))

    for i_local in range(max_iter):

        obj_mat = torch.zeros_like(W)
        if ksize > 1:
            for i2 in range(num_cin):
                if prune_list[i2]:
                    continue
                obj_mat[i2 * ksize:(i2 + 1) * ksize, :] = torch.linalg.inv(
                    XTX_inv[i2 * ksize:(i2 + 1) * ksize, i2 * ksize:(i2 + 1) * ksize]) @ W[i2 * ksize:(i2 + 1) * ksize,
                                                                                         :] / 2
        else:
            obj_mat = (1 / (prune_list + torch.diag(XTX_inv)))[:, None] * W / 2

        obj_cha = W * obj_mat
        obj_cha = obj_cha.reshape(num_cin, ksize, num_cout)
        obj_sum = torch.sum(torch.sum(obj_cha, axis=2), axis=1)

        idx = torch.argsort(obj_sum + 1e20 * (prune_list))

        upd_idx = torch.cat([torch.arange(idx[i] * ksize, (idx[i] + 1) * ksize) for i in range(num_swap)])

        Xinv_tmp = torch.linalg.inv(XTX_inv[upd_idx[:, None], upd_idx])
        W -= XTX_inv[:, upd_idx] @ Xinv_tmp @ W[upd_idx, :]
        W = W.reshape(num_cin, ksize, num_cout)
        W[idx[:num_swap], :, :] = 0
        W = W.reshape(totp, num_cout)

        XTX_inv -= XTX_inv[:, upd_idx] @ Xinv_tmp @ XTX_inv[upd_idx, :]
        XTX_inv[upd_idx, :] = 0
        XTX_inv[:, upd_idx] = 0

        prune_list[idx[:num_swap]] = True

        obj_in = torch.zeros((num_cin,), dtype=torch.float64).to(DEV)

        supp_idx = torch.cat([torch.arange(i * ksize, (i + 1) * ksize) for i in range(num_cin) if not prune_list[i]])
        H_inv = XTX_inv[supp_idx[:, None], supp_idx]
        H_invG = H_inv @ XTY[supp_idx, :]

        if ksize >= 2:
            for i3 in range(num_cin):
                if not prune_list[i3]:
                    continue

                b_ori = XTX[supp_idx, i3 * ksize:(i3 + 1) * ksize]
                C_inv = torch.linalg.inv(
                    XTX[i3 * ksize:(i3 + 1) * ksize, i3 * ksize:(i3 + 1) * ksize] - b_ori.T @ H_inv @ b_ori)

                gt = XTY[i3 * ksize:(i3 + 1) * ksize, :] - b_ori.T @ H_invG
                obj_in[i3] = torch.sum(gt * (C_inv @ gt)) / 2

                # W1 = torch.clone(W)
                # W1[i2*ksize:(i2+1)*ksize,:] = 0
                # nzi = torch.nonzero(W1[:,0], as_tuple=True)[0]
                # XTX_sub = XTX[nzi[:,None],nzi]
                # XTY_sub = XTY[nzi,:]
                # W1[nzi,:] = torch.linalg.inv(XTX_sub)@ XTY_sub
                # obj1 = torch.sum( -W1 * XTY + (1/2) * W1 * (XTX @ W1))

                # W2 = torch.clone(W)
                # W2[i2*ksize:(i2+1)*ksize,:] = 0
                # W2[i3*ksize:(i3+1)*ksize,:] = 1
                # nzi = torch.nonzero(W2[:,0], as_tuple=True)[0]
                # XTX_sub = XTX[nzi[:,None],nzi]
                # XTY_sub = XTY[nzi,:]
                # W2[nzi,:] = torch.linalg.inv(XTX_sub)@ XTY_sub
                # obj2 = torch.sum( -W2 * XTY + (1/2) * W2 * (XTX @ W2))

                # print("Out: {}, in: {}, true obj is ori: {}, out: {}".format(i2,i3,obj1-obj_cur,obj1-obj2))
                # print("Out: {}, in: {}, estimate obj is out: {}, in: {}".format(i2,i3,obj_sum[i2],obj_in[i3]))


        else:
            C_list = 1 / (torch.diag(XTX) - torch.sum(XTX[supp_idx, :] * (H_inv @ XTX[supp_idx, :]), axis=0) + (
                ~prune_list) * 1e-8)
            gt = XTY - XTX[:, supp_idx] @ H_invG
            obj_in = torch.sum(gt ** 2, axis=1) * C_list / 2

        idx2 = torch.argsort(-obj_in + 1e20 * (~prune_list))

        prune_list[idx2[:num_swap]] = False
        supp_idx = torch.cat([torch.arange(i * ksize, (i + 1) * ksize) for i in range(num_cin) if not prune_list[i]])

        XTX_inv = torch.zeros_like(XTX)
        XTX_inv[supp_idx[:, None], supp_idx] = torch.linalg.inv(XTX[supp_idx[:, None], supp_idx])

        W = torch.zeros_like(W)
        W = XTX_inv @ XTY

        obj_new = torch.sum(-W * XTY + (1 / 2) * W * (XTX @ W))

        print("Finish iter {}, old obj is {}, new is {}, numswap is {}".format(i_local, obj_cur, obj_new, num_swap))
        if obj_new < obj_cur * (1 + 1e-9):

            best_prune = torch.clone(prune_list)
            obj_cur = obj_new
        else:
            if switch_lb >= 1 or num_swap <= lb_swap:
                break
            else:
                num_swap = int(np.maximum(num_swap / 2, lb_swap))

                prune_list = torch.clone(best_prune)
                supp_idx = torch.cat(
                    [torch.arange(i * ksize, (i + 1) * ksize) for i in range(num_cin) if not prune_list[i]])

                XTX_inv = torch.zeros_like(XTX)
                XTX_inv[supp_idx[:, None], supp_idx] = torch.linalg.inv(XTX[supp_idx[:, None], supp_idx])

                W = torch.zeros_like(W)
                W = XTX_inv @ XTY

    supp_idx = torch.cat([torch.arange(i * ksize, (i + 1) * ksize) for i in range(num_cin) if not best_prune[i]])

    W = torch.zeros_like(W)
    W[supp_idx, :] = torch.linalg.inv(XTX[supp_idx[:, None], supp_idx]) @ XTY[supp_idx, :]

    W = W.to(Wtype)
    XTY = XTY.to(Wtype)
    XTX = XTX.to(Wtype)

    return W, torch.sum(-W * XTY + (1 / 2) * W * (XTX @ W))


def prune_osscar(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    head_list = {"facebook/opt-125m": 12, "facebook/opt-350m": 16, "facebook/opt-1.3b": 32, "facebook/opt-2.7b": 32,
                 "facebook/opt-6.7b": 32,
                 "facebook/opt-13b": 40, "facebook/opt-30b": 56, "facebook/opt-66b": 72}
    num_heads = head_list[args.base_model] if args.base_model in head_list else model.config.num_attention_heads

    #print('Starting ...')
    dataloader = get_examples("c4", tokenizer, n_samples=args.num_examples, seq_len=2048)

    use_cache = model.config.use_cache
    model.config.use_cache = False

    prefix = None
    if "Llama" in args.base_model or "llama" in args.base_model:
        layers = model.model.layers
        prefix = "model.layers"
    elif "opt" in args.base_model:
        layers = model.model.decoder.layers
        prefix = "model.decoder.layers"
    else:
        raise NotImplementedError

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    with torch.no_grad():
        inps, cache = prepare_calibration_input(model, layers, dataloader, dev)
        
    percdamp = 0.01
    nsamples, seqlen, hidden_size = inps.shape

    prune_time = 0
    outs = torch.zeros_like(inps)
    for i in tqdm(range(len(layers))):
        layer = layers[i]
        p_ratio = args.all_layer_ratio[i]
        if f"{prefix}.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"{prefix}.{i}"]
            inps, outs = inps.to(dev), outs.to(dev)
            for key, value in cache.items():
                if isinstance(value, tuple):
                    cache[key] = tuple([v.to(dev) for v in value])
                else:
                    cache[key] = value.to(dev)

        subset = find_layers(layer)
        osscar = {}
        for name in subset:
            if name not in ['self_attn.out_proj', 'fc2', 'mlp.down_proj', 'self_attn.o_proj']:
                continue

            osscar[name] = OSSCAR_prune(subset[name], nsamples=nsamples, seqlen=seqlen,
                                        update_iter=args.update_iter, update_iter2=args.update_iter2,
                                        lambda2=args.lambda2, layername=name, num_heads=num_heads,
                                        local_out=args.local_out, local_fc=args.local_fc, local_iter=args.local_iter,
                                        local_test=args.local_test)

        def add_batch(name):
            def tmp(_, inp, out):
                osscar[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in subset:
            if name not in ['self_attn.out_proj', 'fc2', 'mlp.down_proj', 'self_attn.o_proj']:
                continue
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0).to(dev), **cache)[0]

        for h in handles:
            h.remove()
        # start

        # print(f"osscar pruing layer {i}")
        for name in subset:
            start_time = time.time()
            if name not in ['self_attn.out_proj', 'fc2', 'mlp.down_proj', 'self_attn.o_proj']:
                continue
            osscar[name].prune(sp_fc=p_ratio, sp_out=args.sp_out)
            prune_time += time.time() - start_time
            osscar[name].free()

        for j in range(args.num_examples):
            outs[j] = layer(inps[j].unsqueeze(0), **cache)[0]

        inps, outs = outs, inps
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    print("time_cost: %.5f sec" % prune_time)


class OSSCARPruner:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def prune(self, args):
        before_pruning_parameters = sum(p.numel() for p in self.model.parameters())
        print("Before prune, #parameters: {}".format(before_pruning_parameters))

        prune_osscar(args, self.model, self.tokenizer, dev=self.model.device, prune_n=args.N, prune_m=args.M)

        after_pruning_parameters = sum(torch.count_nonzero(p).item() for p in self.model.parameters())
        print("After prune, #parameters: {}".format(after_pruning_parameters))

        if args.save_path is not None:
            self.tokenizer.save_pretrained(args.save_path)
            self.model.save_pretrained(args.save_path, max_shard_size="10GB")