import argparse

from owl_sparsity import LAYER_SPARSITY_DICT
from utils.data_utils import *
from utils.model_utils import *
from evaluater import *
from utils.utils import *
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory, get_max_memory

class StopForwardPassException(Exception):
    pass

def calculate_mlp_ratio(model, args):
    assert "llama" in args.model
    layers = model.model.layers
    attn_param_num = 0
    for _, param in layers[0].self_attn.named_parameters():
        attn_param_num += param.numel()
    mlp_param_num = 0
    for _, param in layers[0].mlp.named_parameters():
        mlp_param_num += param.numel()
    mlp_ratio = (args.overall_ratio * (attn_param_num + mlp_param_num) - args.attn_ratio * attn_param_num) / mlp_param_num
    print(f"attn_param_num: {attn_param_num}, mlp_param_num: {mlp_param_num}, args.overall_ratio: {args.overall_ratio}, args.attn_ratio: {args.attn_ratio},  mlp_ratio: {mlp_ratio}")
    assert 0 <= mlp_ratio <= 1
    return mlp_ratio
        

@torch.no_grad()
def profle_svdllm_low_resource(model, calib_loader, args):
    dev = args.DEV
    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.rotary_emb = model.model.rotary_emb.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (len(calib_loader), model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, "position_ids": None, "cos": None, "sin": None}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp.cpu()
            cache['i'] += 1
            cache['position_ids'] = kwargs['position_ids']
            cache['cos'], cache['sin'] = kwargs['position_embeddings']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.rotary_emb = model.model.rotary_emb.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    position_ids = cache['position_ids'].to(dev)
    cos = cache['cos'].to(dev)
    sin = cache['sin'].to(dev)
    profiling_mat = {}
    for i in tqdm(range(len(layers))):
        layer_profile = {}
        layer = layers[i].to(dev)
        subset = find_layers(layer)        
        def hook(module, input, output):
            inp = input[0].detach().float()
            if inp.dim() == 2:
                inp = inp.unsqueeze(0)
            if args.prune == "svd-llm":
                adds = torch.matmul(inp.transpose(1,2), inp)
                adds_sum = torch.sum(adds, dim=0)
                module.scaling_diag_matrix += adds_sum
                del inp, adds, adds_sum, output
            else:
                raise ValueError("Unsupported prune type")
            torch.cuda.empty_cache()
        handles = []
        for name in subset:
            subset[name].scaling_diag_matrix = 0
            handles.append(subset[name].register_forward_hook(hook))
        for j in range(inps.shape[0]):
            outs[j] = layer(inps[j].unsqueeze(0), position_ids=position_ids, position_embeddings=(cos, sin))[0]
        for h in handles:
            h.remove()
        layer = layer.cpu()
        for name in subset:
            subset[name].scaling_diag_matrix = subset[name].scaling_diag_matrix.cpu()
        torch.cuda.empty_cache()
        for name in subset:
            if args.prune == "svd-llm":
                raw_scaling_diag_matrix = subset[name].scaling_diag_matrix.double().to(dev)
                try:
                    scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
                except Exception as e:
                    print("Warning: eigen scaling_diag_matrix is not positive!")
                    eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
                    raw_scaling_diag_matrix += (- eigenvalues[0] + 1e-6) * torch.eye(raw_scaling_diag_matrix.shape[0]).to(dev)
                    scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
                    eigenvalues = None
                    del eigenvalues
            else:
                raise ValueError("Unsupported prune type")
            layer_profile[name] = scaling_diag_matrix.float().cpu()
            torch.cuda.empty_cache()
        layers[i] = layer.cpu()
        profiling_mat[i] = layer_profile
        inps = outs
        torch.cuda.empty_cache()
    return profiling_mat



@torch.no_grad()
def reconstruct(model, dataloader, profiling_mat, args):
    layer_sparsity_list = None
    if args.owl is True:
        for name in LAYER_SPARSITY_DICT:
            if name in args.model:
                layer_sparsity_list = LAYER_SPARSITY_DICT[name]
                break
        assert layer_sparsity_list is not None
    attn_ratio = args.attn_ratio
    mlp_ratio = calculate_mlp_ratio(model, args)
    print("Start SVD decomposition then update...")
    use_cache = model.config.use_cache
    model.config.use_cache = False
    full_name_dict = {module: name for name, module in model.named_modules()}
    linear_info = {}
    modules = [model]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if isinstance(raw_linear, nn.Linear):
                full_name = full_name_dict[raw_linear]

                linear_info[raw_linear] = {
                    "father": submodule,
                    "name": name,
                    "full_name": full_name,
                }
            else:
                modules.append(raw_linear)
    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.rotary_emb = model.model.rotary_emb.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (len(dataloader), model.seqlen, model.config.hidden_size), dtype=dtype
    )
    cache = {'i': 0, "position_ids": None, "cos": None, "sin": None}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp.cpu()
            cache['i'] += 1
            cache['position_ids'] = kwargs['position_ids']
            cache['cos'], cache['sin'] = kwargs['position_embeddings']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.rotary_emb = model.model.rotary_emb.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    old_outs = torch.zeros_like(inps)
    position_ids = cache['position_ids'].to(dev)
    cos = cache['cos'].to(dev)
    sin = cache['sin'].to(dev)
    old_inps = torch.clone(inps)
    prune_order = [["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"], ["self_attn.o_proj"], ["mlp.gate_proj", "mlp.up_proj"], ["mlp.down_proj"]]
    for i in tqdm(range(len(layers))):
        layer = layers[i].to(dev)
        subset = find_layers(layer)
        gpts = {}
        for name in subset:
            if profiling_mat is not None:
                scaling_diag_matrix = profiling_mat[i][name].to(dev)
            else:
                scaling_diag_matrix = None
            if "self_attn" in name:
                ratio = attn_ratio
            else:
                ratio = mlp_ratio
            if args.owl is True:
                assert len(layers) == len(layer_sparsity_list)
                ratio = ratio * (1 - layer_sparsity_list[i]) / (1 - np.mean(layer_sparsity_list))
            assert 0 < ratio < 1
            print(f"{i}, {name} density: {ratio}")
            gpts[name] = local_update(subset[name], scaling_diag_matrix=scaling_diag_matrix, ratio=ratio, name=name, args=args)

        def add_old_batch(name):
            def tmp(_, inp, out):
                return gpts[name].cache_old_input(inp[0].data, out.data)
            return tmp
        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_old_batch(name)))
        for j in range(inps.shape[0]):
            old_outs[j] = layer(old_inps[j].unsqueeze(0).to(dev), position_ids=position_ids, position_embeddings=(cos, sin))[0]
        for h in handles:
            h.remove()
        old_inps = old_outs

        def add_batch(name):
            def tmp(_, inp, out):
                return gpts[name].cache_new_input(inp[0].data, out.data)
            return tmp
        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for cur_prune_name in prune_order:
            for name in cur_prune_name:
                gpts[name].begin_cache = True
            for j in range(inps.shape[0]):
                try:
                    _ = layer(inps[j].unsqueeze(0).to(dev), position_ids=position_ids, position_embeddings=(cos, sin))[0]
                except StopForwardPassException as e:
                    continue
            for name in cur_prune_name:
                gpts[name].do_prune()
        for h in handles:
            h.remove()
        for name in gpts:
            raw_linear = gpts[name].layer
            svd_u, svd_v = gpts[name].fasterprune()
            svd_u, svd_v = svd_u.to(dtype), svd_v.to(dtype)
            info = linear_info[raw_linear]
            if raw_linear.bias is None:
                bias = None
            else:
                bias = raw_linear.bias.data
            svd_linear = SVDLinear(svd_u, svd_v, bias)
            setattr(info["father"], info["name"], svd_linear)
            raw_linear.cpu()
        layer = layer.to(dev)
        for j in range(inps.shape[0]):
            outs[j] = layer(inps[j].unsqueeze(0).to(dev), position_ids=position_ids, position_embeddings=(cos, sin))[0]
        layers[i] = layer.cpu()
        del gpts
        torch.cuda.empty_cache()
        inps = outs
    model.config.use_cache = use_cache


class local_update:
    def __init__(self, layer, scaling_diag_matrix, ratio, name, args):
        self.old_output_factor = args.old_output_factor
        self.layer = layer
        self.name = name
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.W = W
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        if args.prune == "svd-llm":
            try:
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            except Exception as e:
                print("Warning: scaling_diag_matrix is not full rank!")
                scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0])
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            scaling_diag_matrix = scaling_diag_matrix.float()
            scaling_matrix_inv = scaling_matrix_inv.float()
            W_scale = torch.matmul(W, scaling_diag_matrix)
        elif args.prune == "svd":
            W_scale = W
        else:
            raise ValueError("Unsupported prune type")

        U, S, VT = torch.linalg.svd(W_scale, full_matrices=False)
        # trucation SVD
        if args.use_pifa is False:
            low_rank = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))
        else:
            low_rank = int((W.shape[0] + W.shape[1] + 1 - np.sqrt(
                (W.shape[0] + W.shape[1] + 1) ** 2 - 4 * int(W.shape[0] * W.shape[1] * ratio))) / 2)
            low_rank = min(low_rank, W.shape[0], W.shape[1])
        truc_s = S[:low_rank].cuda()
        truc_u = U[:, :low_rank].cuda()
        if args.prune == "svd-llm":
            truc_v = torch.matmul(VT[:low_rank, :].cuda(), scaling_matrix_inv)
        elif args.prune == "svd":
            truc_v = VT[:low_rank, :].cuda()

        sqrtSigma = torch.sqrt(torch.diag(truc_s))
        self.truc_u = truc_u.matmul(sqrtSigma)
        self.truc_v = sqrtSigma.matmul(truc_v)

        self.old_output_wo_bias_list = []
        self.reconstruct_step = args.reconstruct_step
        self.current_sample_index = 0
        self.XTX = 0
        self.YTX = 0
        self.pruned = False
        self.begin_cache = False

    def cache_old_input(self, inp, out):
        if len(inp.shape) == 3:
            old_output_wo_bias = out.view(out.shape[0] * out.shape[1], out.shape[2])
        else:
            old_output_wo_bias = out
        if self.layer.bias is not None:
            old_output_wo_bias = old_output_wo_bias - self.layer.bias
        self.old_output_wo_bias_list.append(old_output_wo_bias.cpu())

    def update_u(self, step_num):
        final_x = self.truc_v @ self.XTX @ self.truc_v.T
        final_y = self.YTX @ self.truc_v.T
        self.truc_u = torch.linalg.solve(final_x, final_y, left=False)

    def update_v(self, step_num):
        lambda_reg = 1e-3
        reg = lambda_reg * torch.eye(self.XTX.size(1), device=self.XTX.device)
        new_w = torch.linalg.solve(self.XTX + reg, self.YTX + lambda_reg * self.W, left=False)
        self.truc_v = torch.linalg.lstsq(self.truc_u, new_w).solution

    def cache_new_input(self, inp, out):
        if self.pruned is False and self.begin_cache is True:
            if len(inp.shape) == 3:
                inp_2d = inp.view(inp.shape[0] * inp.shape[1], inp.shape[2])
                out_2d = out.view(out.shape[0] * out.shape[1], out.shape[2])
            else:
                inp_2d = inp
                out_2d = out
            if self.layer.bias is not None:
                outs_wo_bias = out_2d - self.layer.bias
            else:
                outs_wo_bias = out_2d
            merged_outs_wo_bias = (1 - self.old_output_factor) * outs_wo_bias + self.old_output_factor * self.old_output_wo_bias_list[self.current_sample_index].to(outs_wo_bias.device)
            self.XTX += inp_2d.T @ inp_2d
            self.YTX += merged_outs_wo_bias.T @ inp_2d
            self.current_sample_index += 1
        elif self.begin_cache is False:
            raise StopForwardPassException(self.name)
        return torch.matmul(torch.matmul(inp, self.truc_v.T), self.truc_u.T)

    def do_prune(self):
        assert self.pruned is False
        if self.reconstruct_step == -1:
            self.update_v(-1)
        else:
            for step_num in range(self.reconstruct_step):
                if step_num % 2 == 0:
                    self.update_u(step_num)
                else:
                    self.update_v(step_num)
        torch.cuda.empty_cache()
        self.pruned = True
    
    def fasterprune(self):
        assert self.pruned is True
        return self.truc_u, self.truc_v


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='jeffwan/llama-7b-hf', help='LLaMA model to load, pass `jeffwan/llama-7b-hf`')
    parser.add_argument('--model_path', type=str, default=None, help='local compressed model path or whitening information path')
    parser.add_argument('--overall_ratio', type=float, default=0.2, help='global sparsity')
    parser.add_argument('--attn_ratio', type=float, default=0.2, help='attention module sparsity')
    parser.add_argument('--dataset', type=str, default='wikitext2', help='Where to extract calibration data from [wikitext2, ptb, c4]')
    parser.add_argument('--pruning_nsamples', type=int, default=256, help='Number of calibration data samples for svd pruning.')
    parser.add_argument('--reconstruct_nsamples', type=int, default=16, help='Number of calibration data samples for udpating.')
    parser.add_argument('--save_path', type=str, default=None, help='the path to save the compressed model checkpoints.`')
    parser.add_argument('--profiling_mat_path', type=str, default=None, help='Local path to load the profiling matrices`')
    parser.add_argument('--seed',type=int, default=0, help='Seed for sampling the calibration data')
    parser.add_argument('--DEV', type=str, default="cuda", help='device')
    parser.add_argument('--model_seq_len', type=int, default=2048, help='the default sequence length of the LLM')
    parser.add_argument('--eval_batch_size', type=int, default=4, help='inference bactch size')
    parser.add_argument('--mode', type=str, default=None)
    parser.add_argument("--use_pifa", action="store_true")
    parser.add_argument("--old_output_factor", type=float, default=0)
    parser.add_argument("--reconstruct_step", type=int, default=1)
    parser.add_argument("--owl", action="store_true")
    parser.add_argument("--prune", type=str, choices=["svd-llm", "svd"], default="svd-llm", help="Choose pruning method.")
    parser.add_argument("--prefix", type=str, default="")

    args = parser.parse_args()
    args.attn_ratio = 1 - args.attn_ratio
    args.overall_ratio = 1 - args.overall_ratio

    # setting random seed of numpy and torch
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    if args.mode == "m":
        model, tokenizer = get_model_from_huggingface(model_id=args.model)
        dataloader, _ = get_loaders(args.dataset, nsamples=args.reconstruct_nsamples, seed=args.seed, tokenizer=tokenizer, seqlen=args.model_seq_len)
        model = model.eval()
        model = model.float()
        if args.profiling_mat_path is None:
            cali_white_data = get_calib_train_data(args.dataset, args.model, tokenizer, args.pruning_nsamples, seqlen=args.model_seq_len)
            profiling_mat = profle_svdllm_low_resource(model, cali_white_data, args)
            if args.save_path is not None:
                torch.save(profiling_mat, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") + f'_{args.prune}-profiling_'+ args.dataset + '_' + str(args.pruning_nsamples)  + '_' + str(args.seed)+ '_float32.pt')
        else:
            profiling_mat = torch.load(args.profiling_mat_path)
        reconstruct(model, dataloader, profiling_mat, args)
        if args.save_path is not None:
            path = args.save_path + f"/{args.prefix}" + args.model.replace("/", "_").replace("-", "_") + f'_{args.prune}-update-{args.reconstruct_step}' + f"_owl-{args.owl}_overall-{args.overall_ratio}_attn-{args.attn_ratio}" + f"_{args.use_pifa}" + f"_{args.old_output_factor}" + f"_{args.reconstruct_nsamples}" + '.pt'
            print(f"save path: {path}")
            torch.save({'model': model, 'tokenizer': tokenizer}, path)

    elif args.mode == "pifa":
        model, tokenizer = get_model_from_local(args.model_path)
        model = model.half()
        model.eval()
        replace_with_compress(model)
        print(model)
        new_path = args.model_path.replace("results/", "compress/")
        torch.save({'model': model, 'tokenizer': tokenizer}, new_path)

    elif args.mode == "ppl":
        print(f"evaluating {args.model_path}...")
        if args.model_path == "original":
            model, tokenizer = get_model_from_huggingface(args.model, "auto")
        else:
            model, tokenizer = get_model_from_local(args.model_path)
            model = model.half()
            target_dtype = torch.float16
            no_split_modules = model._no_split_modules
            device_map_kwargs = {"no_split_module_classes": no_split_modules}
            device_map_kwargs["special_dtypes"] = {}
            max_memory = get_balanced_memory(
                model,
                dtype=target_dtype,
                low_zero=False,
                max_memory=None,
                **device_map_kwargs,
            )
            max_memory = get_max_memory(max_memory)
            device_map_kwargs["max_memory"] = max_memory

            device_map = infer_auto_device_map(model, dtype=target_dtype, **device_map_kwargs)
            dispatch_model(model, device_map=device_map)
        print(model)

        model.eval()
        torch.cuda.empty_cache()
        ppl_eval(model, tokenizer, datasets=['wikitext2'], model_seq_len=args.model_seq_len, batch_size=args.eval_batch_size, device=args.DEV)