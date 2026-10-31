import argparse
import os

parser=argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name')
parser.add_argument('--config', type=str, help='path to config file')
parser.add_argument('--gpu', type=str, default='0', help='which GPU to use')
parser.add_argument('--model_name', type=str, default='', help='name of stored model')
parser.add_argument('--qbits', type=int, default=0, help='0=fp32; 8=QTem int8 static-feature transfer')
parser.add_argument('--qpack', type=int, default=0, help='1=lossless bit-packed transfer for bool feature files (GDELT)')
parser.add_argument('--qblock', type=int, default=128)
parser.add_argument('--qgpu', type=int, default=0, help='1=keep compressed static tables resident in GPU memory')
parser.add_argument('--half', type=int, default=0, help='1=store static tables as binary16 (half-precision baseline)')
parser.add_argument('--boolfeat', type=int, default=0, help='1=keep binary feature files as bool/uint8, 1 byte per value (lossless baseline)')
parser.add_argument('--pin', type=int, default=0, help='1=host-side row gathers into pinned staging buffers')
parser.add_argument('--eval_only', type=int, default=0, help='1=skip training and evaluate --ckpt')
parser.add_argument('--ckpt', type=str, default='', help='checkpoint evaluated with --eval_only')
parser.add_argument('--rowgather', type=int, default=0, help='1=fp32 arm gathers rows with index_select (row memcpy) instead of advanced indexing')
parser.add_argument('--seed', type=int, default=-1, help='-1=off; >=0 fixes python/numpy/torch seed before feature gen')
parser.add_argument('--reserve_mib', type=int, default=0, help='hold this much GPU memory in the caching allocator from the start')
parser.add_argument('--skip_eval', type=int, default=0, help='1=timing harness: skip validation/test (training loop untouched)')
parser.add_argument('--eval_neg_samples', type=int, default=1, help='how many negative samples to use at inference. Note: this will change the metric of test set to AP+AUC to AP+MRR!')
args=parser.parse_args()

# Select the GPU with CUDA_VISIBLE_DEVICES (--gpu is not used).

# Host threads of the data path (OMP / MKL / OpenBLAS and the intra-op pool of PyTorch) from $QTEM_CPU_THREADS,
# set before torch is imported; unset = library defaults
_nt = os.environ.get('QTEM_CPU_THREADS', '')
if _nt:
    for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[_v] = _nt
import multiprocessing as mp
import threading
import queue
import torch
if _nt:
    torch.set_num_threads(int(_nt))
    print('[harness] cpu threads = {}'.format(_nt), flush=True)


def _proc_mem():  # peak (VmHWM) and current (VmRSS, RssAnon) host memory of this process, GB
    out = {}
    try:
        for _ln in open('/proc/self/status'):
            _k = _ln.split(':')[0]
            if _k in ('VmHWM', 'VmRSS', 'RssAnon'):
                out[_k + '_gb'] = int(_ln.split()[1]) / 1e6
    except OSError:
        pass
    return out
import time
import random
import dgl
import numpy as np
from modules import *
from sampler import *
from utils import *
from qtem import QuantizedFeatureTable, PackedFeatureTable, reset_stats, stats_line, snapshot_stats, log_stats_delta, ch_reset, ch_json, ch_add, payload_bytes, PlainTable, CODEC_TABLES
import qtem as _qtem_mod
import json as _json
import utils as _utils_mod
import memorys as _memorys_mod
phase_tag = 'unset'
if args.reserve_mib > 0:
    _rsv = torch.empty(int(args.reserve_mib) << 20, dtype=torch.uint8, device='cuda')
    del _rsv
    torch.cuda.reset_peak_memory_stats()
    print('[RESERVE] %d MiB held in the caching allocator' % args.reserve_mib, flush=True)
_BD_KEYS = ('main_wait', 'main_h2d', 'main_recon', 'main_compute', 'main_memupd', 'prep_blocks', 'prep_gather', 'prep_h2d_recon')
_BD = dict.fromkeys(_BD_KEYS, 0.0)
def _set_phase(tag):
    global phase_tag
    phase_tag = tag
    _utils_mod.phase_tag = tag
    _memorys_mod.phase_tag = tag
from sklearn.metrics import average_precision_score, roc_auc_score

def preparation(ret_list, node_list, ts_list, node_feats, edge_feats, history, total, flag1, flag2, q):
    for i in range(total):
        if not flag1:
            _p0 = time.time()
            mfgs, uni_node, inv_node, uni_edge, inv_edge = \
        to_dgl_blocks(ret_list[i], history)
            _p1 = time.time()
            _BD['prep_blocks'] += _p1 - _p0
            _need_edge = False
            if edge_feats is not None:
                for _mfg in mfgs:
                    for _b in _mfg:
                        if _b.num_src_nodes() > _b.num_dst_nodes():
                            _need_edge = True
            if isinstance(node_feats, CODEC_TABLES):
                node_data = node_feats.gather_compressed(uni_node) if flag2 else node_feats.gather_dequant_cuda(uni_node)
            else:
                node_data = ((torch.index_select(node_feats, 0, uni_node.long()) if args.rowgather else node_feats[uni_node.long()]).float()) if node_feats is not None else None
            if isinstance(edge_feats, CODEC_TABLES):
                edge_data = (edge_feats.gather_compressed(uni_edge) if flag2 else edge_feats.gather_dequant_cuda(uni_edge)) if _need_edge else None
            else:
                edge_data = ((torch.index_select(edge_feats, 0, uni_edge.long()) if args.rowgather else edge_feats[uni_edge.long()]).float()) if _need_edge else None
            if node_feats is not None:
                ch_add('static_node', 'h2d', phase_tag, payload_bytes(node_feats, uni_node.shape[0]))
            if edge_feats is not None and _need_edge:
                ch_add('static_edge', 'h2d', phase_tag, payload_bytes(edge_feats, uni_edge.shape[0]))
            _p2 = time.time()
            _BD['prep_gather'] += _p2 - _p1
            if not flag2:
                if node_data is not None:
                    node_data = node_data.cuda()
                if edge_data is not None:
                    edge_data = edge_data.cuda()
                mfgs = feat_reconstruct(mfgs, node_data, edge_data, inv_node, inv_edge)
                node_data, edge_data = None, None
            _BD['prep_h2d_recon'] += time.time() - _p2
            uni_node_r, inv_node_r, edge_r = None, None, None
        elif flag1:
            mfgs, uni_node, inv_node, uni_edge, inv_edge = node_to_dgl_blocks(node_list[i], ts_list[i])
            mfgs_r, uni_node_r, inv_node_r, uni_edge_r, inv_edge_r = to_dgl_blocks(ret_list[i], history, cuda=False,  reverse=True)
            edge_r = mfgs_r[0][0].edges()[0]
            node_data, edge_data = None, None
        
        q.put((mfgs, uni_node, inv_node, uni_edge, inv_edge, uni_node_r, inv_node_r, edge_r, node_data, edge_data))


if __name__ == '__main__':
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if args.seed >= 0:
        random.seed(args.seed); np.random.seed(args.seed)
        torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
        print('[QTEM] seed fixed to %d before feature generation' % args.seed)
    _t_load0 = time.time()
    if args.qpack > 0:
        # lossless tier: pack bool feature files directly, never materialize fp32
        reset_stats()
        node_feats, edge_feats = None, None
        if os.path.exists('DATA/{}/node_features.pt'.format(args.data)):
            raw = torch.load('DATA/{}/node_features.pt'.format(args.data))
            assert raw.dtype == torch.bool, '--qpack requires bool node_features.pt'
            _te = time.time(); node_feats = PackedFeatureTable(raw); print('[ENCODE_TIME] node pack %.2f' % (time.time() - _te), flush=True); del raw
        if os.path.exists('DATA/{}/edge_features.pt'.format(args.data)):
            raw = torch.load('DATA/{}/edge_features.pt'.format(args.data))
            assert raw.dtype == torch.bool, '--qpack requires bool edge_features.pt'
            _te = time.time(); edge_feats = PackedFeatureTable(raw); print('[ENCODE_TIME] edge pack %.2f' % (time.time() - _te), flush=True); del raw
        if node_feats is None and edge_feats is None:
            raise SystemExit('--qpack: dataset %s has no bool feature files (random-feature datasets use fp32 randn; use --qbits instead)' % args.data)
        print('[QTEM] static-feature channel packed losslessly: bool bit-packing (zero error)')
    elif args.boolfeat > 0:
        node_feats, edge_feats = None, None
        for _nm in ('node', 'edge'):
            _fp = 'DATA/{}/{}_features.pt'.format(args.data, _nm)
            _shm = '/dev/shm/etcport_{}_{}_u8.npy'.format(args.data, _nm)
            if _nm == 'edge' and os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SHM_FEATURES'))                     and os.path.exists(_shm) and os.path.exists(_fp):
                import warnings as _w
                with _w.catch_warnings():
                    _w.simplefilter('ignore')  # read-only mapping of the shared byte-identical copy
                    raw = torch.from_numpy(np.load(_shm, mmap_mode='r')).view(torch.bool)
                print('[QTEM] %s table mapped from the shared byte-identical copy %s' % (_nm, _shm), flush=True)
            elif os.path.exists(_fp):
                raw = torch.load(_fp)
            if os.path.exists(_fp):
                assert raw.dtype == torch.bool, '--boolfeat requires bool feature files'
                if _nm == 'node':
                    node_feats = PlainTable(raw)
                else:
                    edge_feats = PlainTable(raw)
                del raw
        print('[QTEM] baseline: binary tables kept as bool/uint8, 1 byte per value (lossless)', flush=True)
    else:
        node_feats, edge_feats = load_feat(args.data)
        if args.half > 0:
            node_feats = PlainTable(node_feats.half()) if node_feats is not None else None
            edge_feats = PlainTable(edge_feats.half()) if edge_feats is not None else None
            print('[QTEM] baseline: static tables stored in binary16', flush=True)
        elif args.qbits == 0 and (args.pin > 0 or args.qgpu > 0):
            node_feats = PlainTable(node_feats) if node_feats is not None else None
            edge_feats = PlainTable(edge_feats) if edge_feats is not None else None
            print('[QTEM] baseline: full-precision tables gathered by row copy', flush=True)
        if args.qbits > 0:
            reset_stats()
            if node_feats is not None:
                _te = time.time(); node_feats = QuantizedFeatureTable(node_feats, args.qbits, args.qblock); print('[ENCODE_TIME] node q%d %.2f' % (args.qbits, time.time() - _te), flush=True)
            if edge_feats is not None:
                _te = time.time(); edge_feats = QuantizedFeatureTable(edge_feats, args.qbits, args.qblock); print('[ENCODE_TIME] edge q%d %.2f' % (args.qbits, time.time() - _te), flush=True)
            print('[QTEM] static-feature channel quantized: int8, block=%d' % args.qblock)
    print('[FEATURES] node=%s edge=%s' % (getattr(node_feats, 'shape', None), getattr(edge_feats, 'shape', None)), flush=True)
    print('[LOAD_TIME] %.2f' % (time.time() - _t_load0), flush=True)
    if args.pin > 0:
        _qtem_mod.PIN['enabled'] = True
        print('[QTEM] pinned staging buffers: %d slots per table' % _qtem_mod.PIN['slots'], flush=True)
    g, df = load_graph(args.data)
    print('load initial data finish.')
    sample_param, memory_param, gnn_param, train_param = parse_config(args.config)
    train_edge_end = df[df['ext_roll'].gt(0)].index[0]
    val_edge_end = df[df['ext_roll'].gt(1)].index[0]

    gnn_dim_node = 0 if node_feats is None else node_feats.shape[1]
    gnn_dim_edge = 0 if edge_feats is None else edge_feats.shape[1]
    combine_first = False
    if 'combine_neighs' in train_param and train_param['combine_neighs']:
        combine_first = True
    model = GeneralModel(gnn_dim_node, gnn_dim_edge, sample_param, memory_param, gnn_param, train_param, combined=combine_first).cuda()
    mailbox = MailBox(memory_param, g['indptr'].shape[0] - 1, gnn_dim_edge) if memory_param['type'] != 'none' else None
    creterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_param['lr'])
    print('initialize module part finish.')
    if args.qgpu > 0:
        for _tab in (node_feats, edge_feats):
            if isinstance(_tab, CODEC_TABLES):
                _tab.to_gpu()
        torch.cuda.synchronize()
        print('[QTEM] compressed static tables resident in GPU memory; allocated %.2f GB' % (torch.cuda.memory_allocated() / 1e9), flush=True)
    if 'all_on_gpu' in train_param and train_param['all_on_gpu']:
        if node_feats is not None:
            node_feats = node_feats.cuda()
        if edge_feats is not None:
            edge_feats = edge_feats.cuda()
        if mailbox is not None:
            mailbox.move_to_gpu()
    
    def get_staleness_constraint(data, batch_size):
        range_ = np.arange(0,len(data),batch_size)
        if range_[-1] != len(data):
            range_ = np.append(range_, [len(data)])
        #Calculate the staleness
        staleness = []
        for i in range(len(range_)):
            if i != len(range_)-1:
                update = 2*(range_[i+1]-range_[i])
                nodes = len(np.unique(data[range_[i]:range_[i+1]]))
                staleness.append(update-nodes)
        staleness = np.array(staleness)
        #return the upper bound for the staleness
        return max(staleness)
    
    def adaptive_split(threshold, data, group_index):
        start_list = []
        start = 0
        while start < len(data)-1:
            count_nodes = 0
            count_edges = 0
            node_set = set()
            for i in range(start, len(data)):
                count_edges += 2
                for j in data[i]:
                    if j not in node_set:
                        node_set.add(j)
                        count_nodes += 1
                if count_edges - count_nodes > threshold or i == len(data)-1:
                    end = i
                    break
            start_list.append(end)
            start = end
        start_list = [0] + start_list
        start_list[-1] = len(data)
        #generate the split
        split = []
        for i in range(len(start_list)):
            if i != len(start_list)-1:
                split.append(range(start_list[i], start_list[i+1]))
        for i in range(len(split)):
            group_index[split[i]] = i

        return group_index

    sampler = None
    if not ('no_sample' in sample_param and sample_param['no_sample']):
        sampler = ParallelSampler(g['indptr'], g['indices'], g['eid'], g['ts'].astype(np.float32),
                                  sample_param['num_thread'], 1, sample_param['layer'], sample_param['neighbor'],
                                  sample_param['strategy']=='recent', sample_param['prop_time'],
                                  sample_param['history'], float(sample_param['duration']))
    neg_link_sampler = NegLinkSampler(g['indptr'].shape[0] - 1)
    print('initialize sampler finish.')
    def eval(group_indices,mode='val'):
        _set_phase('eval_' + mode)
        eval_stats_start = snapshot_stats()
        neg_samples = 1
        model.eval()
        aps = list()
        aucs_mrrs = list()
        if mode == 'val':
            eval_df = df[train_edge_end:val_edge_end]
        elif mode == 'test':
            eval_df = df[val_edge_end:]
            neg_samples = args.eval_neg_samples
        elif mode == 'train':
            eval_df = df[:train_edge_end]
        with torch.no_grad():
            if mode == 'train':
                group_index = group_indices[0]
            if mode == 'val':
                group_index = group_indices[1]
            elif mode == 'test':
                group_index = group_indices[2]
            total_loss = 0
            for _, rows in eval_df.groupby(group_index):
                root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample(len(rows) * neg_samples)]).astype(np.int32)
                ts = np.tile(rows.time.values, neg_samples + 2).astype(np.float32)
                if sampler is not None:
                    if 'no_neg' in sample_param and sample_param['no_neg']:
                        pos_root_end = len(rows) * 2
                        sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
                    else:
                        sampler.sample(root_nodes, ts)
                    ret = sampler.get_ret()
                if gnn_param['arch'] != 'identity':
                    mfgs, uni_node, inv_node, uni_edge, inv_edge = \
                    to_dgl_blocks(ret, sample_param['history'])
                else:
                    mfgs, uni_node, inv_node, uni_edge, inv_edge = node_to_dgl_blocks(root_nodes, ts)
                mfgs = prepare_input(mfgs, node_feats, edge_feats, uni_node, inv_node, uni_edge, inv_edge, combine_first=combine_first)
                if mailbox is not None:
                    mailbox.prep_input_mails(mfgs[0],uni_node, inv_node, uni_edge, inv_edge)
                pred_pos, pred_neg = model(mfgs, neg_samples=neg_samples)
                total_loss += creterion(pred_pos, torch.ones_like(pred_pos))
                total_loss += creterion(pred_neg, torch.zeros_like(pred_neg))
                y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu()
                y_true = torch.cat([torch.ones(pred_pos.size(0)), torch.zeros(pred_neg.size(0))], dim=0)
                aps.append(average_precision_score(y_true, y_pred))
                if neg_samples > 1:
                    aucs_mrrs.append(torch.reciprocal(torch.sum(pred_pos.squeeze() < pred_neg.squeeze().reshape(neg_samples, -1), dim=0) + 1).type(torch.float))
                else:
                    aucs_mrrs.append(roc_auc_score(y_true, y_pred))
                if mailbox is not None:
                    eid = rows['Unnamed: 0'].values
                    if edge_feats is None:
                        mem_edge_feats = None
                    elif isinstance(edge_feats, CODEC_TABLES):
                        mem_edge_feats = edge_feats.gather_dequant_cuda(torch.from_numpy(np.asarray(eid)))
                    else:
                        mem_edge_feats = edge_feats[eid].cuda()
                    if edge_feats is not None:
                        ch_add('static_edge', 'h2d', phase_tag, payload_bytes(edge_feats, len(eid)))
                    root_nodes_gpu = torch.from_numpy(root_nodes).cuda()
                    ts_gpu = torch.from_numpy(ts).cuda()
                    block = None
                    if memory_param['deliver_to'] == 'neighbors':
                        block,_,_,_,_ = to_dgl_blocks(ret, sample_param['history'], reverse=True)
                        block = block[0][0]
                    mailbox.update_mailbox(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes_gpu, ts_gpu, mem_edge_feats, block, neg_samples=neg_samples)
                    mailbox.update_memory(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, model.memory_updater.last_updated_ts, neg_samples=neg_samples)
            if mode == 'val':
                val_losses.append(float(total_loss))
        ap = float(torch.tensor(aps).mean())
        if neg_samples > 1:
            auc_mrr = float(torch.cat(aucs_mrrs).mean())
        else:
            auc_mrr = float(torch.tensor(aucs_mrrs).mean())
        log_stats_delta('eval_' + mode, eval_stats_start)
        return ap, auc_mrr
    
    
    if not os.path.isdir('models'):
        os.mkdir('models')
    if args.model_name == '':
        path_saver = 'models/{}_{}.pkl'.format(args.data, time.time())
    else:
        path_saver = 'models/{}.pkl'.format(args.model_name)
    if args.eval_only:
        path_saver = args.ckpt
        train_param['epoch'] = 0
        print('[QTEM] eval-only: %s, test negatives per positive=%d' % (path_saver, args.eval_neg_samples), flush=True)
    best_ap = 0
    best_e = 0
    val_losses = list()
    group_indices = []
    #get train split
    train_index = np.array(df[:train_edge_end].index)
    train_data = np.stack([df[:train_edge_end].src, df[:train_edge_end].dst]).T
    threshold = get_staleness_constraint(train_data, train_param['batch_size'])
    train_group_index = adaptive_split(threshold, train_data, train_index)
    group_indices.append(train_group_index)
    #get val split
    val_index = np.array(df[train_edge_end:val_edge_end].index)
    val_data = np.stack([df[train_edge_end:val_edge_end].src, df[train_edge_end:val_edge_end].dst]).T
    threshold = get_staleness_constraint(val_data, train_param['batch_size'])
    val_group_index = adaptive_split(threshold, val_data, val_index)
    val_group_index += train_group_index[-1]
    group_indices.append(val_group_index)
    #get test split
    test_index = np.array(df[val_edge_end:].index)
    test_data = np.stack([df[val_edge_end:].src,df[val_edge_end:].dst]).T
    threshold = get_staleness_constraint(test_data, train_param['batch_size'])
    test_group_index = adaptive_split(threshold, test_data, test_index)
    test_group_index += val_group_index[-1]
    group_indices.append(test_group_index)
    
    print('start training.') 
    q = queue.Queue(maxsize=1)
    flag1 = False
    flag2 = False
    if mailbox is None and gnn_param['arch'] == 'transformer_attention':
        flag2 = True
    ch_reset()
    _TEP = int(os.environ.get('QTEM_TIMING_EPOCHS', '0') or 0)  # timing run: stop after $QTEM_TIMING_EPOCHS epochs
    if _TEP:
        try:
            with open('/proc/self/clear_refs', 'w') as _f:
                _f.write('5')  # reset VmHWM to the current RSS: the peaks reported below cover training only
        except OSError as _ex:
            print('[MEM] clear_refs failed: %s' % _ex, flush=True)
        torch.cuda.reset_peak_memory_stats()
        print('[MEM] before training ' + _json.dumps(dict(alloc_gb=torch.cuda.memory_allocated() / 1e9,
                                                          **_proc_mem())), flush=True)
    for e in range(train_param['epoch']):
        _set_phase('train_epoch_%d' % e)
        _BD.update(dict.fromkeys(_BD_KEYS, 0.0))
        epoch_stats_start = snapshot_stats()
        print('Epoch {:d}:'.format(e))
        _tw0 = time.time()  # wall time of the epoch's training loop, until the prefetching thread is joined
        ret_list = []
        neg_list = []
        node_list = []
        ts_list = []
        time_sample = 0
        time_prep = 0
        time_tot = 0
        total_loss = 0
        # training
        model.train()
        if sampler is not None:
            sampler.reset()
        if mailbox is not None:
            mailbox.reset()
            model.memory_updater.last_updated_nid = None
        #sample for all
        t_start = time.time()
        for _, rows in df[:train_edge_end].groupby(train_group_index):
            t0 = time.time()
            neg = neg_link_sampler.sample(len(rows))
            root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg]).astype(np.int32)
            ts = np.concatenate([rows.time.values, rows.time.values, rows.time.values]).astype(np.float32)
            t1 = time.time()
            if sampler is not None:
                if 'no_neg' in sample_param and sample_param['no_neg']:
                    pos_root_end = root_nodes.shape[0] * 2 // 3
                    sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
                    ret = sampler.get_ret()
                    t2 = time.time()
                    ret_list.append(ret)
                    node_list.append(root_nodes)
                    ts_list.append(ts)
                    flag1 = True
                    time_sample += t2 - t1
                else:
                    sampler.sample(root_nodes, ts)
                    ret = sampler.get_ret()
                    t2 = time.time()
                    ret_list.append(ret)
                    node_list.append(root_nodes)
                    ts_list.append(ts)
                    time_sample += t2 - t1
            else:
                node_list.append(root_nodes)
                ts_list.append(ts)
                flag1 = True
        
        if sampler is not None:
            total = len(ret_list)
        else:
            total = len(node_list)
        t_prep_s = time.time()
        print('sample & finish, start the subthread.')
        prep_thread = threading.Thread(target = preparation, args = \
        (ret_list, node_list, ts_list, node_feats, edge_feats, sample_param['history'], total, flag1, flag2, q,))
        prep_thread.start()
        t_end_s = time.time()
        time_prep += t_prep_s - t_start
        time_tot += t_end_s - t_start
        
        #start the main computation
        count = 0
        for _, rows in df[:train_edge_end].groupby(train_group_index):
            t0 = time.time()
            root_nodes = node_list[count]
            ts = ts_list[count]
            count += 1
            #get_data
            mfgs, uni_node, inv_node, uni_edge, inv_edge, uni_node_r, inv_node_r, edge_r, node_data, edge_data = q.get()
            _tq = time.time()
            _BD['main_wait'] += _tq - t0
            #refresh uni_mem
            if mailbox is not None:
                uni_mem, uni_mem_ts, uni_mem_input, uni_mail_ts = \
                mailbox.prep_input_mails_selection(mfgs[0], uni_node)
            # to_cuda 
            if flag2:
                if node_data is not None:
                    node_data = node_feats.decode_cuda(node_data) if isinstance(node_feats, CODEC_TABLES) else node_data.cuda()
                if edge_data is not None:
                    edge_data = edge_feats.decode_cuda(edge_data) if isinstance(edge_feats, CODEC_TABLES) else edge_data.cuda()
            if mailbox is not None: 
                ch_add('dynamic_state', 'h2d', phase_tag, uni_mem.nbytes + uni_mem_ts.nbytes + uni_mem_input.nbytes + uni_mail_ts.nbytes)
                uni_mem, uni_mem_ts, uni_mem_input, uni_mail_ts = \
                uni_mem.cuda(), uni_mem_ts.cuda(), uni_mem_input.cuda(), uni_mail_ts.cuda()
            t3 = time.time()
            _BD['main_h2d'] += t3 - _tq
            #reconstruct input
            mfgs = feat_reconstruct(mfgs, node_data, edge_data, inv_node, inv_edge)
            if mailbox is not None:
                mailbox.reconstruct(mfgs[0], uni_mem, uni_mem_ts, uni_mem_input, uni_mail_ts, inv_node)
            t1 = time.time()
            _BD['main_recon'] += t1 - t3
            time_prep += t1-t0
            #start pipelining
            optimizer.zero_grad()
            pred_pos, pred_neg = model(mfgs)
            loss = creterion(pred_pos, torch.ones_like(pred_pos))
            loss += creterion(pred_neg, torch.zeros_like(pred_neg))
            total_loss += float(loss) * train_param['batch_size']
            loss.backward()
            optimizer.step()
            del mfgs[0]
            t2 = time.time()
            _BD['main_compute'] += t2 - t1
            if mailbox is not None:      
                if edge_feats is None:
                    mem_edge_feats = None
                elif isinstance(edge_feats, CODEC_TABLES):
                    mem_edge_feats = edge_feats.gather_dequant_cuda(torch.from_numpy(np.asarray(rows['Unnamed: 0'].values)))
                else:
                    mem_edge_feats = (torch.index_select(edge_feats, 0, torch.from_numpy(rows['Unnamed: 0'].values).long()) if args.rowgather else edge_feats[rows['Unnamed: 0'].values]).cuda()
                if edge_feats is not None:
                    ch_add('static_edge', 'h2d', phase_tag, payload_bytes(edge_feats, len(rows)))
                root_nodes_gpu = torch.from_numpy(root_nodes).cuda()
                ts_gpu = torch.from_numpy(ts).cuda()
                #push update to GPU
                mail_nid, mail, mail_ts = mailbox.push_update_mailbox(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes_gpu, ts_gpu, mem_edge_feats, edge_r, uni_node_r, inv_node_r)
                mem_nid, mem, mem_ts = mailbox.push_update_memory(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes_gpu, model.memory_updater.last_updated_ts)
                #transfer the update result back to CPU
                mailbox.update_mailbox_trans(mail_nid, mail, mail_ts)
                mailbox.update_memory_trans(mem_nid, mem, mem_ts)
            t3 = time.time()
            time_prep += t3-t2
            time_tot += t3-t0
            _BD['main_memupd'] += t3 - t2
        prep_thread.join()
        _tw1 = time.time()
        print('\ttotal time:{:.2f}s prep time:{:.2f}s sample time:{:.2f}s'.format(time_tot, time_prep, time_sample))
        print('[EPOCH_TIME] ' + _json.dumps({'epoch': e, 'total': time_tot, 'prep': time_prep, 'sample': time_sample}), flush=True)
        print('[EPOCH_BREAKDOWN] ' + _json.dumps(dict(epoch=e, **_BD)), flush=True)
        print('[PEAK_MEM] ' + _json.dumps({'epoch': e, 'max_alloc_gb': torch.cuda.max_memory_allocated() / 1e9, 'max_reserved_gb': torch.cuda.max_memory_reserved() / 1e9}), flush=True)
        print('\t[timing] wall train-loop time:{:.2f}s'.format(_tw1 - _tw0), flush=True)
        if _TEP:
            print('[MEM] ' + _json.dumps(dict(epoch=e, max_alloc_gb=torch.cuda.max_memory_allocated() / 1e9,
                                              max_reserved_gb=torch.cuda.max_memory_reserved() / 1e9,
                                              **_proc_mem())), flush=True)
            if e + 1 >= _TEP:
                print('[timing] stop after {} epochs'.format(e + 1), flush=True)
                os._exit(0)
        ch_json('train_epoch_%d' % e)
        if args.qbits > 0 or args.qpack > 0:
            print(stats_line('epoch %d' % e))
        log_stats_delta('train_epoch_%d' % e, epoch_stats_start)
        if args.skip_eval:
            print('\ttrain loss:{:.4f}  [timing harness: validation skipped]'.format(total_loss))
            continue
        ap, auc = eval(group_indices, 'val')
        if e > 2 and ap > best_ap:
            best_e = e
            best_ap = ap
            torch.save(model.state_dict(), path_saver)
        print('\ttrain loss:{:.4f}  val ap:{:4f}  val auc:{:4f}'.format(total_loss, ap, auc))
        
    _mode = ('pack' if args.qpack > 0 else ('q%d' % args.qbits if args.qbits > 0 else ('u8' if args.boolfeat > 0 else ('fp16' if args.half > 0 else 'fp32')))) + ('+pin' if args.pin > 0 else '') + ('+gpu' if args.qgpu > 0 else '') + ('+rowgather' if args.rowgather > 0 else '')
    if args.skip_eval:
        ch_json('final_cumulative')
        print('[R8] DONE mode=%s skip_eval=1' % _mode, flush=True)
        raise SystemExit(0)
    if os.path.exists(path_saver):
        print('Loading model at epoch {}...'.format(best_e))
        model.load_state_dict(torch.load(path_saver))
    else:
        print('[QTEM] fewer than 3 epochs: no checkpoint saved; evaluating final-epoch weights')
    model.eval()
    if sampler is not None:
        sampler.reset()
    if mailbox is not None:
        mailbox.reset()
        model.memory_updater.last_updated_nid = None
        eval(group_indices, 'train')
        eval(group_indices, 'val')
    ap, auc = eval(group_indices, 'test')
    if args.eval_neg_samples > 1:
        print('\ttest AP:{:4f}  test MRR:{:4f}'.format(ap, auc))
    else:
        print('\ttest AP:{:4f}  test AUC:{:4f}'.format(ap, auc))
    if args.qbits > 0 or args.qpack > 0:
        print(stats_line('final'))
    ch_json('final_cumulative')
    print('[R8] DONE mode=%s skip_eval=0' % _mode, flush=True)
