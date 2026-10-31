"""QTem: bounded-error 8-bit quantization on the CPU->GPU feature transfer path.

Static node/edge features stay on CPU as int8 codes + per-block fp16 scales
(per-block symmetric quantization, block = qblock contiguous dims).
Gathered rows are transferred as int8+scale and dequantized on the GPU inside
gather_dequant_cuda(). QTEM_STATS counts bytes actually moved (int8 path) vs the fp32-equivalent volume.
The sections below add fixed-width row records gathered with one row copy, GPU-resident tables (--qgpu),
lossless bit planes for binary tables (PackedFeatureTable, --qpack), and uncompressed baselines (PlainTable).
"""
import torch

QTEM_STATS = {'q_bytes': 0, 'fp32_bytes': 0, 'calls': 0}

def reset_stats():
    QTEM_STATS.update({'q_bytes': 0, 'fp32_bytes': 0, 'calls': 0})

def stats_line(tag=''):
    s = QTEM_STATS
    gb_q = s['q_bytes'] / 1e9
    gb_f = s['fp32_bytes'] / 1e9
    ratio = (gb_f / gb_q) if gb_q > 0 else float('nan')
    return ('[QTEM%s] transferred %.3f GB (int8+scale) vs %.3f GB fp32-equivalent | ratio %.2fx | gather calls %d'
            % ((' ' + tag) if tag else '', gb_q, gb_f, ratio, s['calls']))

class QuantizedFeatureTable:
    """CPU-resident int8 codes + per-block fp16 scales for a (N, d) fp32 feature table."""

    def __init__(self, feat, bits=8, block=128, chunk_rows=4_000_000):
        assert bits in (4, 8), 'supported: int8, packed int4'
        self.bits = bits
        N, d = feat.shape
        self.N, self.d, self.block = N, d, min(block, d)
        assert d % self.block == 0 and (bits == 8 or self.block % 2 == 0), 'shape not divisible'
        self.nb = d // self.block
        self.qmax = 127 if bits == 8 else 7
        # chunked construction: bounded peak memory (GDELT fp32 table is ~139 GB)
        self.scales = torch.empty(N, self.nb, dtype=torch.float16)                # CPU fp16
        self.codes = torch.empty(N, d if bits == 8 else d // 2,
                                 dtype=torch.int8 if bits == 8 else torch.uint8)
        for i in range(0, N, chunk_rows):
            j = min(i + chunk_rows, N)
            x = feat[i:j].reshape((j - i) * self.nb, self.block).float()
            amax = x.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
            self.scales[i:j] = (amax / self.qmax).to(torch.float16).reshape(j - i, self.nb)
            q = torch.round(x / self.scales[i:j].float().reshape((j - i) * self.nb, 1)).clamp(-self.qmax, self.qmax)
            if bits == 8:
                self.codes[i:j] = q.to(torch.int8).reshape(j - i, d)
            else:
                u = (q.to(torch.int16) + 8).to(torch.uint8).reshape(j - i, self.nb, self.block // 2, 2)
                self.codes[i:j] = (u[..., 0] | (u[..., 1] << 4)).reshape(j - i, d // 2)
        self._deq = None

    @property
    def shape(self):
        return (self.N, self.d)

    def __getitem__(self, idx):
        """Dequantized fp32 rows on CPU (eval / mailbox consumers see identical values)."""
        if self._deq is None:
            s = self.scales.reshape(self.N, self.nb, 1).float()
            if self.bits == 8:
                self._deq = (self.codes.reshape(self.N, self.nb, self.block).float() * s).reshape(self.N, self.d)
            else:
                u = self.codes.reshape(self.N, self.nb, self.block // 2)
                lo = (u & 0x0F).to(torch.int16) - 8
                hi = (u >> 4).to(torch.int16) - 8
                qq = torch.stack([lo, hi], dim=-1).reshape(self.N, self.nb, self.block).float()
                self._deq = (qq * s).reshape(self.N, self.d)
        return self._deq[idx]

    def gather_dequant_cuda(self, uni):
        """Quantized transfer: int8 codes + scales CPU->GPU, dequant on GPU."""
        c = self.codes[uni.long()]                     # (R, d) int8 or (R, d/2) packed, CPU
        s = self.scales[uni.long()]                    # (R, nb) fp16, CPU
        QTEM_STATS['q_bytes'] += int(c.nbytes) + int(s.nbytes)
        QTEM_STATS['fp32_bytes'] += int(c.shape[0]) * self.d * 4
        QTEM_STATS['calls'] += 1
        c = c.cuda()
        s = s.cuda()
        if self.bits == 8:
            out = c.reshape(-1, self.nb, self.block).float() * s.float().unsqueeze(-1)
        else:
            lo = (c & 0x0F).to(torch.int16) - 8
            hi = (c >> 4).to(torch.int16) - 8
            qq = torch.stack([lo, hi], dim=-1).reshape(-1, self.nb, self.block).float()
            out = qq * s.float().unsqueeze(-1)
        return out.reshape(-1, self.d)

    def full_dequant_cpu(self):
        return self[torch.arange(self.N)]


class PackedFeatureTable:
    """Lossless bit-packed bool feature table: 8 bool dims -> 1 uint8.

    Transfer path: packed uint8 rows CPU->GPU, unpacked on GPU with bit ops.
    Zero error by construction (bool 0/1 preserved exactly).
    Baseline (load_feat) converts bool->fp32, so its gather rows cost d*4 B;
    packed rows cost ceil(d/8)*1 B (GDELT edge d=182: 728B -> 23B = 31.65x).
    """

    def __init__(self, feat_bool, chunk_rows=4_000_000):
        assert feat_bool.dtype == torch.bool, 'PackedFeatureTable requires bool input'
        N, d = feat_bool.shape
        self.N, self.d = N, d
        self.dpad = (d + 7) // 8 * 8
        weights = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8)
        self.codes = torch.empty(N, self.dpad // 8, dtype=torch.uint8)
        for i in range(0, N, chunk_rows):
            j = min(i + chunk_rows, N)
            blk = feat_bool[i:j]
            if self.dpad != d:
                blk = torch.cat([blk, torch.zeros(j - i, self.dpad - d, dtype=torch.bool)], dim=1)
            self.codes[i:j] = (blk.reshape(j - i, self.dpad // 8, 8).to(torch.uint8) * weights).sum(dim=2).to(torch.uint8)

    @property
    def shape(self):
        return (self.N, self.d)

    def _to_index(self, uni):
        if isinstance(uni, torch.Tensor):
            return uni.long()
        return torch.as_tensor(uni).long()

    def _unpack_bits(self, codes):
        shifts = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.uint8, device=codes.device)
        bits = ((codes.unsqueeze(-1) >> shifts) & 1).to(torch.float32)
        return bits.reshape(-1, self.dpad)[:, :self.d]

    def __getitem__(self, idx):
        """fp32 rows on CPU (lossless unpack). Chunked over dim 0 to bound memory."""
        c = self.codes[idx]
        if not torch.is_tensor(idx):
            idx = torch.as_tensor(idx)
        if idx.dim() == 0:
            return self._unpack_bits(c.unsqueeze(0)).squeeze(0)
        return torch.cat([self._unpack_bits(c[i:i + 1_000_000]) for i in range(0, c.shape[0], 1_000_000)], dim=0)

    def gather_dequant_cuda(self, uni):
        u = self._to_index(uni).reshape(-1)
        c = self.codes[u]
        QTEM_STATS['q_bytes'] += int(c.nbytes)
        QTEM_STATS['fp32_bytes'] += int(c.shape[0]) * self.d * 4
        QTEM_STATS['calls'] += 1
        c = c.cuda()
        return self._unpack_bits(c)

    def full_dequant_cpu(self):
        raise RuntimeError('PackedFeatureTable: full dequant disabled (GDELT too large); use per-batch dispatch')


# Exact application payload counts, not a hardware PCIe bus profiler.
def snapshot_stats():
    return dict(QTEM_STATS)

def log_stats_delta(phase, before):
    import json
    after = snapshot_stats()
    delta = {k: int(after[k] - before.get(k, 0)) for k in after}
    if delta['calls']:
        print('[PAYLOAD_JSON] ' + json.dumps({'phase': phase, **delta}, sort_keys=True), flush=True)


# ---- per-channel transfer accounting (exact payload bytes at each CPU<->GPU boundary) ----
# channels: static_node, static_edge (QTem-encoded static features),
#           dynamic_state (node memory / mailbox; fp32, never encoded)
# directions: h2d (host->device), d2h (device->host); phases: train_epoch_k / eval_<mode>
import threading as _qtem_threading
QTEM_CHANNELS = {}
_CH_LOCK = _qtem_threading.Lock()


def ch_reset():
    with _CH_LOCK:
        QTEM_CHANNELS.clear()


def ch_add(channel, direction, phase, nbytes):
    nb = int(nbytes)
    with _CH_LOCK:
        d = QTEM_CHANNELS.setdefault(channel, {}).setdefault(direction, {})
        d[phase] = d.get(phase, 0) + nb


def ch_json(phase):
    import json
    with _CH_LOCK:
        snap = {c: {d: dict(p) for d, p in dd.items()} for c, dd in QTEM_CHANNELS.items()}
    print('[QTEM_CHANNELS_JSON] ' + json.dumps({'phase': phase, 'channels': snap}, sort_keys=True), flush=True)


def payload_bytes(table, n):
    """Exact bytes moved over the host->device boundary for n gathered rows."""
    if table is None:
        return 0
    n = int(n)
    if isinstance(table, QuantizedFeatureTable):
        return n * (int(table.codes.shape[1]) * table.codes.element_size()
                    + int(table.nb) * table.scales.element_size())
    if isinstance(table, PackedFeatureTable):
        return n * int(table.codes.shape[1]) * table.codes.element_size()
    return n * int(table.shape[1]) * 4


# ---- split host-side compressed gather from GPU decode; optional GPU residency ----
def _q_decode(self, c, s):
    if self.bits == 8:
        out = c.reshape(-1, self.nb, self.block).float() * s.float().unsqueeze(-1)
    else:
        lo = (c & 0x0F).to(torch.int16) - 8
        hi = (c >> 4).to(torch.int16) - 8
        qq = torch.stack([lo, hi], dim=-1).reshape(-1, self.nb, self.block).float()
        out = qq * s.float().unsqueeze(-1)
    return out.reshape(-1, self.d)


def _as_index(uni):
    return uni.long().reshape(-1) if torch.is_tensor(uni) else torch.as_tensor(uni).long().reshape(-1)


def _q_gather_compressed(self, uni):
    u = _as_index(uni)
    if getattr(self, 'on_gpu', False):
        return ('ids', u)
    return ('rows', self.codes[u], self.scales[u])


def _q_decode_cuda(self, payload):
    if payload[0] == 'ids':
        idx = payload[1].cuda()
        return _q_decode(self, self.codes[idx], self.scales[idx])
    return _q_decode(self, payload[1].cuda(), payload[2].cuda())


def _q_gather_dequant_cuda(self, uni):
    p = self.gather_compressed(uni)
    if p[0] == 'rows':
        QTEM_STATS['q_bytes'] += int(p[1].nbytes) + int(p[2].nbytes)
    QTEM_STATS['fp32_bytes'] += int(p[1].shape[0]) * self.d * 4
    QTEM_STATS['calls'] += 1
    return self.decode_cuda(p)


def _q_to_gpu(self):
    self.codes = self.codes.cuda()
    self.scales = self.scales.cuda()
    self._deq = None
    self.on_gpu = True


QuantizedFeatureTable.gather_compressed = _q_gather_compressed
QuantizedFeatureTable.decode_cuda = _q_decode_cuda
QuantizedFeatureTable.gather_dequant_cuda = _q_gather_dequant_cuda
QuantizedFeatureTable.to_gpu = _q_to_gpu


def _p_gather_compressed(self, uni):
    u = _as_index(uni)
    if getattr(self, 'on_gpu', False):
        return ('ids', u)
    return ('rows', self.codes[u])


def _p_decode_cuda(self, payload):
    if payload[0] == 'ids':
        return self._unpack_bits(self.codes[payload[1].cuda()])
    return self._unpack_bits(payload[1].cuda())


def _p_gather_dequant_cuda(self, uni):
    p = self.gather_compressed(uni)
    if p[0] == 'rows':
        QTEM_STATS['q_bytes'] += int(p[1].nbytes)
    QTEM_STATS['fp32_bytes'] += int(p[1].shape[0]) * self.d * 4
    QTEM_STATS['calls'] += 1
    return self.decode_cuda(p)


def _p_to_gpu(self):
    self.codes = self.codes.cuda()
    self.on_gpu = True


PackedFeatureTable.gather_compressed = _p_gather_compressed
PackedFeatureTable.decode_cuda = _p_decode_cuda
PackedFeatureTable.gather_dequant_cuda = _p_gather_dequant_cuda
PackedFeatureTable.to_gpu = _p_to_gpu

_payload_bytes_r8 = payload_bytes


def payload_bytes(table, n):
    """GPU-resident compressed tables move no static-feature payload over PCIe (IDs are not counted for any arm)."""
    if getattr(table, 'on_gpu', False):
        return 0
    return _payload_bytes_r8(table, n)


# ---- fixed-width row records gathered with one row copy (index_select) ----
_q_init_r9 = QuantizedFeatureTable.__init__


def _q_bind_views(self):
    cdt = torch.int8 if self.bits == 8 else torch.uint8
    self.codes = self.rec[:, :self.cb].view(cdt)
    self.scales = self.rec[:, self.cb:].view(torch.float16)


def _q_build_records(self, chunk_rows=4_000_000):
    cb = int(self.codes.shape[1]) * self.codes.element_size()
    sb = int(self.nb) * self.scales.element_size()
    assert cb % 2 == 0, 'record layout needs an even number of code bytes'
    rec = torch.empty(self.N, cb + sb, dtype=torch.uint8)
    for i in range(0, self.N, chunk_rows):
        j = min(i + chunk_rows, self.N)
        rec[i:j, :cb] = self.codes[i:j].view(torch.uint8)
        rec[i:j, cb:] = self.scales[i:j].contiguous().view(torch.uint8).reshape(j - i, sb)
    self.rec, self.cb, self.w = rec, cb, cb + sb
    _q_bind_views(self)


def _q_init_r10(self, *a, **k):
    _q_init_r9(self, *a, **k)
    _q_build_records(self)


def _q_gather_compressed_r10(self, uni):
    u = _as_index(uni)
    if getattr(self, 'on_gpu', False):
        return ('ids', u)
    return ('rec', torch.index_select(self.rec, 0, u))


def _q_decode_rec(self, r):
    cdt = torch.int8 if self.bits == 8 else torch.uint8
    return _q_decode(self, r[:, :self.cb].view(cdt), r[:, self.cb:].view(torch.float16))


def _q_decode_cuda_r10(self, payload):
    if payload[0] == 'ids':
        return _q_decode_rec(self, self.rec[payload[1].cuda()])
    return _q_decode_rec(self, payload[1].cuda())


def _q_gather_dequant_cuda_r10(self, uni):
    p = self.gather_compressed(uni)
    if p[0] == 'rec':
        QTEM_STATS['q_bytes'] += int(p[1].nbytes)
    QTEM_STATS['fp32_bytes'] += int(p[1].shape[0]) * self.d * 4
    QTEM_STATS['calls'] += 1
    return self.decode_cuda(p)


def _q_to_gpu_r10(self):
    self.rec = self.rec.cuda()
    _q_bind_views(self)
    self._deq = None
    self.on_gpu = True


QuantizedFeatureTable.__init__ = _q_init_r10
QuantizedFeatureTable.gather_compressed = _q_gather_compressed_r10
QuantizedFeatureTable.decode_cuda = _q_decode_cuda_r10
QuantizedFeatureTable.gather_dequant_cuda = _q_gather_dequant_cuda_r10
QuantizedFeatureTable.to_gpu = _q_to_gpu_r10


def _p_gather_compressed_r10(self, uni):
    u = _as_index(uni)
    if getattr(self, 'on_gpu', False):
        return ('ids', u)
    return ('rec', torch.index_select(self.codes, 0, u))


def _p_decode_cuda_r10(self, payload):
    if payload[0] == 'ids':
        return self._unpack_bits(self.codes[payload[1].cuda()])
    return self._unpack_bits(payload[1].cuda())


def _p_gather_dequant_cuda_r10(self, uni):
    p = self.gather_compressed(uni)
    if p[0] == 'rec':
        QTEM_STATS['q_bytes'] += int(p[1].nbytes)
    QTEM_STATS['fp32_bytes'] += int(p[1].shape[0]) * self.d * 4
    QTEM_STATS['calls'] += 1
    return self.decode_cuda(p)


PackedFeatureTable.gather_compressed = _p_gather_compressed_r10
PackedFeatureTable.decode_cuda = _p_decode_cuda_r10
PackedFeatureTable.gather_dequant_cuda = _p_gather_dequant_cuda_r10


# ---- baselines: plain fp32 / fp16 / uint8 tables, pinned staging buffers ----
import queue as _r11_queue
import threading as _r11_threading

PIN = {'enabled': False, 'slots': 6}
_PIN_LOCK = _r11_threading.Lock()


class _PinPool:
    """Ring of pinned staging buffers for host-side row gathers, one pool per table. A slot is taken before
    the gather and given back after its synchronous host-to-device copy, so a buffer is never overwritten
    while a queued batch still refers to it (the prefetch queue holds at most one batch)."""

    def __init__(self, width, dtype, nslots):
        self.width, self.dtype = width, dtype
        self.bufs = [None] * nslots
        self.free = _r11_queue.Queue()
        for i in range(nslots):
            self.free.put(i)

    def take(self, rows):
        k = self.free.get()
        b = self.bufs[k]
        if b is None or b.shape[0] < rows:
            cap = max(rows, 1 << 16) if b is None else max(rows, 2 * b.shape[0])
            self.bufs[k] = b = torch.empty((cap, self.width), dtype=self.dtype).pin_memory()
        return k, b[:rows]

    def give(self, k):
        self.free.put(k)


def _pool(self, src):
    with _PIN_LOCK:
        p = getattr(self, '_pinpool', None)
        if p is None:
            p = self._pinpool = _PinPool(int(src.shape[1]), src.dtype, PIN['slots'])
    return p


def _host_rows(self, src, u):
    """Gather rows src[u] on the host with one row copy: into a pinned staging slot if enabled."""
    if PIN['enabled']:
        k, out = _pool(self, src).take(int(u.shape[0]))
        torch.index_select(src, 0, u, out=out)
        return out, k
    return torch.index_select(src, 0, u), None


def _to_dev(self, rows, k):
    d = rows.cuda()  # synchronous copy (a direct DMA from pinned memory); the slot is free afterwards
    if k is not None:
        self._pinpool.give(k)
    return d


class PlainTable:
    """Uncompressed static table stored as float32, float16, or bool (1 byte per value), with the interface
    of the QTem tables: host row gather (optionally pinned), conversion to float32 on the GPU, and optional
    GPU residency. Used for the full-precision+pinned, half-precision, and uint8 baselines."""

    def __init__(self, data):
        self.data = data.contiguous()
        self.N, self.d = int(data.shape[0]), int(data.shape[1])
        self.on_gpu = False

    @property
    def shape(self):
        return (self.N, self.d)

    def __getitem__(self, idx):
        return self.data[idx].float()

    def gather_compressed(self, uni):
        u = _as_index(uni)
        if self.on_gpu:
            return ('ids', u)
        r, k = _host_rows(self, self.data, u)
        return ('rec', r, k)

    def decode_cuda(self, payload):
        if payload[0] == 'ids':
            return self.data[payload[1].cuda()].float()
        return _to_dev(self, payload[1], payload[2] if len(payload) > 2 else None).float()

    def gather_dequant_cuda(self, uni):
        p = self.gather_compressed(uni)
        if p[0] == 'rec':
            QTEM_STATS['q_bytes'] += int(p[1].nbytes)
        QTEM_STATS['fp32_bytes'] += int(p[1].shape[0]) * self.d * 4
        QTEM_STATS['calls'] += 1
        return self.decode_cuda(p)

    def to_gpu(self):
        self.data = self.data.cuda()
        self.on_gpu = True

    def full_dequant_cpu(self):
        return self.data.float()


def _q_gather_compressed_r11(self, uni):
    u = _as_index(uni)
    if getattr(self, 'on_gpu', False):
        return ('ids', u)
    r, k = _host_rows(self, self.rec, u)
    return ('rec', r, k)


def _q_decode_cuda_r11(self, payload):
    if payload[0] == 'ids':
        return _q_decode_rec(self, self.rec[payload[1].cuda()])
    return _q_decode_rec(self, _to_dev(self, payload[1], payload[2] if len(payload) > 2 else None))


def _p_gather_compressed_r11(self, uni):
    u = _as_index(uni)
    if getattr(self, 'on_gpu', False):
        return ('ids', u)
    r, k = _host_rows(self, self.codes, u)
    return ('rec', r, k)


def _p_decode_cuda_r11(self, payload):
    if payload[0] == 'ids':
        return self._unpack_bits(self.codes[payload[1].cuda()])
    return self._unpack_bits(_to_dev(self, payload[1], payload[2] if len(payload) > 2 else None))


QuantizedFeatureTable.gather_compressed = _q_gather_compressed_r11
QuantizedFeatureTable.decode_cuda = _q_decode_cuda_r11
PackedFeatureTable.gather_compressed = _p_gather_compressed_r11
PackedFeatureTable.decode_cuda = _p_decode_cuda_r11

_payload_bytes_r10 = payload_bytes


def payload_bytes(table, n):
    """Exact static-feature payload bytes moved host->device for n gathered rows (IDs not counted)."""
    if isinstance(table, PlainTable):
        return 0 if table.on_gpu else int(n) * table.d * table.data.element_size()
    return _payload_bytes_r10(table, n)


CODEC_TABLES = (QuantizedFeatureTable, PackedFeatureTable, PlainTable)
