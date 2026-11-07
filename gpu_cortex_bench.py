# GPU 脉冲皮层 benchmark（torch + fp16 权重优化）：测 13.9万 → 30万 → 50万 → 100万
# 约束：只用 4090 空闲显存（~3.4GB），不碰 72B 主力。权重 fp16 省一半，累加 fp32 保精度。
import torch, time, sys

def build_cortex(N, K, device, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    pre_idx = torch.randint(0, N, (N, K), device=device, dtype=torch.int32)
    # randn 直接 fp16 生成（原来默认 fp32 再 .half()，28.9万×720 时 fp32 临时 1.66GB → OOM）
    w = torch.randn(N, K, device=device, dtype=torch.float16) * (1.0 / (K ** 0.5))
    # sign 用 int8 生成（避免 float32 rand+where 的 ~1.4GB 峰值，50 万规模的关键）
    sign = (torch.randint(0, 2, (N, K), device=device, dtype=torch.int8, generator=g) * 2 - 1).half()
    w = w * sign
    del sign
    return pre_idx, w

def bench(N, K, device, steps=200, dt_check=0.9, chunk=48):
    torch.cuda.empty_cache()
    pre_idx, w = build_cortex(N, K, device)
    v = torch.zeros(N, device=device)
    theta = torch.full((N,), 1.0, device=device)
    def step(v):
        spk = v >= theta
        I = torch.zeros(N, device=device, dtype=torch.float32)
        for c in range(0, K, chunk):
            # ★ 分块 gather：spk[pre_idx] 整块会触发 int64 索引转换（28.9万×720 → 1.66GB OOM），
            #   分块后每块 [N,48] 索引临时只有 ~110MB
            I += (w[:, c:c + chunk] * spk[pre_idx[:, c:c + chunk]]).sum(dim=1, dtype=torch.float32)
        v = v * dt_check + I
        v = torch.where(spk, torch.zeros_like(v), v)
        return v, spk
    for _ in range(20): v, _ = step(v)            # warmup
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fire_total = 0
    for _ in range(steps):
        v, spk = step(v)
        fire_total += spk.sum().item()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    ms_step = dt / steps * 1000
    hz = 1000.0 / ms_step
    fire_rate = fire_total / steps / N * 100
    mem = torch.cuda.memory_allocated(device) / 1024 / 1024
    del pre_idx, w, v, theta
    torch.cuda.empty_cache()
    return ms_step, hz, fire_rate, mem

if __name__ == '__main__':
    dev = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    device = torch.device('cuda:%d' % dev)
    torch.cuda.set_device(device)
    print('device:', torch.cuda.get_device_name(device), '| phys free ~3447MB (72B 占 45GB)')
    print('%-12s %-10s %-10s %-8s %-9s %-8s' % ('neurons', 'synapses', 'ms/step', 'Hz', 'fire%', 'VRAM(MB)'))
    for N, K in [(288964, 720), (500000, 360)]:
        syn = N * K
        try:
            ms, hz, fr, mem = bench(N, K, device)
            print('%-12d %-10d %-10.2f %-8.0f %-9.2f %-8.0f' % (N, syn, ms, hz, fr, mem))
        except RuntimeError as e:
            print('%-12d %-10d OOM: %s' % (N, syn, str(e)[:70]))
            break
    print('baseline: CPU 139260/360 = 5.84ms = 171Hz | 果蝇 139255/5000万')
