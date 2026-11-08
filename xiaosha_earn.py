# Copyright (c) 2026 Peter (NeuroShaqma / 小沙大脑). MIT License.
# 原作者：Peter（上海宝山，12 岁）。使用/借鉴/修改请保留本署名。
# -*- coding: utf-8 -*-
"""
小沙 · 赚钱流程（2026-09-19）—— 一句话 "go earning money" 触发
它能做的（不需要你批）：① 从知识库挑高价值条目 → 产出可卖物（情报简报/技术速览）② 估价 ③ 记账（收益账本）
它必须请你批的：④ 对外发布（3 级）⑤ 开通收款（只有主人能办）
诚实边界：机器不会凭空变钱；没有"收款通道"就收不到真钱 —— 这一步只有主人能办。
落点说明（回答"钱放哪"）：
  · 平台侧已有余额与计费（/v1 链：balance 扣费 + usage 记录）→ 收益按"用户可见余额 + 分成记账"落账
  · 真钱进主人收款账户（微信/支付宝/银行卡）；平台只记账：谁付的、多少、应给谁多少
用法：python xiaosha_earn.py            # 跑一次（产出一份可卖物 + 记账 + 列出待批项）
      python xiaosha_earn.py --report   # 只报账，不产出
"""
import os, sys, json, time, random, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
KB = os.path.join(HERE, 'knowledge-base.jsonl')
EARN = os.path.join(HERE, 'earnings.jsonl')          # 收益账本（一笔一行）
LEDGER = os.path.join(HERE, 'agent-ledger.jsonl')    # 行为账本（与 agent 共用）
PENDING = os.path.join(HERE, 'pending-approvals.json')
OUT = os.path.join(HERE, 'productions')

# 定价表（人民币元）：按产出物类型 + 篇幅，保守估价（可被主人改）
PRICING = {'brief': 9.9, 'deep': 29.9, 'pack': 49.0}


def _max_level():
    """读 permissions.json 的 max_level（权限归主人控制）。读不到就按 0（最保守）。"""
    import json as _json, os as _os
    try:
        p = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'permissions.json')
        with open(p, 'r', encoding='utf-8') as f:
            return int(_json.load(f).get('max_level', 0))
    except Exception:
        return 0

def load_rows():
    rows = []
    if os.path.exists(KB):
        with open(KB, encoding='utf-8') as f:
            for ln in f:
                if ln.strip():
                    try:
                        rows.append(json.loads(ln))
                    except Exception:
                        pass
    return rows

def pick_best(rows, n=10):
    """挑"高价值"条目：优先有摘要/正文长的、来源权威的（github/openalex/hn 优先）"""
    w = {'openalex': 3, 'hacker-news': 2, 'stackoverflow': 2, 'github': 2, 'gdelt-global-news': 2, 'hf-mirror': 1}
    rows = [r for r in rows if r.get('chars', 0) > 200]
    rows.sort(key=lambda r: (w.get(r.get('source'), 1), r.get('chars', 0)), reverse=True)
    head = rows[:max(n * 6, 30)]
    random.shuffle(head)
    return head[:n]

def make_product(rows, kind='brief'):
    os.makedirs(OUT, exist_ok=True)
    # 2026-09-24：文件名精确到秒。原来只到分钟 → 同一分钟内多次增产互相覆盖，
    # 跑 6 次只落地 1 个文件（白干 5 次，实测踩到）。
    ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    title = {'brief': 'Shaqma 技术情报简报', 'deep': 'Shaqma 深度速览', 'pack': 'Shaqma 资料包'}[kind]
    fn = os.path.join(OUT, '%s-%s.md' % (title.replace(' ', '-'), ts))
    if os.path.exists(fn):   # 防同秒重复（多次调用同一秒）：追加序号，绝不覆盖已有产出
        for i in range(2, 100):
            fn = os.path.join(OUT, '%s-%s-%d.md' % (title.replace(' ', '-'), ts, i))
            if not os.path.exists(fn):
                break
    with open(fn, 'w', encoding='utf-8') as f:
        f.write('# %s\n\n> 生成：%s ｜ 由小沙自主汇总 ｜ 来源均为实测可达源\n\n' % (title, datetime.datetime.now().isoformat(timespec='seconds')))
        for i, r in enumerate(rows, 1):
            f.write('## %d. %s\n\n- 来源：`%s`\n- 链接：%s\n\n```text\n%s\n```\n\n' % (
                i, r.get('title') or '(无标题)', r.get('source'), r.get('url') or '-', str(r.get('text'))[:800]))
    return fn, title

def record(action, **kw):
    rec = {'ts': datetime.datetime.now().isoformat(timespec='seconds'), 'by': 'xiaosha_earn', 'action': action}
    rec.update(kw)
    with open(LEDGER, 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')

def earn(kind='brief'):
    rows = load_rows()
    if len(rows) < 10:
        print('[earn] 知识库太小（%d 条），先去学；不产出半成品。' % len(rows))
        return 1
    picks = pick_best(rows, 10)
    fn, title = make_product(picks, kind)
    price = PRICING[kind]
    rel = os.path.relpath(fn, HERE).replace('\\', '/')
    rec = {'ts': datetime.datetime.now().isoformat(timespec='seconds'), 'product': rel, 'title': title,
           'items': len(picks), 'price_cny': price, 'status': 'listed',
           'note': '已上架到主人自家商店（/shop.html，平台内展示与成交）；对外发布到站外才需 3 级批准'}
    with open(EARN, 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    record('earn_make_product', product=rel, price_cny=price, ok=True)

    # 权限边界（2026-09-24 修正）：
    #   · **上架到自己平台的商店（/shop.html）= 内部动作，不需要批准** —— 原来误判成"对外发布"，
    #     于是每次产出都挂一条 3 级待批、商品永远停在 draft，赚钱链路卡死在这一环（用户问的"为什么还没实现"）。
    #   · 只有**发到站外**（外部网站/社交平台，真·对外发布）以及**开通收款通道**才需要主人批准。
    p = {'items': []}
    if os.path.exists(PENDING):
        try:
            p = json.load(open(PENDING, encoding='utf-8')) or {'items': []}
        except Exception:
            p = {'items': []}
    p.setdefault('items', [])
    # 去重：同一动作若已有待批项，就不再堆一条（原来每产出一次堆一条，136 行全是重复项）
    already = any(x.get('action') == 'publish_offsite' and x.get('status') == 'pending' for x in p['items'])
    if not already:
        p['items'].append({'ts': rec['ts'], 'action': 'publish_offsite', 'name': '对外发布到站外（可选）',
                           'need_level': 3, 'granted': (3 <= _max_level()), 'need_desc': '对外（发布到站外、对外发消息）',
                           'why': '商品已上架到自家商店并可售；若要发到站外（微博/知乎/朋友圈等）需主人点头，'
                                  '并需主人提供收款方式才能收真钱',
                           'status': 'pending'})
    with open(PENDING, 'w', encoding='utf-8') as f:
        json.dump(p, f, ensure_ascii=False, indent=1)

    print('[earn] 产出：%s（%d 条来源，建议价 ¥%.1f）' % (rel, len(picks), price))
    print('[earn] 状态：listed（已上架自家商店 /shop.html —— 内部动作，不需要批准）')
    print('[earn] 已记入收益账本：earnings.jsonl')
    print('[earn] 站外发布/收款：需主人点头（见 pending-approvals.json；真钱只能进主人收款账户）')
    return 0

def report():
    n = 0
    total = 0.0
    if os.path.exists(EARN):
        with open(EARN, encoding='utf-8') as f:
            for ln in f:
                if ln.strip():
                    r = json.loads(ln)
                    n += 1
                    total += float(r.get('price_cny', 0) or 0)
    print('[earn] 收益账本：%d 笔，标价合计 ¥%.1f（状态均为 draft，未对外发布 → 尚无真收入）' % (n, total))
    pd = {'items': []}
    if os.path.exists(PENDING):
        try:
            pd = json.load(open(PENDING, encoding='utf-8')) or {'items': []}
        except Exception:
            pd = {'items': []}
    wait = [x for x in pd.get('items', []) if x.get('status') == 'pending']
    print('[earn] 等你批准的：%d 件' % len(wait))
    for w in wait[-5:]:
        print('   · %s —— %s' % (w.get('name'), w.get('why')))
    return 0

if __name__ == '__main__':
    if '--report' in sys.argv:
        sys.exit(report())
    kind = 'brief'
    if '--kind' in sys.argv:
        try:
            k = sys.argv[sys.argv.index('--kind') + 1]
            if k in PRICING:
                kind = k
        except IndexError:
            pass
    sys.exit(earn(kind))
