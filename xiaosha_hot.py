# Copyright (c) 2026 Peter (NeuroShaqma / 小沙大脑). MIT License.
# 原作者：Peter（上海宝山，12 岁）。使用/借鉴/修改请保留本署名。
# -*- coding: utf-8 -*-
"""
小沙 · 热点速读（2026-09-25）—— 抓可达热榜，产出大众真正会看的内容
为什么做这个：旧的「技术情报简报」是抓 GitHub/论文拼的冷门内容，注定引不起大众注意。
  现在改成抓三个实测可达的大众热榜（知乎 / B站 / 每日新闻），产出「今日热点速读」——
  内容是大众正在关心的东西，才谈得上"引起大部分人的注意"。
诚实边界：热点本身免费公开；这份价值在于"帮你一页看完今天在聊什么"，省去逐个刷的时间。
用法：python xiaosha_hot.py            # 抓热榜 + 产出热点速读 + 记账上架
      python xiaosha_hot.py --report   # 只看账本
"""
import os, sys, json, datetime, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'productions')
EARN = os.path.join(HERE, 'earnings.jsonl')
PRICE = 2.9   # 热点速读定价：大众化、时效强，走量低价

def fetch_json(url, timeout=12):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode('utf-8', 'replace'))

def zhihu_hot(n=10):
    try:
        j = fetch_json('https://api.zhihu.com/topstory/hot-lists/total?limit=' + str(n))
        out = []
        for i, d in enumerate((j.get('data') or [])[:n]):
            t = d.get('target') or {}
            out.append({'rank': i + 1, 'title': t.get('title') or '(无标题)', 'heat': d.get('detail_text') or ''})
        return out
    except Exception:
        return []

def bili_hot(n=6):
    try:
        j = fetch_json('https://api.bilibili.com/x/web-interface/popular?ps=' + str(n))
        out = []
        for i, v in enumerate((j.get('data', {}).get('list') or [])[:n]):
            st = v.get('stat') or {}
            out.append({'rank': i + 1, 'title': v.get('title') or '(无标题)', 'heat': ('播放 ' + str(st.get('view', ''))) if st.get('view') else ''})
        return out
    except Exception:
        return []

def daily_news(n=10):
    try:
        j = fetch_json('https://60s.viki.moe/')
        news = j.get('news') or (j.get('data') or {}).get('news') or []
        return [{'rank': i + 1, 'title': str(x).strip(), 'heat': ''} for i, x in enumerate(news[:n]) if str(x).strip()]
    except Exception:
        return []

def make_hot():
    zhihu = zhihu_hot(10)
    bili = bili_hot(6)
    news = daily_news(10)
    total = len(zhihu) + len(bili) + len(news)
    if total < 5:
        print('[hot] 热榜抓取不足（%d 条），可能网络波动；本轮不产出半成品。' % total)
        return 1
    ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    fn = os.path.join(OUT, '热点速读-%s.md' % ts)
    now = datetime.datetime.now().isoformat(timespec='seconds')
    with open(fn, 'w', encoding='utf-8') as f:
        f.write('# 今日热点速读\n\n> 生成：%s ｜ 由小沙自主汇总 ｜ 来源：知乎热榜 / B站热门 / 每日新闻（实测可达源）\n\n' % now)
        if zhihu:
            f.write('## 🔥 知乎热榜 Top %d\n\n' % len(zhihu))
            for z in zhihu:
                f.write('- **%d.** %s%s\n' % (z['rank'], z['title'], (' ｜ ' + z['heat']) if z['heat'] else ''))
            f.write('\n')
        if bili:
            f.write('## 📺 B站热门 Top %d\n\n' % len(bili))
            for b in bili:
                f.write('- **%d.** %s%s\n' % (b['rank'], b['title'], (' ｜ ' + b['heat']) if b['heat'] else ''))
            f.write('\n')
        if news:
            f.write('## 📰 每日新闻速览\n\n')
            for n in news:
                f.write('- %s\n' % n['title'])
            f.write('\n')
        f.write('---\n\n> 说明：以上为公开热榜的实时汇总，供快速浏览「今天在聊什么」。真实、可溯源，不添油加醋。\n')
    rel = os.path.relpath(fn, HERE).replace('\\', '/')
    rec = {'ts': now, 'product': rel, 'title': '热点速读', 'items': total, 'price_cny': PRICE,
           'status': 'listed', 'note': '热点速读：抓公开热榜汇总，大众内容走量'}
    with open(EARN, 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print('[hot] 产出：%s（知乎 %d + B站 %d + 新闻 %d = %d 条，建议价 ¥%.1f）' % (rel, len(zhihu), len(bili), len(news), total, PRICE))
    print('[hot] 状态：listed（已上架自家商店；内容对路大众热点）')
    return 0

def report():
    n = 0
    total = 0.0
    if os.path.exists(EARN):
        with open(EARN, encoding='utf-8') as f:
            for ln in f:
                if ln.strip():
                    try:
                        r = json.loads(ln)
                        if '热点速读' in (r.get('title') or ''):
                            n += 1
                            total += float(r.get('price_cny', 0) or 0)
                    except Exception:
                        pass
    print('[hot] 热点速读账本：%d 份，标价合计 ¥%.1f' % (n, total))
    return 0

if __name__ == '__main__':
    sys.exit(report() if '--report' in sys.argv else make_hot())
