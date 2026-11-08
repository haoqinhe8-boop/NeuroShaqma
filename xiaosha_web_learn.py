# Copyright (c) 2026 Peter (NeuroShaqma / 小沙大脑). MIT License.
# 原作者：Peter（上海宝山，12 岁）。使用/借鉴/修改请保留本署名。
# -*- coding: utf-8 -*-
"""
小沙大脑 · 上网自学（2026-09-19）
实测出口限制：中文维基/B站/YouTube 原站均不可达；**GitHub API / Gitee API / CSDN / HF-mirror 可达**。
因此本模块只从「实测可达的白名单源」取资料，存进知识库，并**让话题自己长出来**（学到的新词/新项目 → 成为下一轮的话题）。
诚实边界：这一步是**资料积累 + 可检索**（不是"神经网络变聪明"）；要真影响行为，得把知识库喂给语言层/训练循环。
用法：
  python xiaosha_web_learn.py --once        # 跑一轮（自检用）
  python xiaosha_web_learn.py --cycles 5    # 跑 5 轮
"""
import os, sys, json, time, re, urllib.request, urllib.parse, datetime, random

HERE = os.path.dirname(os.path.abspath(__file__))
KB = os.path.join(HERE, 'knowledge-base.jsonl')     # 知识库（一条一行）
STATE = os.path.join(HERE, 'web-learn-state.json')  # 话题/统计
UA = {'User-Agent': 'XiaoshaLearn/1.0 (personal research; contact: 2731952033@qq.com)'}

SEED_TOPICS = ['llm', 'spiking neural network', 'robot learning', 'reinforcement learning',
               'diffusion model', 'quantization', 'embodied ai', 'flywire connectome',
               '3d generation', 'small language model', 'RAG', 'agent']

def get(url, timeout=20, headers=None):
    req = urllib.request.Request(url, headers=dict(UA, **(headers or {})))
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode('utf-8', 'replace')

def load_state():
    if os.path.exists(STATE):
        try:
            with open(STATE, encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {'topics': list(SEED_TOPICS), 'done': {}, 'cycles': 0, 'items': 0, 'chars': 0}

def save_state(s):
    with open(STATE, 'w', encoding='utf-8') as f:
        json.dump(s, f, ensure_ascii=False, indent=1)

def add_item(src, title, url, text, tags):
    if not text or len(text) < 40:
        return 0
    rec = {'ts': datetime.datetime.now().isoformat(timespec='seconds'), 'source': src,
           'title': title[:120], 'url': url, 'chars': len(text), 'tags': tags[:6], 'text': text[:4000]}
    with open(KB, 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    return len(text)

def learn_github(topic):
    """GitHub 搜索 API（实测可达）：拿热门仓库的简介 + 话题标签"""
    n = 0
    try:
        q = urllib.parse.quote(topic)
        j = json.loads(get('https://api.github.com/search/repositories?q=%s&sort=stars&order=desc&per_page=5' % q))
        for it in (j.get('items') or [])[:5]:
            desc = (it.get('description') or '')
            txt = '项目：%s\n简介：%s\n语言：%s ｜ star：%s ｜ 更新时间：%s\n话题：%s' % (
                it.get('full_name'), desc, it.get('language'), it.get('stargazers_count'),
                (it.get('updated_at') or '')[:10], ','.join(it.get('topics') or []))
            n += add_item('github', it.get('full_name') or '', it.get('html_url') or '', txt, it.get('topics') or [])
            tags = it.get('topics') or []
            if it.get('language'):
                tags.append(str(it['language']).lower())
            n_state_topics = tags
            if n_state_topics:
                _grow_topics(n_state_topics)
    except Exception as e:
        print('  [github] 失败：' + str(e)[:80])
    return n

def learn_gitee(topic):
    n = 0
    try:
        q = urllib.parse.quote(topic)
        j = json.loads(get('https://gitee.com/api/v5/search/repositories?q=%s&sort=stars_count&order=desc&per_page=5' % q))
        for it in (j if isinstance(j, list) else [])[:5]:
            txt = '项目：%s\n简介：%s\n语言：%s ｜ star：%s' % (
                it.get('full_name'), it.get('description'), it.get('language'), it.get('stargazers_count'))
            n += add_item('gitee', it.get('full_name') or '', it.get('html_url') or '', txt, [])
    except Exception as e:
        print('  [gitee] 失败：' + str(e)[:80])
    return n

def learn_hf(topic):
    n = 0
    try:
        q = urllib.parse.quote(topic)
        j = json.loads(get('https://hf-mirror.com/api/models?search=%s&limit=5&sort=downloads&direction=-1' % q))
        for it in (j if isinstance(j, list) else [])[:5]:
            txt = '模型：%s\n下载量：%s ｜ 点赞：%s ｜ 任务：%s ｜ 更新时间：%s' % (
                it.get('modelId') or it.get('id'), it.get('downloads'), it.get('likes'),
                it.get('pipeline_tag'), (it.get('lastModified') or '')[:10])
            n += add_item('hf-mirror', it.get('modelId') or it.get('id') or '', 'https://hf-mirror.com/' + (it.get('modelId') or it.get('id') or ''), txt, [str(it.get('pipeline_tag') or '')])
    except Exception as e:
        print('  [hf] 失败：' + str(e)[:80])
    return n

def learn_openalex(topic):
    """国际学术（2.5 亿+ 论文）：实测 200 可达"""
    n = 0
    try:
        q = urllib.parse.quote(topic)
        j = json.loads(get('https://api.openalex.org/works?search=%s&per-page=3' % q))
        for it in (j.get('results') or [])[:3]:
            inv = it.get('abstract_inverted_index') or {}
            # 把倒排索引还原成摘要（OpenAlex 的做法）
            words = []
            if inv:
                pos = {}
                for w, idxs in inv.items():
                    for i in idxs:
                        pos[i] = w
                words = [pos[i] for i in sorted(pos)][:120]
            txt = '论文：%s\n年份：%s ｜ 引用：%s ｜ 期刊：%s\n摘要：%s' % (
                it.get('title'), it.get('publication_year'), it.get('cited_by_count'),
                ((it.get('primary_location') or {}).get('source') or {}).get('display_name') or '-',
                ' '.join(words) or '(无摘要)')
            n += add_item('openalex', it.get('title') or '', it.get('doi') or (it.get('id') or ''), txt, [])
    except Exception as e:
        print('  [openalex] 失败：' + str(e)[:80])
    return n


def learn_hn():
    """Hacker News 实时科技新闻（国际）：实测 200 可达"""
    n = 0
    try:
        ids = json.loads(get('https://hacker-news.firebaseio.com/v0/topstories.json'))[:4]
        for i in ids:
            it = json.loads(get('https://hacker-news.firebaseio.com/v0/item/%s.json' % i))
            txt = '热帖：%s\n分数：%s ｜ 评论：%s ｜ 链接：%s' % (
                it.get('title'), it.get('score'), it.get('descendants'), it.get('url') or ('HN#' + str(i)))
            n += add_item('hacker-news', it.get('title') or '', it.get('url') or '', txt, [])
    except Exception as e:
        print('  [hn] 失败：' + str(e)[:80])
    return n


def learn_so(topic):
    """StackOverflow 全球技术问答：实测 200 可达"""
    n = 0
    try:
        q = urllib.parse.quote(topic)
        j = json.loads(get('https://api.stackexchange.com/2.3/search/advanced?order=desc&sort=relevance&q=%s&site=stackoverflow&pagesize=3&filter=withbody' % q))
        for it in (j.get('items') or [])[:3]:
            body = re.sub(r'<[^>]+>', ' ', str(it.get('body') or ''))
            txt = '问题：%s\n票数：%s ｜ 回答：%s\n正文：%s' % (
                it.get('title'), it.get('score'), it.get('answer_count'), re.sub(r'\s+', ' ', body)[:600])
            n += add_item('stackoverflow', it.get('title') or '', it.get('link') or '', txt, it.get('tags') or [])
    except Exception as e:
        print('  [so] 失败：' + str(e)[:80])
    return n


def learn_gdelt(topic):
    """GDELT 全球新闻（实测接口通、会 429 限流 → 退避重试 3 次，任意国家/语言）"""
    n = 0
    q = urllib.parse.quote(topic)
    url = 'https://api.gdeltproject.org/api/v2/doc/doc?query=%s&mode=artlist&maxrecords=3&format=json&sort=datedesc' % q
    for attempt in range(3):
        try:
            j = json.loads(get(url, timeout=25))
            for it in (j.get('articles') or [])[:3]:
                txt = '新闻：%s\n来源：%s ｜ 国家：%s ｜ 语言：%s ｜ 时间：%s\n链接：%s' % (
                    it.get('title'), it.get('domain'), it.get('sourcecountry'),
                    it.get('language'), it.get('seendate'), it.get('url'))
                n += add_item('gdelt-global-news', it.get('title') or '', it.get('url') or '', txt, [str(it.get('sourcecountry') or '')])
            break
        except Exception as e:
            if '429' in str(e):
                time.sleep(3 + attempt * 4)
                continue
            print('  [gdelt] 失败：' + str(e)[:80])
            break
    return n


def _grow_topics(tags):
    """让话题自己长：学到的新词进话题池（这才是"学更多"的引擎）"""
    s = load_state()
    added = []
    for t in tags:
        t = str(t).strip().lower()
        if 2 < len(t) < 40 and t not in s['topics'] and re.match(r'^[a-z0-9 \-\.\+#/]+$', t):
            s['topics'].append(t)
            added.append(t)
    if added:
        save_state(s)
    return added

def main():
    once = '--once' in sys.argv
    cycles = 1 if once else int((sys.argv[sys.argv.index('--cycles') + 1] if '--cycles' in sys.argv else 0) or 10 ** 9)
    st = load_state()
    print('[web-learn] 话题池 %d 个 ｜ 知识库已有 %d 条 / %d 字' % (len(st['topics']), st['items'], st['chars']))
    for c in range(cycles):
        # 优先挑没学过的、随机取（探索）
        pool = [t for t in st['topics'] if st['done'].get(t, 0) < 2] or st['topics']
        topic = random.choice(pool)
        print('[web-learn] 第%d轮 · 话题：%s' % (c + 1, topic), flush=True)
        got = 0
        got += learn_github(topic)
        got += learn_gitee(topic)
        got += learn_hf(topic)
        got += learn_openalex(topic)   # 国际学术（2.5 亿+ 论文，含摘要）
        got += learn_so(topic)         # 全球技术问答
        got += learn_hn()              # 国际科技新闻（不受话题限制，永远抓最新）
        got += learn_gdelt(topic)      # 全球各国新闻（429 时自动退避重试）
        st = load_state()
        st['cycles'] += 1
        st['done'][topic] = st['done'].get(topic, 0) + 1
        # 重新统计知识库真实规模（不累加估计值，直接数）
        items = chars = 0
        if os.path.exists(KB):
            with open(KB, encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        items += 1
                        chars += r.get('chars', 0)
        st['items'], st['chars'] = items, chars
        st['updated'] = datetime.datetime.now().isoformat(timespec='seconds')
        save_state(st)
        print('[web-learn]   本轮新增 %d 字 ｜ 知识库共 %d 条 / %d 字 ｜ 话题池 %d 个' % (got, items, chars, len(st['topics'])))
        if not once and c + 1 < cycles:
            time.sleep(5)
    print('[web-learn] 完成')


if __name__ == '__main__':
    main()
