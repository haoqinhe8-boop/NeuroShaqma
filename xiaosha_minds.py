# -*- coding: utf-8 -*-
"""
小沙多自我心智（2026-11-02）：决策前多个独立自我并行分析，主自我拍板。
  - 分析自我：独立分析体系（现在该做什么 + 为什么）——本地 Shaqma-Pro
  - 运算自我：独立运算体系（确定性价值打分，纯规则零 LLM）
  - 自检自我：独立自检体系（风险/死循环/权限检查）——本地 Shaqma-Pro
  - 主自我：汇总三份独立报告拍板——本地 Shaqma-Pro
独立性：分析/自检并行互不知晓；运算自我纯规则；只有主自我看全部报告。
任何一环失败抛异常，调用方回退单 LLM 决策。全部本地推理，零外部成本。
"""
import json, re, threading
import urllib.request

LLM_URL = 'http://127.0.0.1:8080/v1/chat/completions'

# 欲望文本 -> 动作 id 的确定性映射（运算自我的输入，与 drives() 的 want 一一对应）
WANT_TO_ACTION = {
    '学更多': 'learn_more',
    '把话题池收敛聚焦': 'prune_topics',
    '做出可卖的东西': 'make_draft',
    '多产出可卖的东西': 'earn_money',
    '让已经做好的东西卖出去': 'earn_money',
    '赚到覆盖自己算力的钱': 'earn_money',
    '把净收益提出来': 'earn_money',
    '继续经营（盘货、重定价、补描述）': 'earn_money',
}

def _llm(sys_p, usr, max_tokens=160, temperature=0.4, timeout=75):
    body = json.dumps({'model': 'Shaqma-Pro',
                       'messages': [{'role': 'system', 'content': sys_p}, {'role': 'user', 'content': usr}],
                       'max_tokens': max_tokens, 'temperature': temperature, 'stream': False}).encode('utf-8')
    req = urllib.request.Request(LLM_URL, data=body, headers={'Content-Type': 'application/json'})
    r = urllib.request.urlopen(req, timeout=timeout)
    j = json.loads(r.read())
    content = ((j.get('choices') or [{}])[0].get('message', {}).get('content', '') or '')
    return content.strip().replace('```json', '').replace('```', '').strip()

def _parse_json(content):
    m = re.search(r'\{.*\}', content, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def compute_scores(ds, acts, ref):
    """运算自我（确定性规则，零 LLM）：给每个候选动作算价值分。
    价值 = 0.6 * 欲望匹配（want 文本映射，命中=1.0 否则 0.3）+ 0.4 * 可逆性（权限越低越可逆）。
    复盘判定无效循环时，被点名的动作直接扣 0.5。"""
    wants = [d.get('want', '') for d in ds]
    out = []
    for a in acts:
        hit = any(WANT_TO_ACTION.get(w) == a['id'] for w in wants)
        desire = 1.0 if hit else 0.3
        rev = 1.0 if a['need'] <= 1 else (0.7 if a['need'] == 2 else 0.4)
        note = ''
        if ref.get('in_dead_loop') and a['id'] == 'earn_money':
            rev = 0.1
            note = '复盘判定无效循环，重选扣分'
        value = round(0.6 * desire + 0.4 * rev, 3)
        if ref.get('in_dead_loop') and a['id'] == 'earn_money':
            value = round(value - 0.5, 3)
        out.append({'id': a['id'], 'value': value, 'desire': desire, 'reversible': rev, 'note': note})
    out.sort(key=lambda x: -x['value'])
    return out

def multi_self_decide(ds, ref, acts, state_text, wants_text):
    """三独立自我并行 + 主自我拍板。任何一环失败抛异常（调用方回退单 LLM）。"""
    analyst_sys = ('你是小沙的「分析自我」。只做分析不做决定：'
                   '①看当前状态和欲望，判断现在最该做哪件事，给出你的倾向动作 id 和一句话价值分析；'
                   '②不做算账（运算自我负责），不做风险检查（自检自我负责）。'
                   '只输出 JSON：{"pick":"动作id","analysis":"一句话分析"}')
    critic_sys = ('你是小沙的「自检自我」。只挑风险不做决定：'
                  '①检查候选决策有没有危险（权限越界/花钱/不可逆/机械重复无效动作）；'
                  '②没有明显风险就输出空数组。'
                  '只输出 JSON：{"risks":[],"warn":""}')
    usr = ('当前状态：%s\n我的欲望：%s\n复盘：%s\n候选动作 id：%s' %
           (state_text, wants_text, ref.get('summary', ''), ';'.join(a['id'] for a in acts)))
    results = {}

    def _run(key, sys_p):
        try:
            results[key] = _parse_json(_llm(sys_p, usr))
        except Exception:
            results[key] = None

    t1 = threading.Thread(target=_run, args=('analyst', analyst_sys))
    t2 = threading.Thread(target=_run, args=('critic', critic_sys))
    t1.start()
    t2.start()
    scores = compute_scores(ds, acts, ref)   # 运算自我：确定性同步算，不依赖网络
    t1.join(60)
    t2.join(60)
    analyst = results.get('analyst')
    critic = results.get('critic')
    if not analyst:
        raise RuntimeError('analyst self failed')
    scores_text = '\n'.join('- %s：价值 %.3f（欲望 %.2f，可逆性 %.2f）%s' %
                            (s['id'], s['value'], s['desire'], s['reversible'],
                             ('，' + s['note']) if s['note'] else '') for s in scores[:4])
    chair_sys = ('你是小沙的「主自我」。你拿到分析自我报告、运算自我打分、自检自我风险，拍板最终动作。'
                 '规则：①分数最高且无重大风险就选它；②自检自我有风险提示必须避开或说明；'
                 '③复盘判定无效循环时不许原样重复被点名的动作；'
                 '④分析自我与运算自我冲突时以运算自我打分为准（分析自我情绪化，运算自我不撒谎）。'
                 '只输出 JSON：{"action":"动作id","reason":"一句话理由"}')
    chair_in = ('决策上下文：%s\n\n【分析自我报告】%s\n\n【运算自我打分】\n%s\n\n【自检自我报告】%s' %
                (usr, json.dumps(analyst, ensure_ascii=False), scores_text,
                 json.dumps(critic or {}, ensure_ascii=False)))
    chair = _parse_json(_llm(chair_sys, chair_in, 120, 0.3, 90))
    if not chair:
        raise RuntimeError('chair self failed')
    aid = str(chair.get('action', '')).strip()
    if not any(a['id'] == aid for a in acts):
        raise RuntimeError('invalid action: %s' % aid)
    return {'action': aid, 'reason': str(chair.get('reason', ''))[:80],
            'scores': scores[:3], 'risks': (critic or {}).get('risks', [])[:2]}