"""Compare full stored BF16 target logits and audit greedy acceptance, without NumPy."""
import argparse
import array
import json
import math
import heapq
import re
import struct
from pathlib import Path


class NonfiniteLogitsError(ValueError):
    """Numerical failure: exact verification cannot be judged from nonfinite logits."""


def tensor(folder, record):
    a=array.array('H' if record['name'] in ['logits','hidden'] else 'i')
    a.frombytes((folder/record['file']).read_bytes())
    return a


def groups(folder):
    data={}
    for line in (folder/'index.jsonl').read_text(encoding='utf-8').splitlines():
        r=json.loads(line); data.setdefault(r['group'],{'kind':r['kind']})[r['name']]=r
    return data


def floats(values):
    bits=array.array('I',(v<<16 for v in values)); f=array.array('f'); f.frombytes(bits.tobytes())
    return f


def compare(a,b):
    af,bf=floats(a),floats(b)
    dif=[abs(x-y) for x,y in zip(af,bf)]
    topa=heapq.nsmallest(5,range(len(af)),key=lambda i:(-af[i],i))
    topb=heapq.nsmallest(5,range(len(bf)),key=lambda i:(-bf[i],i))
    first=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),None)
    return {'different_elements':sum(x!=y for x,y in zip(a,b)),
            'first_differing_vocab_id':first,
            'first_differing_values':[af[first],bf[first]] if first is not None else None,
            'max_abs_delta':max(dif),
            'rms_delta':math.sqrt(sum(d*d for d in dif)/len(dif)),
            'plain_top5':[[i,af[i]] for i in topa],
            'verify_top5':[[i,bf[i]] for i in topb],
            'plain_margin':af[topa[0]]-af[topa[1]],'verify_margin':bf[topb[0]]-bf[topb[1]]}


def audit_verification(positions, logits, vocab, targets, drafts, licensed, count, accepted,
                       valid_width=None):
    width = valid_width if valid_width is not None else next(
        (j for j,p in enumerate(positions) if p!=positions[0]+j),len(positions))
    if not 1 <= width <= len(positions) or len(logits) < width*vocab:
        raise ValueError('invalid trace width/logit extent')
    if min(len(targets),len(licensed)) < width:
        raise ValueError('truncated verification metadata')
    available=min(len(drafts),width-1)
    expected=next((j for j in range(available) if drafts[j]!=targets[j]),available)
    errors=[]
    if not 1<=count<=width: errors.append('licensed_count_out_of_bounds')
    if accepted!=expected: errors.append('wrong_accepted_prefix')
    if count!=accepted+1: errors.append('wrong_bonus_count')
    columns=[]
    for j in range(width):
        f=floats(logits[j*vocab:(j+1)*vocab])
        if not all(math.isfinite(x) for x in f): raise NonfiniteLogitsError('nonfinite target logits')
        argmax=max(range(vocab),key=lambda i:f[i])
        if targets[j]!=argmax: errors.append(f'wrong_target_argmax:{j}')
        if j<count and licensed[j]!=argmax: errors.append(f'wrong_licensed_token:{j}')
        columns.append({'column':j,'target':targets[j],
                        'licensed':licensed[j] if j<count else None,'raw_argmax':argmax,
                        'draft':drafts[j] if j<available else None})
    return {'position':positions[0],'count':count,'accepted':accepted,
            'valid_width':width,'valid_width_source':'explicit' if valid_width is not None else 'legacy_positions',
            'expected_accepted':expected,'acceptance_matches_prefix':accepted==expected,
            'count_equals_accepted_plus_one':count==accepted+1,'errors':errors,'columns':columns}


def output_ids(folder):
    path=folder/'stderr.txt'
    if not path.exists():return None
    match=re.search(r'^tokens\s+generated ids\s+([\d ]+)$',path.read_text(encoding='utf-8'),re.M)
    return [int(x) for x in match.group(1).split()] if match else None


def analyze(plain,verify):
    pm={}; audits=[]; pairs=[]
    for g in groups(plain).values():
        if g['kind']!='plain': continue
        pos=tensor(plain,g['positions'])[0]
        if pos<=0: continue  # graph priming is not an actual prompt
        pm[pos]=(tensor(plain,g['ids'])[0],tensor(plain,g['logits']))
    for g in groups(verify).values():
        if g['kind']!='verify':continue
        positions=tensor(verify,g['positions']); ids=tensor(verify,g['ids'])
        if positions[0]<=0: continue
        logits=tensor(verify,g['logits']); v=g['logits']['shape'][0]
        targets=tensor(verify,g['targets']); licensed=tensor(verify,g['licensed'])
        count=tensor(verify,g['counts'])[0]; accepted=tensor(verify,g['accepted'])[0]
        drafts=tensor(verify,g['drafts'])
        width=tensor(verify,g['valid'])[0] if 'valid' in g else None
        audit=audit_verification(positions,logits,v,targets,drafts,licensed,count,accepted,width)
        for j in range(max(0,min(count,audit['valid_width']))):
            col=logits[j*v:(j+1)*v]
            if positions[j] in pm and pm[positions[j]][0]==ids[j]:
                _,a=pm[positions[j]]
                pairs.append({'position':positions[j],'input_id':ids[j],'column':j,
                              **compare(a,col)})
        audits.append(audit)
    pairs.sort(key=lambda x:x['position'])
    first=next((p for p in pairs if p['different_elements']),None)
    first_argmax=next((p for p in pairs if p['plain_top5'][0][0]!=p['verify_top5'][0][0]),None)
    cutoff=first_argmax['position'] if first_argmax else math.inf
    a_ids,b_ids=output_ids(plain),output_ids(verify)
    identity=None
    if a_ids is not None and b_ids is not None:
        first_id=next((i for i,(a,b) in enumerate(zip(a_ids,b_ids)) if a!=b),
                      min(len(a_ids),len(b_ids)) if len(a_ids)!=len(b_ids) else None)
        identity={'equal':a_ids==b_ids,'first_difference_1_based':None if first_id is None else first_id+1,
                  'plain_ids':a_ids,'verify_ids':b_ids}
        if first_id is not None:cutoff=min(cutoff,min(pm)+first_id-1)
    for pair in pairs: pair['same_observed_prefix']=pair['position']<=cutoff
    first=next((p for p in pairs if p['same_observed_prefix'] and p['different_elements']),None)
    first_argmax=next((p for p in pairs if p['same_observed_prefix'] and
                      p['plain_top5'][0][0]!=p['verify_top5'][0][0]),None)
    near=[p for p in pairs if p['same_observed_prefix'] and min(p['plain_margin'],p['verify_margin'])<=.25]
    return {'plain':str(plain),'verify':str(verify),'first_logit_difference':first,
            'first_argmax_difference':first_argmax,
            'token_identity':identity,
            'verification_errors':[{'position':r['position'],'errors':r['errors']} for r in audits if r['errors']],
            'numerical_tolerance':'No full-model logit tolerance is defined; use independent Op oracle tests, not backend parity.',
            'near_top2_threshold_absolute':.25,'near_top2_shared_prefix':near,
            'comparison_scope':'Same feed position and input ID; only comparisons up to the first argmax divergence share the observed generation prefix.',
            'compared_columns':pairs,'acceptance':audits}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plain',type=Path,required=True); p.add_argument('--verify',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    result=analyze(a.plain,a.verify)
    a.out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['first_logit_difference','first_argmax_difference']},indent=2))


if __name__=='__main__': main()
