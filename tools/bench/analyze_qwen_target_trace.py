"""Compare full stored BF16 target logits and audit greedy acceptance, without NumPy."""
import argparse
import array
import json
import math
import re
import struct
from pathlib import Path


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
    topa=sorted(range(len(af)),key=lambda i:(-af[i],i))[:5]
    topb=sorted(range(len(bf)),key=lambda i:(-bf[i],i))[:5]
    first=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),None)
    return {'different_elements':sum(x!=y for x,y in zip(a,b)),
            'first_differing_vocab_id':first,
            'first_differing_values':[af[first],bf[first]] if first is not None else None,
            'max_abs_delta':max(dif),
            'rms_delta':math.sqrt(sum(d*d for d in dif)/len(dif)),
            'plain_top5':[[i,af[i]] for i in topa],
            'verify_top5':[[i,bf[i]] for i in topb],
            'plain_margin':af[topa[0]]-af[topa[1]],'verify_margin':bf[topb[0]]-bf[topb[1]]}


def analyze(plain,verify):
    pm={}; audits=[]; pairs=[]
    for g in groups(plain).values():
        if g['kind']!='plain': continue
        pos=tensor(plain,g['positions'])[0]
        pm[pos]=(tensor(plain,g['ids'])[0],tensor(plain,g['logits']))
    for g in groups(verify).values():
        if g['kind']!='verify':continue
        positions=tensor(verify,g['positions']); ids=tensor(verify,g['ids'])
        logits=tensor(verify,g['logits']); v=g['logits']['shape'][0]
        targets=tensor(verify,g['targets']); licensed=tensor(verify,g['licensed'])
        count=tensor(verify,g['counts'])[0]; accepted=tensor(verify,g['accepted'])[0]
        drafts=tensor(verify,g['drafts'])
        width=next((j for j,p in enumerate(positions) if p!=positions[0]+j),len(positions))
        available=min(len(drafts),max(0,width-1))
        expected=next((j for j in range(available) if drafts[j]!=targets[j]),available)
        audit={'position':positions[0],'count':count,'accepted':accepted,
               'available_drafts_from_positions':available,'expected_accepted':expected,
               'acceptance_matches_prefix':accepted==expected,
               'count_equals_accepted_plus_one':count==accepted+1,'columns':[]}
        for j in range(count):
            col=logits[j*v:(j+1)*v]; f=floats(col); argmax=max(range(v),key=lambda i:f[i])
            audit['columns'].append({'column':j,'target':targets[j],'licensed':licensed[j],
                                      'raw_argmax':argmax,'draft':drafts[j] if j<accepted else None})
            if positions[j] in pm and pm[positions[j]][0]==ids[j]:
                _,a=pm[positions[j]]
                pairs.append({'position':positions[j],'input_id':ids[j],'column':j,
                              **compare(a,col)})
        audits.append(audit)
    pairs.sort(key=lambda x:x['position'])
    first=next((p for p in pairs if p['different_elements']),None)
    first_argmax=next((p for p in pairs if p['plain_top5'][0][0]!=p['verify_top5'][0][0]),None)
    return {'plain':str(plain),'verify':str(verify),'first_logit_difference':first,
            'first_argmax_difference':first_argmax,
            'comparison_scope':'Same feed position and input ID. After the first argmax divergence, histories may differ; later comparisons do not isolate width effects.',
            'compared_columns':pairs,'acceptance':audits}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plain',type=Path,required=True); p.add_argument('--verify',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    result=analyze(a.plain,a.verify)
    a.out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['first_logit_difference','first_argmax_difference']},indent=2))


if __name__=='__main__': main()
