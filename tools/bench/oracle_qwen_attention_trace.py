"""Independent FP64 causal attention over captured public BF16 Q and stored rk4v4 K/V.

The orthogonal H256/16 is applied to Q in FP64, equivalent to decoding K back
to original coordinates. No transient query quantization or CUDA arithmetic is reused.
Only column zero (the common, undrafted input) is qualified.
"""
import argparse
import array
import json
import math
import struct
from pathlib import Path


def load(folder):
    records=[json.loads(s) for s in (folder/'index.jsonl').read_text().splitlines()]
    records={r['name']:r for r in records if r['kind']=='attention'}
    data={}
    for name,r in records.items():
        raw=(folder/r['file']).read_bytes()
        if r['dtype']==0:
            bits=array.array('H'); bits.frombytes(raw)
            wide=array.array('I',(b<<16 for b in bits))
            val=array.array('f'); val.frombytes(wide.tobytes())
        elif r['dtype']==6: val=[x[0] for x in struct.iter_unpack('<e',raw)]
        elif r['dtype']==2:
            val=array.array('i'); val.frombytes(raw)
        else: val=raw
        data[name]=val
    return records,data


def logical_rows(d):
    n=d['positions'][0]+1; stride=128
    table=d['tables'][d['rows'][0]*stride:]
    keys=[]; values=[]
    codes=[6,18,31,44,58,75,96,127]
    for h in range(4):
        kh=[]; vh=[]
        for p in range(n):
            row=(table[p//64]*4+h)*64+p%64
            kb=d['k_pages'][row*128:(row+1)*128]
            vb=d['v_pages'][row*128:(row+1)*128]
            ks=d['k_scales'][row*4:(row+1)*4]
            vs=d['v_scales'][row*8:(row+1)*8]
            kr=[]; vr=[]
            for j in range(256):
                ki=(kb[j//2]>>(4*(j%2)))&15
                vi=(vb[j//2]>>(4*(j%2)))&15
                kr.append((-1 if ki&8 else 1)*codes[ki&7]*ks[j//64])
                vr.append((vi if vi<8 else vi-16)*vs[j//32])
            kh.append(kr); vh.append(vr)
        keys.append(kh); values.append(vh)
    return keys,values


def oracle(d,keys,values):
    out=[]
    for h in range(24):
        q=list(d['q'][h*256:(h+1)*256])
        span=1
        while span<256:
            for base in range(0,256,2*span):
                for j in range(span):
                    a,b=q[base+j],q[base+j+span]
                    q[base+j],q[base+j+span]=a+b,a-b
            span*=2
        q=[v/16 for v in q]
        kh,vh=keys[h//6],values[h//6]
        scores=[sum(x*y for x,y in zip(q,k))/16 for k in kh]
        m=max(scores); probs=[math.exp(s-m) for s in scores]; denom=sum(probs)
        acc=[0.0]*256
        for prob,v in zip(probs,vh):
            for j in range(256): acc[j]+=prob*v[j]
        for j,a in enumerate(acc):
            out.append(a/denom/(1+math.exp(-d['gate'][h*256+j])))
    return out


def stats(actual,reference):
    errors=[a-b for a,b in zip(actual,reference)]
    l2=math.sqrt(sum(e*e for e in errors)/sum(v*v for v in reference))
    maximum=max(map(abs,errors)); limit=.0011+(2/256+2**-9)*max(map(abs,reference))
    return {'relative_l2':l2,'relative_l2_limit':.00325+2**-9,
            'max_abs':maximum,'gross_limit':limit,
            'passes':l2<=.00325+2**-9 and maximum<=limit}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ['plain','verify','out']:p.add_argument('--'+arg,type=Path,required=True)
    a=p.parse_args(); _,pd=load(a.plain); _,vd=load(a.verify)
    pk,pv=logical_rows(pd); vk,vv=logical_rows(vd)
    equal={name:list(pd[name][:count])==list(vd[name][:count])
           for name,count in [('q',6144),('gate',6144),('k',1024),('v',1024)]}
    equal.update(logical_k=pk==vk,logical_v=pv==vv)
    result={'position':pd['positions'][0],'common_inputs_equal':equal,
            'plain':stats(pd['output'][:6144],oracle(pd,pk,pv)),
            'verify':stats(vd['output'][:6144],oracle(vd,vk,vv)),
            'output_different':sum(x!=y for x,y in zip(pd['output'][:6144],vd['output'][:6144]))}
    a.out.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))


if __name__=='__main__':main()
