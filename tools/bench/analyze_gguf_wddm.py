"""Attribute the last two measured generations in whole-process WDDM captures.

Warmups and Engine priming precede these generations. CPU state comes from
scheduling events, not a sampling estimate. GPU active time is the union of
target-process CUDA kernels; residual time does not identify another GPU client.
"""
import argparse
import bisect
import json
import sqlite3
from pathlib import Path


def union(intervals):
    total=0;end=-1
    for a,b in sorted(intervals):
        total+=max(0,b-max(a,end));end=max(end,b)
    return total


def analyze(path):
    db=sqlite3.connect(path)
    nvtx=db.execute('SELECT coalesce(n.text,s.value),n.start,n.end,n.globalTid FROM '
                    'NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id WHERE n.end IS NOT NULL').fetchall()
    generations=sorted((a,b) for name,a,b,tid in nvtx if name=='generate')[-2:]
    pid=db.execute('SELECT globalPid FROM CUPTI_ACTIVITY_KIND_KERNEL GROUP BY globalPid '
                   'ORDER BY count(*) DESC LIMIT 1').fetchone()[0]
    kernels=db.execute('SELECT start,end FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE globalPid=? '
                       'ORDER BY start',(pid,)).fetchall();starts=[a for a,b in kernels]
    timelines={}
    for tid in {tid for name,a,b,tid in nvtx if name in ['prefill','decode.dflash_round']}:
        events=db.execute('SELECT start,isSchedIn,threadState FROM SCHED_EVENTS WHERE globalTid=? '
                          'ORDER BY start',(tid,)).fetchall()
        timelines[tid]=(events,[e[0] for e in events])

    def attribution(a,b,tid):
        lo=max(0,bisect.bisect_left(starts,a)-1);hi=bisect.bisect_right(starts,b)
        active=union((max(a,x),min(b,y)) for x,y in kernels[lo:hi] if y>a)
        states={'running_ms':0,'ready_ms':0,'blocked_ms':0}
        events,times=timelines[tid];lo=max(0,bisect.bisect_right(times,a)-1)
        for i in range(lo,len(events)):
            t,on,state=events[i]
            if t>=b:break
            end=min(b,events[i+1][0] if i+1<len(events) else b)
            key='running_ms' if on else ('ready_ms' if state==1 else 'blocked_ms')
            states[key]+=max(0,end-max(a,t))/1e6
        return {'wall_ms':(b-a)/1e6,'target_kernel_active_ms':active/1e6,
                'outside_target_kernels_ms':(b-a-active)/1e6,**states}

    reps=[]
    for ga,gb in generations:
        parts={}
        for name in ['prefill','decode.dflash_round']:
            rows=[attribution(a,b,tid) for n,a,b,tid in nvtx if n==name and ga<=a and b<=gb]
            parts[name]={'intervals':rows,'totals':{k:sum(r[k] for r in rows) for k in rows[0]}}
        reps.append(parts)
    diagnostics=db.execute('SELECT text FROM DIAGNOSTIC_EVENT WHERE globalPid=? AND severity=2',(pid,)).fetchall()
    db.close()
    return {'capture':path.stem,'target_global_pid':pid,'target_warnings':[r[0] for r in diagnostics],
            'scope':'Last two generate ranges; CPU running/ready/blocked are scheduling states, not causes of GPU gaps.',
            'measured_repetitions':reps}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);a=p.parse_args()
    result=[analyze(path) for path in sorted(a.folder.glob('fixed-*.sqlite'))]
    (a.folder/'attribution.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps([{**{k:r[k] for k in ['capture','target_warnings']},
                       'reps':[{k:v['totals'] for k,v in rep.items()} for rep in r['measured_repetitions']]}
                       for r in result],indent=2))


if __name__=='__main__':main()
