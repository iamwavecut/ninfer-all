"""Phase-aligned public Engine measurements from explicit benchmark NVTX markers.

CUDA elapsed intervals are not SM busy time. Host NVTX submit/wait durations
include scheduling and driver time; they are not CPU instruction execution time.
"""
import argparse
import bisect
import json
import sqlite3
from pathlib import Path


def clipped_union(intervals,start,end):
    spans=sorted((max(start,a),min(end,b)) for a,b in intervals if a<end and b>start)
    merged=[]
    for a,b in spans:
        if merged and a<=merged[-1][1]:merged[-1]=(merged[-1][0],max(b,merged[-1][1]))
        else:merged.append((a,b))
    return merged


def phase_stats(start,end,kernels,host_ranges,transfers=()):
    spans=clipped_union(kernels,start,end)
    gpu=sum(b-a for a,b in spans)
    activities=clipped_union([*kernels,*transfers],start,end)
    device=sum(b-a for a,b in activities)
    gaps=[];cursor=start
    for a,b in spans:
        if a>cursor:gaps.append(a-cursor)
        cursor=b
    if cursor<end:gaps.append(end-cursor)
    host={}
    for label in ['submit','wait']:
        host[label+'_wall_ms']=sum(b-a for a,b in clipped_union(
            [(a,b) for name,a,b in host_ranges if name.endswith('.'+label)],start,end))/1e6
    return {'host_phase_wall_ms':(end-start)/1e6,'cuda_kernel_elapsed_union_ms':gpu/1e6,
            'cuda_activity_elapsed_union_ms':device/1e6,
            'nonkernel_cuda_activity_ms':(device-gpu)/1e6,
            'outside_target_cuda_activity_ms':(end-start-device)/1e6,
            'outside_target_kernel_intervals_ms':(end-start-gpu)/1e6,
            'longest_uncovered_interval_ms':max(gaps,default=0)/1e6,**host}


def analyze(path):
    db=sqlite3.connect(path)
    rows=db.execute('SELECT coalesce(n.text,s.value),n.start,n.end,n.globalTid '
        'FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id WHERE n.end IS NOT NULL').fetchall()
    markers=sorted((a,b,name,tid) for name,a,b,tid in rows if name and name.startswith('bench.measured/'))
    if not markers:raise ValueError('missing explicit measured-repetition markers: '+str(path))
    # Nsight global IDs reserve the low 24 bits for the thread ID. Use the benchmark's
    # own measured markers, not the process with most kernels (PROCESSES is optional).
    candidates={tid & ~((1<<24)-1) for _,_,_,tid in markers}
    if len(candidates)!=1:raise ValueError('benchmark measured markers span multiple processes')
    pid=candidates.pop()
    rows=[row for row in rows if row[3] & ~((1<<24)-1)==pid]
    kernels=db.execute('SELECT start,end FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE globalPid=? ORDER BY start',(pid,)).fetchall()
    if not kernels:raise ValueError('target CUDA kernel trace missing; attribution incomplete')
    starts=[a for a,b in kernels]
    latest_end=[];last=0
    for _,end in kernels:
        last=max(last,end);latest_end.append(last)
    warnings=[x[0] for x in db.execute('SELECT text FROM DIAGNOSTIC_EVENT WHERE severity=2 AND globalPid=?',(pid,))]
    tables={x[0] for x in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    transfers=[]
    for table in ['CUPTI_ACTIVITY_KIND_MEMCPY','CUPTI_ACTIVITY_KIND_MEMSET']:
        if table in tables:
            transfers.extend(db.execute(f'SELECT start,end FROM {table} WHERE globalPid=?',(pid,)).fetchall())
    has_sched='SCHED_EVENTS' in tables and db.execute('SELECT 1 FROM SCHED_EVENTS LIMIT 1').fetchone() is not None
    native=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    epoch=db.execute('SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME').fetchone()[0]/1e9
    telemetry_path=path.parent/'telemetry.json'
    telemetry=json.loads(telemetry_path.read_text()) if telemetry_path.exists() else []
    tests={t['label']:t for t in native['tests']};reps=[]
    for a,b,name,_ in markers:
        _,label,index=name.split('/');index=int(index)
        phases=[]
        for n,x,y,tid in rows:
            if n not in ['prefill','decode'] or x<a or y>b:continue
            lo=bisect.bisect_right(latest_end,x);hi=bisect.bisect_right(starts,y)
            host=[(rn,ra,rb) for rn,ra,rb,rt in rows if rn and x<=ra and rb<=y and (rn.endswith('.submit') or rn.endswith('.wait'))]
            phases.append({'phase':n,'start_ns':x,'end_ns':y,
                           **phase_stats(x,y,kernels[lo:hi],host,transfers)})
        totals={}
        for phase in ['prefill','decode']:
            selected=[p for p in phases if p['phase']==phase]
            totals[phase]={k:sum(p[k] for p in selected) for k in [
                'host_phase_wall_ms','cuda_kernel_elapsed_union_ms',
                'cuda_activity_elapsed_union_ms','nonkernel_cuda_activity_ms','outside_target_cuda_activity_ms',
                'outside_target_kernel_intervals_ms','submit_wall_ms','wait_wall_ms']}
            totals[phase]['longest_uncovered_interval_ms']=max((p['longest_uncovered_interval_ms'] for p in selected),default=0)
            observed=tests[label]['reps'][index]['timings'].get(phase+'_seconds')
            totals[phase]['native_engine_phase_ms']=None if observed is None else observed*1000
            totals[phase]['native_minus_nvtx_scope_ms']=None if observed is None else observed*1000-totals[phase]['host_phase_wall_ms']
            samples=[s for s in telemetry if any(epoch+p['start_ns']/1e9<=s['unix_seconds']<=epoch+p['end_ns']/1e9 for p in selected)]
            totals[phase]['gpu_telemetry']={'samples':len(samples),
                **{field:sorted({s[field] for s in samples}) for field in [
                    'clocks.gr','clocks.mem','temperature.gpu','power.draw','memory.used','clocks_event_reasons.active']}}
        ma=bisect.bisect_right(latest_end,a);mb=bisect.bisect_right(starts,b)
        marker=phase_stats(a,b,kernels[ma:mb],[],transfers)
        reps.append({'label':label,'repetition':index,'marker_wall_ms':(b-a)/1e6,
                     'marker_cuda_activity_ms':marker['cuda_activity_elapsed_union_ms'],
                     'cuda_activity_outside_phase_scopes_ms':marker['cuda_activity_elapsed_union_ms']-sum(p['cuda_activity_elapsed_union_ms'] for p in totals.values()),
                     'engine':tests[label]['reps'][index],'phase_totals':totals,'phase_intervals':phases})
    db.close()
    return {'capture':path.stem,'cpu_scheduling_events_present':has_sched,
            'cpu_profiling_complete':False,
            'cpu_profiling_status':'incomplete' if not has_sched or warnings else 'coverage_unverified',
            'cpu_profiling_limit':'Scheduling absent or target warning means incomplete. Host phase wall time is still distinct from CPU busy time.',
            'wddm_profiling_complete':False,
            'wddm_limit':'Base WDDM events do not certify complete context/preemption or residency coverage.',
            'telemetry_limit':'Session UTC epoch aligns NVTX to sampled board telemetry; sub-sample excursions remain unverified.',
            'target_warnings':warnings,'repetitions':reps}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);a=p.parse_args()
    result=[analyze(path) for path in sorted(a.folder.glob('*.sqlite'))]
    (a.folder/'phases.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    for capture in result:
        print(capture['capture'],'CPU complete:',capture['cpu_profiling_complete'])
        for r in capture['repetitions']:print(r['label'],r['repetition'],r['phase_totals'])


if __name__=='__main__':main()
