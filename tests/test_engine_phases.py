from tools.bench.analyze_engine_phases import phase_stats,clipped_union
from tools.bench.analyze_engine_phases import analyze
import json
import sqlite3


def test_overlapping_kernels_and_boundary_crossings_do_not_double_count():
    assert clipped_union([(-10,20),(10,40),(60,120)],0,100)==[(0,40),(60,100)]
    r=phase_stats(0,100,[(0,20),(10,40),(60,100)],[])
    assert r['cuda_kernel_elapsed_union_ms']==80/1e6
    assert r['outside_target_kernel_intervals_ms']==20/1e6
    assert r['longest_uncovered_interval_ms']==20/1e6


def test_host_submit_and_wait_are_overlapping_observations_not_extra_gpu_time():
    r=phase_stats(0,100,[(20,90)],[('decode.x.submit',0,30),('decode.x.wait',30,100)])
    assert r['host_phase_wall_ms']==100/1e6
    assert r['submit_wall_ms']==30/1e6
    assert r['wait_wall_ms']==70/1e6
    assert r['cuda_kernel_elapsed_union_ms']==70/1e6


def test_no_kernel_is_explicitly_uncovered_wall_time():
    r=phase_stats(100,200,[],[])
    assert r['cuda_kernel_elapsed_union_ms']==0
    assert r['longest_uncovered_interval_ms']==r['host_phase_wall_ms']


def test_copy_and_memset_intervals_are_not_misattributed_to_cpu():
    r=phase_stats(0,100,[(20,60)],[],[(10,30),(60,90)])
    assert r['cuda_kernel_elapsed_union_ms']==40/1e6
    assert r['nonkernel_cuda_activity_ms']==40/1e6
    assert r['outside_target_cuda_activity_ms']==20/1e6


def test_explicit_measured_marker_excludes_warmup_and_later_generations(tmp_path):
    path=tmp_path/'capture.sqlite';db=sqlite3.connect(path)
    pid=(1<<48)+(17<<24);tid=pid+42
    db.executescript('''
      CREATE TABLE StringIds(id INTEGER,value TEXT);
      CREATE TABLE NVTX_EVENTS(text TEXT,start INTEGER,end INTEGER,globalTid INTEGER,textId INTEGER);
      CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,globalPid INTEGER);
      CREATE TABLE DIAGNOSTIC_EVENT(text TEXT,severity INTEGER,globalPid INTEGER);
      CREATE TABLE TARGET_INFO_SESSION_START_TIME(utcEpochNs INTEGER);
      INSERT INTO TARGET_INFO_SESSION_START_TIME VALUES(0);
    ''')
    db.executemany('INSERT INTO NVTX_EVENTS VALUES(?,?,?,?,NULL)',[
        ('bench.warmup/pp2048+tg128/0',0,90,tid),
        ('bench.measured/pp2048+tg128/0',100,1000,tid),
        ('prefill',150,450,tid),('decode',500,950,tid),
        ('decode.x.submit',500,550,tid),('decode.x.wait',550,950,tid),
        ('decode',500,950,pid+(1<<24)+42),
        ('generate',1200,2000,tid),(None,100,110,tid)])
    db.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?)',[
        (150,400,pid),(600,800,pid),(1200,2000,pid)])
    db.commit();db.close()
    path.with_suffix('.json').write_text(json.dumps({'tests':[
        {'label':'pp2048+tg128','reps':[{'timings':{'total_seconds':.001}}]}]}))
    result=analyze(path)
    assert not result['cpu_profiling_complete']
    assert not result['wddm_profiling_complete']
    assert len(result['repetitions'])==1
    phases=result['repetitions'][0]['phase_totals']
    assert phases['prefill']['cuda_kernel_elapsed_union_ms']==250/1e6
    assert phases['decode']['cuda_kernel_elapsed_union_ms']==200/1e6
    assert phases['decode']['gpu_telemetry']['samples']==0
