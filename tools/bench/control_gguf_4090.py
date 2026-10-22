"""Identical-binary B/B experiment; exclusive GPU use, fixed clocks and thermal gate."""
from __future__ import annotations
import argparse
import json
import subprocess
import threading
import time
from pathlib import Path
from compare_gguf_iq3s_4090 import run_one


def smi(*args):
    p = subprocess.run(['nvidia-smi', *args], capture_output=True, text=True, encoding='utf-8')
    if p.returncode:
        raise RuntimeError(p.stdout + p.stderr)
    return p.stdout


def sample():
    fields = ['clocks.gr','clocks.mem','temperature.gpu','power.draw','memory.used',
              'utilization.gpu','fan.speed','clocks_event_reasons.active']
    values = smi('--query-gpu='+','.join(fields), '--format=csv,noheader,nounits').splitlines()[0].split(',')
    return {'unix_seconds':time.time(), **dict(zip(fields,[x.strip() for x in values]))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--pairs',type=int,default=3)
    p.add_argument('--modes',nargs='+',choices=['plain','dflash2'],default=['plain','dflash2'])
    p.add_argument('--entry-min-c',type=float,default=50)
    p.add_argument('--entry-max-c',type=float,default=52)
    args=p.parse_args()
    for f in [args.exe,args.weights]:
        if not f.is_absolute() or not f.is_file(): p.error('absolute existing paths required')
    args.out.mkdir(parents=True,exist_ok=False)
    before=smi('-q')
    (args.out/'gpu-before.txt').write_text(before,encoding='utf-8')
    stop=threading.Event()
    samples=[]
    def monitor():
        while not stop.is_set():
            try: samples.append(sample())
            except Exception as e: samples.append({'error':str(e),'unix_seconds':time.time()})
            stop.wait(.5)
    thread=threading.Thread(target=monitor)
    try:
        for command in [('-lgc','2100,2100'),('-lmc','10501,10501'),('-pl','450')]:
            smi(*command)
        manifest={'exe':str(args.exe),'weights':str(args.weights),'pairs':args.pairs,'modes':args.modes,
                  'gpu_clock_mhz':2100,'memory_clock_mhz':10501,'power_limit_w':450,
                  'entry_temperature_c':[args.entry_min_c,args.entry_max_c],'warmup_per_case':2,
                  'cache':'OS file cache warm; two identical Engine warmups per case; prefix cache disabled',
                  'order':'AB/BA, both arms the identical absolute executable'}
        (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        thread.start()
        conditioning=args.out/'conditioning'; conditioning.mkdir()
        run_one(args.exe,args.weights,conditioning,'plain',0,'same',warmup=2)
        for mode in args.modes:
            for pair in range(1,args.pairs+1):
                for arm in (['A','B'] if pair%2 else ['B','A']):
                    start=time.time()
                    while True:
                        entry=sample()
                        if args.entry_min_c <= float(entry['temperature.gpu']) <= args.entry_max_c: break
                        if time.time()-start>240: raise RuntimeError('temperature gate not reached: '+str(entry))
                        time.sleep(1)
                    (args.out/f'{mode}-{pair}-{arm}.entry.json').write_text(json.dumps(entry,indent=2),encoding='utf-8')
                    result=run_one(args.exe,args.weights,args.out,mode,pair,arm,warmup=2)
                    print(mode,pair,arm,'complete',result['elapsed_seconds'],flush=True)
    except Exception as error:
        (args.out/'failure.json').write_text(json.dumps({'error':str(error),
            'unix_seconds':time.time()},indent=2),encoding='utf-8')
        raise
    finally:
        stop.set()
        if thread.is_alive(): thread.join()
        (args.out/'control-telemetry.json').write_text(json.dumps(samples,indent=2),encoding='utf-8')
        # The session's original state was unlocked graphics/memory, 450 W.
        for command in [('-rgc',),('-rmc',),('-pl','450')]:
            smi(*command)
        (args.out/'gpu-after.txt').write_text(smi('-q'),encoding='utf-8')


if __name__=='__main__': main()
