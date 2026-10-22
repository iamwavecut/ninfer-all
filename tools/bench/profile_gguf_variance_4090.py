"""Paired free/fixed-clock Nsight Systems captures of the identical Engine binary."""
import argparse
import json
import subprocess
import threading
import time
from pathlib import Path
from control_gguf_4090 import sample, smi


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['nsys','exe','weights','out']: p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--prompt',type=int,default=8192)
    p.add_argument('--reps',type=int,default=1)
    p.add_argument('--captures',type=int,default=2)
    p.add_argument('--modes',nargs='+',choices=['free','fixed'],default=['free','fixed'])
    p.add_argument('--dflash2',action='store_true')
    p.add_argument('--wddm',action='store_true')
    p.add_argument('--cpu-context',action='store_true')
    p.add_argument('--whole-process',action='store_true')
    a=p.parse_args()
    if a.reps!=1 and not a.whole_process:p.error('measured capture requires one repetition; use --whole-process')
    a.out.mkdir(parents=True,exist_ok=False)
    measurements=[]; telemetry=[]; stop=threading.Event()
    def monitor():
        while not stop.is_set():
            telemetry.append(sample()); stop.wait(.5)
    t=threading.Thread(target=monitor); t.start()
    try:
        smi('-pl','450')
        for mode in a.modes:
            if mode=='free': smi('-rgc'); smi('-rmc')
            else: smi('-lgc','2100,2100'); smi('-lmc','10501,10501')
            for rep in range(1,a.captures+1):
                started=time.time()
                while float(sample()['temperature.gpu'])>52:
                    if time.time()-started>240: raise RuntimeError('thermal gate timeout')
                    time.sleep(1)
                stem=a.out/f'{mode}-{rep}'
                cmd=[str(a.nsys),'profile','--trace='+('cuda,nvtx,wddm' if a.wddm else 'cuda,nvtx'),
                     '--sample=none','--cpuctxsw='+('process-tree' if a.cpu_context else 'none'),
                     '--cuda-graph-trace=node','--capture-range='+('none' if a.whole_process else 'cudaProfilerApi'),
                     *([] if a.whole_process else ['--capture-range-end=stop']),
                     '--force-overwrite=true','--output',str(stem),
                     str(a.exe),'--weights',str(a.weights),'-pg',f'{a.prompt},128','-r',str(a.reps),'--warmup','2',
                     '--kv-dtype','rk4v4','--max-ctx','16384',
                     '-o','json','--output-file',str(stem)+'.json']
                if not a.whole_process:cmd+=['--profile-measured']
                if a.dflash2:cmd+=['--spec','dflash2','--draft-tokens','5','--ngram-draft-tokens','0']
                entry=sample(); start=time.time()
                r=subprocess.run(cmd,capture_output=True,text=True,encoding='utf-8',errors='replace')
                Path(str(stem)+'.log').write_text(r.stdout+r.stderr,encoding='utf-8')
                measurements.append({'mode':mode,'rep':rep,'command':cmd,'entry':entry,
                                     'start':start,'end':time.time(),'exit':r.returncode})
                if r.returncode: raise RuntimeError('Nsight failed: '+str(stem))
                subprocess.run([str(a.nsys),'export','--type=sqlite','--force-overwrite=true',
                                '--output',str(stem)+'.sqlite',str(stem)+'.nsys-rep'],check=True)
                print(mode,rep,'complete',flush=True)
    finally:
        stop.set(); t.join()
        (a.out/'manifest.json').write_text(json.dumps(measurements,indent=2),encoding='utf-8')
        (a.out/'telemetry.json').write_text(json.dumps(telemetry,indent=2),encoding='utf-8')
        smi('-rgc'); smi('-rmc'); smi('-pl','450')


if __name__=='__main__': main()
