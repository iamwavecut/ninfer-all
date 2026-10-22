"""Collect raw target/verification traces with the separately instrumented CLI."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from compare_gguf_iq3s_4090 import gpu_sample, ProcessVram


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['exe','weights','prompts','out']: p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--study',choices=['stability','crosscheck','target','attention','pv32'],default='stability')
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=False)
    cases=[('long',16,True,'rk4v4',False)]
    jobs=[(*case,mode,5) for case in cases for mode in ['none','dflash2']]
    if a.study=='target':
        cases=[('long',16,True,'rk4v4',False),('cuda',40,True,'rk4v4',False),
               ('long',16,False,'rk4v4',False),('long',7,False,'rk4v4',True),
               ('long',16,True,'bf16',False)]
        jobs=[(*case,mode,5) for case in cases for mode in ['none','dflash2']]
    if a.study=='attention':
        jobs=[('long',2,False,'rk4v4',False,mode,5) for mode in ['none','dflash2']]
    if a.study=='pv32':
        jobs=[('long',16,True,'rk4v4',False,mode,5) for mode in ['none','dflash2']]
    if a.study=='stability':
        jobs=[(name,40 if name=='cuda' else 16,True,kv,False,
               'none' if k==0 else 'dflash2',k)
              for name in ['cuda','medium','long'] for kv in ['bf16','rk4v4']
              for k in [0,1,3,5,15]]
    manifest=[]
    for name,outputs,graph,kv,layers,mode,drafts in jobs:
            stem=f'{name}-{outputs}-{mode}-{kv}-'+('graph' if graph else 'eager')+('-layers' if layers else '')
            if a.study=='stability':stem+=f'-k{drafts}'
            folder=a.out/stem; folder.mkdir()
            env=os.environ.copy(); env['NINFER_TARGET_TRACE']=str(folder)
            env.pop('NINFER_TRACE_LAYERS',None)
            env.pop('NINFER_TRACE_ATTN',None)
            env.pop('NINFER_SMALLT_PV_F16',None)
            env.pop('NINFER_PROMPT_PV_F16',None)
            if a.study=='attention': env['NINFER_TRACE_ATTN']='1'
            if a.study=='pv32': env['NINFER_SMALLT_PV_F16']='0'
            if layers: env['NINFER_TRACE_LAYERS']='1'
            cmd=[str(a.exe),str(a.weights),'--messages',str(a.prompts/f'{name}.messages.json'),
                 '--max-new',str(outputs),'--max-context','8192','--kv-dtype',kv,'--greedy',
                 '--raw-output','--print-token-ids','--spec',mode]
            if mode=='dflash2': cmd+=['--draft-tokens',str(drafts),'--ngram-draft-tokens','0']
            if not graph: cmd+=['--no-cuda-graph']
            start=time.time(); samples=[]
            with (folder/'stdout.txt').open('wb') as stdout, (folder/'stderr.txt').open('wb') as stderr:
                process=subprocess.Popen(cmd,env=env,stdout=stdout,stderr=stderr)
                vram=ProcessVram(process.pid)
                while process.poll() is None:
                    s=gpu_sample()
                    if s:
                        s['unix_seconds']=time.time(); s['process_vram_mib']=vram.sample_mib(); samples.append(s)
                    time.sleep(.25)
                vram.close()
            manifest.append({'case':stem,'command':cmd,'exit':process.returncode,'start':start,'end':time.time()})
            (folder/'telemetry.json').write_text(json.dumps(samples,indent=2),encoding='utf-8')
            (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
            if process.returncode: raise RuntimeError(stem+' failed')
            print(stem,'complete',flush=True)


if __name__=='__main__': main()
