"""Audit each completed width/KV/prompt pair; optionally follow a running numerical matrix."""
import argparse
import json
import time
from pathlib import Path
from analyze_qwen_target_trace import analyze


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path)
    p.add_argument('--watch',action='store_true');a=p.parse_args();done={}
    while True:
        manifest_path=a.folder/'manifest.json'
        manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else []
        if any(r['exit'] for r in manifest):raise RuntimeError('matrix contains a failed process')
        complete={r['case'] for r in manifest}
        for case in sorted(complete):
            if '-dflash2-' not in case or case in done:continue
            prefix,tail=case.split('-dflash2-');kv,graph,k=tail.split('-')
            base=f'{prefix}-none-{kv}-{graph}-k0'
            if base not in complete:continue
            result=analyze(a.folder/base,a.folder/case)
            (a.folder/(case+'.analysis.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
            done[case]={'case':case,'token_identity':result['token_identity'],
                        'first_logit_difference':result['first_logit_difference'],
                        'first_argmax_difference':result['first_argmax_difference'],
                        'verification_errors':result['verification_errors'],
                        'near_top2_shared_prefix_columns':len(result['near_top2_shared_prefix']),
                        'audited_target_columns':sum(len(r['columns']) for r in result['acceptance']),
                        'valid_width_sources':sorted({r['valid_width_source'] for r in result['acceptance']})}
            print(case,'identity',result['token_identity']['equal'],'verification errors',len(result['verification_errors']),flush=True)
            (a.folder/'summary.json').write_text(json.dumps(list(done.values()),indent=2),encoding='utf-8')
        if not a.watch or len(manifest)==30:break
        time.sleep(5)
    if any(r['verification_errors'] for r in done.values()):raise SystemExit(1)


if __name__=='__main__':main()
