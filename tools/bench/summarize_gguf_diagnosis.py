"""Flatten controlled B/B processes and attribute captured ordinary rounds."""
import argparse
import csv
import json
import sqlite3
import statistics
from pathlib import Path


def bb(folder,out):
    rows=[]
    for path in sorted(folder.glob('*.process.json')):
        p=json.loads(path.read_text()); entry=json.loads(path.with_name(path.name.replace('.process','.entry')).read_text())
        for label,metrics in p['tests'].items():
            row={k:p[k] for k in ['mode','pair','arm','elapsed_seconds','gpu_clock_mhz_median',
                 'temperature_c_median','power_w_median','power_w_peak',
                 'total_vram_mib_peak','process_dedicated_vram_mib_peak']}
            row.update(entry_temperature_c=entry['temperature.gpu'],memory_clock_mhz=entry['clocks.mem'],
                       clock_min=p['gpu_clock_mhz_range'][0],clock_max=p['gpu_clock_mhz_range'][1],
                       temperature_min=p['temperature_c_range'][0],temperature_max=p['temperature_c_range'][1],
                       label=label,**metrics)
            rows.append(row)
    with out.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def nsys(folder):
    results=[]
    for path in sorted(folder.glob('*.sqlite')):
        db=sqlite3.connect(path)
        ranges=db.execute('SELECT n.start,n.end FROM NVTX_EVENTS n LEFT JOIN StringIds s '
                          'ON n.textId=s.id WHERE coalesce(n.text,s.value)=?',('decode.ordinary_round',)).fetchall()
        rounds=[]
        for start,end in ranges:
            kernels=db.execute('SELECT start,end FROM CUPTI_ACTIVITY_KIND_KERNEL '
                               'WHERE start>=? AND end<=? ORDER BY start',(start,end)).fetchall()
            if not kernels:continue
            active=0; last=kernels[0][0]
            for a,b in kernels:
                active+=max(0,b-max(a,last));last=max(last,b)
            rounds.append({'wall_ms':(end-start)/1e6,'kernel_ms':active/1e6,
                           'pre_queue_ms':(kernels[0][0]-start)/1e6,
                           'gap_ms':(last-kernels[0][0]-active)/1e6,'post_ms':(end-last)/1e6})
        results.append({'capture':path.stem,'means':{k:statistics.mean(r[k] for r in rounds)
                        for k in rounds[0]},'rounds':rounds})
        db.close()
    (folder/'analysis.json').write_text(json.dumps(results,indent=2),encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bb',type=Path,required=True);p.add_argument('--csv',type=Path,required=True)
    p.add_argument('--nsys',type=Path,required=True);a=p.parse_args();bb(a.bb,a.csv);nsys(a.nsys)


if __name__=='__main__':main()
