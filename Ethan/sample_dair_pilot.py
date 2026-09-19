"""Create a reproducible label-stratified pilot from official dair splits."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def main():
    p=argparse.ArgumentParser();p.add_argument("--csv",required=True);p.add_argument("--output",required=True)
    p.add_argument("--train",type=int,default=800);p.add_argument("--dev",type=int,default=100);p.add_argument("--test",type=int,default=100)
    p.add_argument("--seed",type=int,default=42);a=p.parse_args()
    frame=pd.read_csv(a.csv)
    parts=[]
    for split,count in [("train",a.train),("dev",a.dev),("test",a.test)]:
        group=frame.loc[frame.split.eq(split)]
        fractions=group.label.value_counts(normalize=True).sort_index()
        counts=(fractions*count).astype(int)
        for label in (fractions*count-counts).sort_values(ascending=False).index[:count-counts.sum()]:counts.loc[label]+=1
        for label,n in counts.items():parts.append(group.loc[group.label.eq(label)].sample(n=int(n),random_state=a.seed))
    pilot=pd.concat(parts,ignore_index=True)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);pilot.to_csv(out,index=False)
    print(pilot.groupby(["split","emotion"]).size().to_string())
if __name__=="__main__":main()
