"""Convert the Hugging Face dair-ai/emotion split to the project schema."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

LABELS = {0:"sadness",1:"joy",2:"love",3:"anger",4:"fear",5:"surprise"}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--source-dir",default="data/dair_emotion/source"); p.add_argument("--output-dir",default="data/dair_emotion"); a=p.parse_args()
    source, output=Path(a.source_dir),Path(a.output_dir); output.mkdir(parents=True,exist_ok=True)
    parts=[]
    for filename,split in [("train.parquet","train"),("validation.parquet","dev"),("test.parquet","test")]:
        frame=pd.read_parquet(source/filename)
        if not {"text","label"}.issubset(frame.columns): raise ValueError(f"Unexpected fields in {filename}")
        frame=frame[["text","label"]].copy(); frame["split"]=split; parts.append(frame)
    data=pd.concat(parts,ignore_index=True)
    if data.isna().any().any(): raise ValueError("Unexpected missing values")
    if not data["label"].isin(LABELS).all(): raise ValueError("Unexpected label")
    data.insert(0,"id",[f"dair_{i:05d}" for i in range(len(data))])
    data["emotion"]=data["label"].map(LABELS); data["source"]="dair-ai/emotion"
    data.to_csv(output/"dair_emotion.csv",index=False)
    for split in ["train","dev","test"]: data.loc[data.split.eq(split)].to_csv(output/f"dair_emotion_{split}.csv",index=False)
    meta={"rows":len(data),"split_counts":data.split.value_counts().to_dict(),"labels":LABELS,"source":"https://huggingface.co/datasets/dair-ai/emotion","purpose":"external categorical-emotion validation"}
    (output/"dataset_metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8"); print(json.dumps(meta,indent=2))
if __name__=="__main__": main()
