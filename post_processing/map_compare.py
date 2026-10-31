#!/usr/bin/env python3
"""
map_compare.py — compare feature vectors for a fixed conversation map, on every Snapper session.

Nodes are exchanges: a question and the reply that answered it. Stages, each cached in --dir and
safe to rerun:

  corpus    gather exchanges, dropping speaker bleed, repeats and reset requests
  minilm    all-MiniLM-L6-v2 (local ONNX cache): exchange text, question alone, reply alone
  gemma     Gemma 4 hidden states at --layer for each turn alone; speakers centred separately
  summary   Gemma (through local Ollama) names each exchange's topic; MiniLM embeds that
  report    scores and a contact sheet

usage:
  python3 map_compare.py corpus minilm [gemma summary] [report] --model MODEL_DIR [--dir outputs/map]
"""
import os
import re
import sys
import glob
import json
import time
import difflib
import argparse
import numpy as np

BY="/Volumes/Cohab2024/BFF/by-phase"
RESET=re.compile(r"(start|starting) over|let'?s reset",re.I)

def corpus(d):
    out=[]; seen=set(); kept=0; drop=dict(bleed=0,reset=0,short=0,repeat=0)
    dirs=sorted(x for x in glob.glob(BY+"/*/*") if x.rstrip("/").endswith("__snapper"))
    for sd in dirs:
        t=os.path.join(sd,"transcript"); p=os.path.join(t,"session-complete.jsonl")
        if not os.path.exists(p): p=os.path.join(t,"session.jsonl")
        if not os.path.exists(p): continue
        rows=[]; prompt=""; model=""
        for l in open(p):
            try: r=json.loads(l)
            except ValueError: continue
            if r.get("type")=="session_start":
                c=r.get("config") or {}; prompt=c.get("system_prompt") or ""; model=c.get("ollama_model") or ""
            elif r.get("type") in ("user","assistant") and (r.get("text") or "").strip(): rows.append(r)
        if prompt.startswith("You are Helper"): continue
        sid=[s for s in os.path.basename(sd).split("__") if s.startswith("session-")][0]; ph=os.path.basename(os.path.dirname(sd))
        last=""
        for a,b in zip(rows,rows[1:]):
            if b["type"]=="assistant": la=b["text"].strip()
            if a["type"]!="user" or b["type"]!="assistant":
                if a["type"]=="assistant": last=a["text"].strip()
                continue
            q,r=a["text"].strip(),b["text"].strip()
            if RESET.search(q): drop["reset"]+=1; continue
            if len(q.split())<2 or len(r.split())<3: drop["short"]+=1; continue
            if last and difflib.SequenceMatcher(None,q.lower(),last.lower()).ratio()>0.6: drop["bleed"]+=1; continue
            k=(q.lower(),r.lower())
            if k in seen: drop["repeat"]+=1; continue
            seen.add(k); out.append(dict(session=sid,phase=ph,model=model,t=a.get("timestamp"),q=q,r=r))
    json.dump(out,open(os.path.join(d,"corpus.json"),"w"),indent=0)
    print(f"corpus: {len(out)} exchanges from {len({x['session'] for x in out})} sessions; dropped {drop}")
    c={}
    for x in out: c[x["phase"]]=c.get(x["phase"],0)+1
    print(c)

class MiniLM:
    def __init__(s):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        b=glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/*"))[0]
        s.t=Tokenizer.from_file(b+"/tokenizer.json"); s.t.enable_truncation(256); s.t.enable_padding()
        s.m=ort.InferenceSession(b+"/onnx/model_qint8_arm64.onnx",providers=["CPUExecutionProvider"]); s.n=[i.name for i in s.m.get_inputs()]
    def __call__(s,texts,bs=32):
        out=[]
        for i in range(0,len(texts),bs):
            e=s.t.encode_batch(texts[i:i+bs]); ids=np.array([x.ids for x in e],np.int64); m=np.array([x.attention_mask for x in e],np.int64)
            f=dict(input_ids=ids,attention_mask=m,token_type_ids=np.zeros_like(ids)); h=s.m.run(None,{k:f[k] for k in s.n})[0]
            v=(h*m[...,None]).sum(1)/m.sum(1,keepdims=True); out.append(v)
        V=np.vstack(out); return V/np.linalg.norm(V,axis=1,keepdims=True)

def minilm(d):
    C=json.load(open(os.path.join(d,"corpus.json"))); M=MiniLM()
    np.savez_compressed(os.path.join(d,"minilm.npz"),x=M([c["q"]+" "+c["r"] for c in C]),q=M([c["q"] for c in C]),r=M([c["r"] for c in C]))
    print("minilm done")

def gemma(d,model,layer):
    import torch
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
    C=json.load(open(os.path.join(d,"corpus.json"))); p=os.path.join(d,"gemma.partial.npz")
    texts=sorted({c["q"] for c in C}|{c["r"] for c in C}); got={}
    if os.path.exists(p):
        z=np.load(p,allow_pickle=True); got=dict(zip(z["k"].tolist(),z["v"]))
    dev="mps" if torch.backends.mps.is_available() else "cpu"; tok=AutoTokenizer.from_pretrained(model)
    net=Gemma4ForConditionalGeneration.from_pretrained(model,dtype=torch.bfloat16).to(dev).eval(); t0=time.time()
    with torch.inference_mode():
        for n,t in enumerate(x for x in texts if x not in got):
            ids=[tok.bos_token_id]+tok(t,add_special_tokens=False)["input_ids"]
            h=net(input_ids=torch.tensor([ids],device=dev),output_hidden_states=True).hidden_states[layer]
            got[t]=h[0,1:].float().mean(0).cpu().numpy()
            if (n+1)%200==0:
                np.savez(p,k=np.array(list(got),dtype=object),v=np.stack(list(got.values()))); print(f"gemma {len(got)}/{len(texts)} {time.time()-t0:.0f}s",flush=True)
    Q=np.stack([got[c["q"]] for c in C]); R=np.stack([got[c["r"]] for c in C]); np.savez_compressed(os.path.join(d,"gemma.npz"),q=Q,r=R); print("gemma done",flush=True)

PROMPT=("In a short phrase of at most six plain words, name the topic of this exchange. "
        "Give only the topic, with no speakers and no quotation marks.\n\nQuestion: {q}\nReply: {r}")
def summary(d):
    import urllib.request
    C=json.load(open(os.path.join(d,"corpus.json"))); p=os.path.join(d,"summaries.json")
    S=json.load(open(p)) if os.path.exists(p) else {}; t0=time.time(); n=0
    for i,c in enumerate(C):
        if str(i) in S: continue
        body=dict(model="gemma4:e2b",stream=False,think=False,messages=[dict(role="user",content=PROMPT.format(q=c["q"],r=c["r"]))],options=dict(temperature=0,num_predict=24,seed=7))
        req=urllib.request.Request("http://localhost:11434/api/chat",json.dumps(body).encode(),{"Content-Type":"application/json"})
        S[str(i)]=json.load(urllib.request.urlopen(req,timeout=300))["message"]["content"].strip().strip('."').lower(); n+=1
        if n%50==0: json.dump(S,open(p,"w")); print(f"summary {len(S)}/{len(C)} {time.time()-t0:.0f}s",flush=True)
    json.dump(S,open(p,"w")); M=MiniLM(); np.savez_compressed(os.path.join(d,"summary.npz"),x=M([S[str(i)] for i in range(len(C))])); print("summary done",flush=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stages",nargs="+"); ap.add_argument("--model"); ap.add_argument("--dir",default="outputs/map"); ap.add_argument("--layer",type=int,default=18)
    a=ap.parse_args(); os.makedirs(a.dir,exist_ok=True)
    for s in a.stages:
        if s=="corpus": corpus(a.dir)
        elif s=="minilm": minilm(a.dir)
        elif s=="gemma": gemma(a.dir,a.model,a.layer)
        elif s=="summary": summary(a.dir)
        elif s=="report":
            import map_report; map_report.report(a.dir)
if __name__=="__main__": main()
