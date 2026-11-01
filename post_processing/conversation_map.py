#!/usr/bin/env python3
"""
conversation_map.py — one fixed map of everything Snapper has talked about, and conversations on it.

  fit    fits UMAP on every exchange (a question with its reply, MiniLM vectors) from
         map_compare.py, names regions with Gemma's topic phrases, and saves the map
  place  puts each turn of one or more sessions on that map separately, so a conversation steps between
         question and reply. The map does not change, so every plot shares its coordinates

Only the canned greeting, reset cues and reset requests go in the stack; everything else
someone said is on the map, answered or not. Sessions in the older chat_session format,
which has no turn list, are rebuilt from the message histories sent to the model: they have
no audio, so their circles are one size and their times are request times. The output is a graph JSON for plotter_drawing.py.

Runs in the bff conda env (umap-learn, onnxruntime, tokenizers):
  /opt/homebrew/anaconda3/envs/bff/bin/python post_processing/conversation_map.py fit
  /opt/homebrew/anaconda3/envs/bff/bin/python post_processing/conversation_map.py place SESSION_DIR [SESSION_DIR ...] [--names "a" "b"] -o outputs/x-map.json
"""
import os
import re
import glob
import json
import pickle
import argparse
import numpy as np
import conversation_graph as cg
from map_compare import MiniLM, RESET

def load_legacy(session_dir):
    """Rebuild a chat_session log. Each request carries the history so far, so segments between
    resets are merged by overlap, and a segment's last reply comes from its response record."""
    p=glob.glob(os.path.join(session_dir,"transcript","chat_session*.jsonl"))[0]; R=[json.loads(l) for l in open(p) if l.strip()]
    segs=[]; seg=[]; last=None
    def close():
        nonlocal seg,last
        if last and not any(m[0]=="assistant" and m[1]==last[0] for m in seg[-3:]): seg.append(("assistant",last[0],last[1]))
        if seg: segs.append(seg)
        seg=[]; last=None
    for r in R:
        k=r.get("type")
        if k=="reset_triggered": close()
        elif k=="chat_stream_response": last=((r.get("reply") or "").strip(),r["timestamp"])
        elif k=="chat_stream_request":
            m=[(x["role"],x["content"].strip(),r["timestamp"]) for x in r["messages"] if x["role"]!="system" and x["content"].strip()]
            kk=next((n for n in range(min(len(m),len(seg)),0,-1) if [q[:2] for q in m[:n]]==[q[:2] for q in seg[-n:]]),0)
            seg+=m[kk:]
    close(); U=[]; resets=[]
    for g in segs:
        for role,text,t in g:
            i=len(U); u=role=="user"
            U.append(dict(t=cg.parse_ts(t),type="user" if u else "assistant",speaker="HUMAN" if u else "SNAPPER",text=text,logged="",dur=None,block=None,cue=None,i=i))
            if u and RESET.search(text): resets.append(i)
    return {},U,resets,0

def load_any(session_dir):
    if glob.glob(os.path.join(session_dir,"transcript","chat_session*.jsonl")): return load_legacy(session_dir)
    return cg.load(session_dir)

def fit(d,regions):
    import umap
    from sklearn.cluster import KMeans
    C=json.load(open(os.path.join(d,"corpus.json"))); X=np.load(os.path.join(d,"minilm.npz"))["x"]
    S=json.load(open(os.path.join(d,"summaries.json"))); T=np.load(os.path.join(d,"summary.npz"))["x"]
    m=umap.UMAP(n_components=2,n_neighbors=30,min_dist=0.3,metric="cosine",random_state=7).fit(X); Y=m.embedding_
    km=KMeans(regions,n_init=10,random_state=7).fit(Y); names=[]
    for k in range(regions):
        ids=np.where(km.labels_==k)[0]; c=T[ids].mean(0); best=ids[np.argmax(T[ids]@c)]
        names.append(dict(p=[round(float(v),3) for v in Y[ids].mean(0)],name=S[str(best)],n=len(ids)))
    pickle.dump(m,open(os.path.join(d,"map.pkl"),"wb"))
    pad=(Y.max(0)-Y.min(0))*0.04
    json.dump(dict(backdrop=np.round(Y,3).tolist(),regions=names,extent=[*(Y.min(0)-pad).tolist(),*(Y.max(0)+pad).tolist()]),open(os.path.join(d,"map.json"),"w"))
    print(f"map fitted on {len(C)} exchanges"); [print(f'  {r["n"]:>4}  {r["name"]}') for r in names]

def place(d,sessions,out,names,by_day=False):
    M=json.load(open(os.path.join(d,"map.json"))); m=pickle.load(open(os.path.join(d,"map.pkl"),"rb")); E=MiniLM()
    w=M["extent"][2]-M["extent"][0]; nodes=[]; links=[]; info=[]; off=0; sids=[]; groups={}
    for sd in sessions:
        groups.setdefault(re.search(r"(\d{8})",os.path.basename(sd.rstrip("/")))[1] if by_day else sd,[]).append(sd)
    groups=list(groups.values()); multi=len(groups)>1
    for g,dirs in enumerate(groups):
        L="ABCDEFGHIJKLMNOPQRSTUVWXYZ"[g]; base=off; P=[]; st=[]; first=last=None; total=0
        for r,session_dir in enumerate(dirs):
            cfg,U,resets,_=load_any(session_dir); reset=set(resets)
            keep=[u["i"] for u in U if u["type"]=="assistant" or (u["type"]=="user" and not RESET.search(u["text"]))]
            pos=dict(zip(keep,m.transform(E([U[i]["text"] for i in keep])))) if keep else {}
            for u in U:
                i=u["i"]; who="ROBERT" if u["speaker"]=="HUMAN" else ("SNAPPER" if u["type"]=="assistant" else "SYSTEM")
                n=dict(i=i+off,who=who,type=u["type"],text=u["text"],dur=u["dur"],parked=i not in pos,c="#000000",
                       p=[round(float(pos[i][0]),4),round(float(pos[i][1]),4),0.0] if i in pos else None)
                if multi: n.update(label=f"{L}{i+off-base}",session=g,recording=r)
                nodes.append(n)
                if i: links.append([i-1+off,i+off,cg.RST if i-1 in reset or i in reset else "#9a948c",i-1 not in pos or i not in pos])
            sid=[x for x in os.path.basename(session_dir.rstrip("/")).split("__") if "session" in x][0]; sids.append(sid)
            P+=[pos[i] for i in keep]; first=first or U[0]["t"]; last=U[-1]["t"]; total+=len(U); off+=len(U)
            st+=[np.linalg.norm(pos[a]-pos[b])/w for a,b in zip(keep,keep[1:]) if U[a]["type"]=="user" and U[b]["type"]=="assistant"]
        info.append(dict(letter=L,session=sid,name=names[g] if g<len(names) else "",start=first.isoformat(),end=last.isoformat(),utterances=total))
        P=np.array(P); print(f"{L} {len(dirs)} recording(s)  {len(P)} of {total} turns on the map; centre {P.mean(0).round(1)}; spread {100*np.ptp(P,axis=0).max()/w:.0f}% of map width; median step {100*np.median(st):.0f}%")
    data=dict(session="+".join(sids),space="fixed exchange map",start=info[0]["start"],end=info[-1]["end"],nodes=nodes,links=links,view=[0,0,0],
              echoes=[],extent=M["extent"],backdrop=M["backdrop"],regions=M["regions"])
    if multi: data["sessions"]=info
    json.dump(data,open(out,"w"),indent=0); print(out)

if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command",choices=("fit","place")); ap.add_argument("sessions",nargs="*"); ap.add_argument("--names",nargs="*",default=[]); ap.add_argument("-o","--out",default="outputs/map/session.json")
    ap.add_argument("--by-day",action="store_true",help="group sessions from one date as a single letter, not joined to each other")
    ap.add_argument("--dir",default="outputs/map"); ap.add_argument("--regions",type=int,default=12)
    a=ap.parse_args()
    fit(a.dir,a.regions) if a.command=="fit" else place(a.dir,a.sessions,a.out,a.names,a.by_day)
