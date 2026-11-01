#!/usr/bin/env python3
"""
conversation_map.py — one fixed map of everything Snapper has talked about, and conversations on it.

  fit    fits UMAP on every exchange (a question with its reply, MiniLM vectors) from
         map_compare.py, names regions with Gemma's topic phrases, and saves the map
  place  puts each turn of one or more sessions on that map separately, so a conversation steps between
         question and reply. The map does not change, so every plot shares its coordinates

Turns that are not content (the greeting, reset requests and cues, unanswered lines) have no
place on the map and go in the stack. The output is a graph JSON for plotter_drawing.py.

Runs in the bff conda env (umap-learn, onnxruntime, tokenizers):
  /opt/homebrew/anaconda3/envs/bff/bin/python post_processing/conversation_map.py fit
  /opt/homebrew/anaconda3/envs/bff/bin/python post_processing/conversation_map.py place SESSION_DIR [SESSION_DIR ...] [--names "a" "b"] -o outputs/x-map.json
"""
import os
import re
import json
import pickle
import argparse
import numpy as np
import conversation_graph as cg
from map_compare import MiniLM, RESET

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

def place(d,sessions,out,names):
    M=json.load(open(os.path.join(d,"map.json"))); m=pickle.load(open(os.path.join(d,"map.pkl"),"rb")); E=MiniLM()
    w=M["extent"][2]-M["extent"][0]; nodes=[]; links=[]; info=[]; off=0; multi=len(sessions)>1; sids=[]
    for g,session_dir in enumerate(sessions):
        cfg,U,resets,_=cg.load(session_dir); reset=set(resets); keep=[]; L="ABCDEFGHIJKL"[g]
        for k,u in enumerate(U):
            nxt=U[k+1] if k+1<len(U) else None
            if u["type"]=="assistant" or (u["type"]=="user" and nxt is not None and nxt["type"]=="assistant" and not RESET.search(u["text"])): keep.append(u["i"])
        pos=dict(zip(keep,m.transform(E([U[i]["text"] for i in keep]))))
        for u in U:
            i=u["i"]; who="ROBERT" if u["speaker"]=="HUMAN" else ("SNAPPER" if u["type"]=="assistant" else "SYSTEM")
            n=dict(i=i+off,who=who,type=u["type"],text=u["text"],dur=u["dur"],parked=i not in pos,c="#000000",
                   p=[round(float(pos[i][0]),4),round(float(pos[i][1]),4),0.0] if i in pos else None)
            if multi: n.update(label=f"{L}{i}",session=g)
            nodes.append(n)
            if i: links.append([i-1+off,i+off,cg.RST if i-1 in reset or i in reset else "#9a948c",i-1 not in pos or i not in pos])
        sid=[x for x in os.path.basename(session_dir.rstrip("/")).split("__") if x.startswith("session-")][0]; sids.append(sid)
        info.append(dict(letter=L,session=sid,name=names[g] if g<len(names) else "",start=U[0]["t"].isoformat(),end=U[-1]["t"].isoformat(),utterances=len(U)))
        P=np.array([pos[i] for i in keep]); st=[np.linalg.norm(pos[a]-pos[b])/w for a,b in zip(keep,keep[1:]) if U[a]["type"]=="user" and U[b]["type"]=="assistant"]
        print(f"{L} {sid}  {len(keep)} of {len(U)} turns on the map; centre {P.mean(0).round(1)}; spread {100*np.ptp(P,axis=0).max()/w:.0f}% of map width; median step {100*np.median(st):.0f}%")
        for v,a,b in sorted(((np.linalg.norm(pos[a]-pos[b])/w,a,b) for a,b in zip(keep,keep[1:]) if U[a]["type"]=="user" and U[b]["type"]=="assistant"),reverse=True)[:2]:
            print(f"    largest {100*v:.0f}%  {a}->{b}  {U[a]['text'][:44]!r} -> {U[b]['text'][:44]!r}")
        off+=len(U)
    data=dict(session="+".join(sids),space="fixed exchange map",start=info[0]["start"],end=info[-1]["end"],nodes=nodes,links=links,view=[0,0,0],
              echoes=[],extent=M["extent"],backdrop=M["backdrop"],regions=M["regions"])
    if multi: data["sessions"]=info
    json.dump(data,open(out,"w"),indent=0); print(out)

if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command",choices=("fit","place")); ap.add_argument("sessions",nargs="*"); ap.add_argument("--names",nargs="*",default=[]); ap.add_argument("-o","--out",default="outputs/map/session.json")
    ap.add_argument("--dir",default="outputs/map"); ap.add_argument("--regions",type=int,default=12)
    a=ap.parse_args()
    fit(a.dir,a.regions) if a.command=="fit" else place(a.dir,a.sessions,a.out,a.names)
