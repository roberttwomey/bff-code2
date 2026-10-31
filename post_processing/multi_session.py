#!/usr/bin/env python3
"""
multi_session.py — several successive conversations in one shared semantic space.

Replays each session through Gemma 4, then reduces all their utterances together with MDS, so
positions can be compared across recordings. --space picks what places them:

  activation  (default) the hidden state in context at --layer; utterances that never entered the
              model have no position and go in a stack
  standalone  each utterance alone through the model, with no prompt or history, at --layer
  tfidf       word overlap across all the sessions' text (TF-IDF -> LSA -> MDS); no model

Writes the same two files graph3d.py
and attention_flow.py write for one session, with turns renumbered (A12, B3), so
plotter_drawing.py draws the result without further options. Also writes the viewer page.

usage:
  python3 multi_session.py SESSION_DIR [SESSION_DIR ...] --model MODEL_DIR [--space activation]
          --system-prompt performance-script.json-02 [--names "dead inside" "apple" ...]
          [--layer 18] [--flat] [--cache-dir outputs/multi] [-o outputs/snapper-multi.json]

Sessions are ordered as given. All are replayed with one system prompt, so differences between
them are not differences of prompt; scene switches inside a session are not reproduced.
Writes OUT.json, OUT-attention.json and OUT.html. Per-session replays are cached in --cache-dir.
"""
import os
import json
import argparse
import numpy as np
import conversation_graph as cg
import activation_graph as ag
import graph3d as g3

HUM,DOG,SYS,RST=cg.HUM,cg.DOG,cg.SYS,cg.RST

def replay(d,model,spec,cache_dir):
    """One session's replay, cached: activations and attention, built with the given system prompt."""
    cfg,U,resets,_=cg.load(d); sp,src=ag.resolve_system_prompt(spec,cfg)
    ex,_=ag.build_replay(d,cfg,U,system_prompt=sp)
    sid=[s for s in os.path.basename(d.rstrip("/")).split("__") if s.startswith("session-")][0]
    pa,pj=os.path.join(cache_dir,sid+".acts.npz"),os.path.join(cache_dir,sid+".attn.json")
    z=np.load(pa) if os.path.exists(pa) else None
    if z is None or str(z["system_prompt"])!=sp:
        print(f"replaying {sid}: activations, {len(ex)} exchanges",flush=True)
        acts,_=ag.extract(ex,model,None)
        np.savez_compressed(pa,user=acts["user"],asst=acts["asst"],user_i=[x["user_i"] for x in ex],
                            asst_i=[x["asst_i"] for x in ex],system_prompt=sp); z=np.load(pa)
    r=json.load(open(pj)) if os.path.exists(pj) else None
    if r is None or r.get("system_prompt")!=sp:
        print(f"replaying {sid}: attention and surprise",flush=True)
        r=ag.extract_attention_surprise(ex,model); r.update(system_prompt=sp,system_prompt_source=src); json.dump(r,open(pj,"w"))
    return dict(sid=sid,cfg=cfg,U=U,resets=set(resets),z=z,r=r)

def main():
    ap=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions",nargs="+"); ap.add_argument("--model",required=True)
    ap.add_argument("--system-prompt",default=None,metavar="SCRIPT_OR_TXT")
    ap.add_argument("--names",nargs="*",default=[],help="a short name for each session, for the key")
    ap.add_argument("--space",choices=("activation","standalone","tfidf"),default="activation")
    ap.add_argument("--flat",action="store_true",help="MDS to 2D rather than 3D, drawn straight on")
    ap.add_argument("--layer",type=int,default=18); ap.add_argument("--cache-dir",default="outputs/multi")
    ap.add_argument("-o","--out",default="outputs/snapper-multi.json")
    a=ap.parse_args(); os.makedirs(a.cache_dir,exist_ok=True)
    S=[replay(d,a.model,a.system_prompt,a.cache_dir) for d in a.sessions]
    nodes=[]; vecs={}; links=[]; ex_all=[]; info=[]; off=0; prev=None
    for k,s in enumerate(S):
        L="ABCDEFGHIJKL"[k]; U=s["U"]; z=s["z"]; rs=s["resets"]
        if a.space=="activation":
            for n,i in enumerate(z["user_i"]): vecs[int(i)+off]=z["user"][n,a.layer]
            for n,i in enumerate(z["asst_i"]): vecs[int(i)+off]=z["asst"][n,a.layer]
        elif a.space=="standalone":
            pc=os.path.join(a.cache_dir,s["sid"]+".standalone.npz")
            if not os.path.exists(pc):
                print(f'replaying {s["sid"]}: each utterance alone',flush=True); np.savez_compressed(pc,H=ag.extract_standalone(U,a.model))
            H=np.load(pc)["H"]
            for u in U: vecs[u["i"]+off]=H[u["i"],a.layer]
        else:
            for u in U: vecs[u["i"]+off]=None
        for u in U:
            g=u["i"]+off; live=g in vecs; who="ROBERT" if u["speaker"]=="HUMAN" else ("SNAPPER" if u["type"]=="assistant" else "SYSTEM")
            nodes.append(dict(i=g,label=f'{L}{u["i"]}',session=k,who=who,type=u["type"],text=(u["logged"] or u["text"]) if live and a.space!="tfidf" else u["text"],
                              dur=u["dur"],parked=not live,c=HUM if who=="ROBERT" else (DOG if who=="SNAPPER" else SYS)))
            if u["i"]>0:                           # recordings stay separate: no link from one to the next
                p=u["i"]-1; links.append([prev,g,RST if p in rs or u["i"] in rs else "#9a948c",True])
            prev=g
        for x in s["r"]["exchanges"]:
            y=dict(x,user_i=x["user_i"]+off,asst_i=x["asst_i"]+off,history=[dict(h,i=(h["i"]+off if h["i"] is not None else None)) for h in x["history"]])
            ex_all.append(y)
        info.append(dict(letter=L,session=s["sid"],name=a.names[k] if k<len(a.names) else "",start=U[0]["t"].isoformat(),end=U[-1]["t"].isoformat(),
                         utterances=len(U),in_model=int(len(z["user_i"])+len(z["asst_i"]))))
        off+=len(U)
    live=sorted(vecs); lp={i:k for k,i in enumerate(live)}; by={n["i"]:n for n in nodes}
    if a.space=="tfidf":
        _,sim,_=cg.embed([dict(by[i],i=k,speaker="HUMAN" if by[i]["who"]=="ROBERT" else by[i]["who"],block=None) for k,i in enumerate(live)],1.01)
        D=np.clip(1-sim,0,None); D=(D+D.T)/2
    else:
        X=np.stack([vecs[i] for i in live]); D=ag.cosdist(ag.std_frame(X)(X)); D=(D+D.T)/2
    np.fill_diagonal(D,0)
    for l in links: l[3]=l[0] not in lp or l[1] not in lp
    hum=[lp[i] for i in live if by[i]["who"]=="ROBERT"]; b1=[lp[i] for i in live if by[i]["session"]==0]
    P=g3.layout3d(D,hum,b1,2 if a.flat else 3)
    for n in nodes: n["p"]=[round(float(v),5) for v in P[lp[n["i"]]]] if n["i"] in lp else None
    from scipy.stats import spearmanr
    from scipy.spatial.distance import pdist
    from sklearn.metrics import silhouette_score
    iu=np.triu_indices(len(live),1); fid=float(spearmanr(pdist(P),D[iu]).statistic)
    who=np.array([by[i]["who"]=="ROBERT" for i in live]); ses=np.array([by[i]["session"] for i in live])
    sw,ss=float(silhouette_score(D,who,metric="precomputed")),float(silhouette_score(D,ses,metric="precomputed"))
    data=dict(session="+".join(s["sid"] for s in S),space=a.space if a.space=="tfidf" else f"{a.space} layer {a.layer}",start=info[0]["start"],end=info[-1]["end"],sessions=info,nodes=nodes,links=links,
              floor=[round(v,4) for v in g3.floor_of(P)],view=[0,0,0] if a.flat else [30,20,4],flat=a.flat,echoes=[],shared=dict(rho_3d=round(fid,3),separation_by_speaker=round(sw,3),separation_by_session=round(ss,3)))
    json.dump(data,open(a.out,"w"),indent=1); stem=os.path.splitext(a.out)[0]
    json.dump(dict(parts=S[0]["r"]["parts"],layers=S[0]["r"]["layers"],exchanges=ex_all,system_prompt=S[0]["r"]["system_prompt"],
                   system_prompt_source=S[0]["r"]["system_prompt_source"]),open(stem+"-attention.json","w"))
    title=" + ".join(s["sid"][8:] for s in S)
    open(stem+".html","w").write(g3.VIEWER.replace("__TITLE__",cg.e(title)).replace("__DATA__",json.dumps(data).replace("</","<\\/")))
    print(f"{a.out}  ({len(nodes)} utterances, {len(live)} in the space; rho {fid:.2f} in {"2D" if a.flat else "3D"}; separation by speaker {sw:.2f}, by session {ss:.2f})")
    for s in info: print(f'  {s["letter"]} {s["session"]}  {s["in_model"]}/{s["utterances"]} in the model  {s["name"]}')

if __name__=="__main__": main()
