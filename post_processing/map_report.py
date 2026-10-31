#!/usr/bin/env python3
"""
map_report.py — scores and a contact sheet for the feature vectors map_compare.py builds.

Scores for each candidate:
  topic        of an exchange's 10 nearest, the share sharing its keyword group (questions only define
               the groups, and keywords are checked in the reply-independent question text)
  continuity   chance that the next exchange in a session is nearer than a random one (0.5 = no flow)
  own reply    (turn-level candidates) median rank of a question's own reply among all replies; lower is better
  speaker      (turn-level) silhouette of questions against replies; near 0 means style is not the main axis
  phase        silhouette by project phase; near 0 means the map is not split by era or model version
  stability    hold out 10% of sessions, fit a map on the rest, place the held-out by nearest neighbours,
               and compare to the full map: median displacement as a share of map width
Maps here use t-SNE (sklearn) as a stand-in for UMAP.
"""
import os
import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, pairwise_distances

GROUPS={"mccarthy / ai history":r"mccarthy|lovelace|turing|ada ","food / apple":r"apple|honeycrisp|taste|eat|food|hungry",
        "movement / body":r"velocity|moving|walk|legs|joint|motor|standing","memory":r"memor|remember","feeling":r"feel|feeling|emotion|sad|happy|dead inside"}

def unit(X): X=np.asarray(X,float); return X/np.clip(np.linalg.norm(X,axis=1,keepdims=True),1e-9,None)
def zs(X): s=X.std(0); return (X-X.mean(0))/np.maximum(s,1e-3*np.median(s))

def embed2(X,seed=7):
    n=min(50,X.shape[1],X.shape[0]-1)
    return TSNE(2,perplexity=30,init=PCA(2,random_state=seed).fit_transform(PCA(n,random_state=seed).fit_transform(X)),metric="cosine",random_state=seed).fit_transform(X)

def place(Xtrain,Ytrain,Xnew,k=10):
    D=pairwise_distances(unit(Xnew),unit(Xtrain),metric="cosine"); idx=np.argsort(D,1)[:,:k]
    W=1/(np.take_along_axis(D,idx,1)+1e-3); return (Ytrain[idx]*W[...,None]).sum(1)/W.sum(1,keepdims=True)

def scores(X,C,labels):
    import re
    N=len(C); U=unit(X); S=U@U.T; np.fill_diagonal(S,-2); nn=np.argsort(-S,1)[:,:10]; out={}
    tops=[(g,[i for i,c in enumerate(C) if re.search(p,c["q"].lower())]) for g,p in GROUPS.items()]
    out["topic"]={}
    for g,ids in tops:
        m=np.zeros(N,bool); m[ids]=True; out["topic"][g]=(round(float(m[nn[ids]].mean()),3),len(ids),round(float(m.mean()),3))
    rng=np.random.default_rng(7); w=[]
    for i in range(N-1):
        if C[i]["session"]==C[i+1]["session"]:
            for j in rng.integers(0,N,5): w.append(S[i,i+1]>S[i,j])
    out["continuity"]=round(float(np.mean(w)),3)
    out["phase"]=round(float(silhouette_score(U,labels,metric="cosine")),3)
    Y=embed2(X); sid=sorted({c["session"] for c in C}); rng=np.random.default_rng(7); held=set(rng.choice(sid,len(sid)//10,replace=False).tolist())
    te=np.array([c["session"] in held for c in C]); Yt=embed2(X[~te]); from scipy.linalg import orthogonal_procrustes as op
    a=Yt-Yt.mean(0); b=Y[~te]-Y[~te].mean(0); R,_=op(a,b); Yta=a@R*(np.linalg.norm(b)/np.linalg.norm(a@R))+Y[~te].mean(0)
    P=place(X[~te],Yta,X[te]); out["stability"]=round(float(np.median(np.linalg.norm(P-Y[te],axis=1))/np.ptp(Y[:,0])),3)
    return out,Y

def turn_scores(Q,R):
    u=unit(Q); v=unit(R); S=u@v.T; rank=(S>np.diag(S)[:,None]).sum(1)
    A=np.vstack([u,v]); lab=np.r_[np.zeros(len(u)),np.ones(len(v))]; rng=np.random.default_rng(7); i=rng.choice(len(A),min(2000,len(A)),replace=False)
    return int(np.median(rank)),round(float(silhouette_score(A[i],lab[i],metric="cosine")),3)

def report(d):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, re
    C=json.load(open(os.path.join(d,"corpus.json"))); M=np.load(os.path.join(d,"minilm.npz")); G=np.load(os.path.join(d,"gemma.npz")); T=np.load(os.path.join(d,"summary.npz"))
    gq,gr=zs(G["q"]),zs(G["r"])                        # each speaker centred on its own, which removes the register axis
    gxraw=zs(np.vstack([G["q"],G["r"]])); gq0,gr0=gxraw[:len(C)],gxraw[len(C):]
    cand={"Gemma turns, speakers centred":(gq+gr,(gq,gr)),"Gemma turns, not centred":(gq0+gr0,(gq0,gr0)),"MiniLM on exchange text":(M["x"],(M["q"],M["r"])),"Gemma topic phrase + MiniLM":(T["x"],None)}
    ph=sorted({c["phase"] for c in C}); lab=np.array([ph.index(c["phase"]) for c in C]); R={}; Ys={}
    for k,(X,tu) in cand.items():
        r,Y=scores(X,C,lab); Ys[k]=Y
        if tu: r["own_reply_rank"],r["speaker"]=turn_scores(*tu)
        R[k]=r; print(k,json.dumps(r),flush=True)
    json.dump(R,open(os.path.join(d,"scores.json"),"w"),indent=1)
    sess=max({c["session"] for c in C},key=lambda s:0); target="session-20260722-114251"
    fig,ax=plt.subplots(2,2,figsize=(15,11),dpi=120); col=dict(zip(list(GROUPS)[:3],["#2a78d6","#eb6834","#1baf7a"]))
    for a,(k,Y) in zip(ax.flat,Ys.items()):
        a.scatter(Y[:,0],Y[:,1],s=5,c="#c3c2b7",lw=0)
        for g,c_ in col.items():
            m=np.array([bool(re.search(GROUPS[g],x["q"].lower())) for x in C]); a.scatter(Y[m,0],Y[m,1],s=11,c=c_,lw=0,label=g)
        t=[i for i,x in enumerate(C) if x["session"]==target]; a.plot(Y[t,0],Y[t,1],color="#0b0b0b",lw=1,marker="o",ms=2.5,label="McCarthy session, in order")
        a.set_title(k,loc="left",fontsize=12); a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values(): s.set_color("#e1e0d9")
    ax[0,0].legend(loc="lower left",fontsize=8,frameon=False); fig.tight_layout(); fig.savefig(os.path.join(d,"contact-sheet.png")); print("sheet saved")
