#!/usr/bin/env python3
"""
conversation_graph.py — static node/edge portrait of one BFF dialogue session.

Renders an SVG in two panels:
  LEFT   semantic space — utterances placed by --space, linked by the dialogue spine (time)
         and by semantic echo arcs (recurrence).
  RIGHT  prompt payload — what was actually inside the context window at each generation,
         segmented into system / visual (VLM) / history / user tokens, with resets marked.

--space picks what "semantic" means:
  activation  (default) Gemma 4's hidden state for each utterance, in context, at --layer: how the
              model represented it at that point in the conversation. Utterances that never
              entered the model have no such vector and are parked in the upper left.
  standalone  each utterance alone through the model, with no prompt or history. All utterances
              get a vector, including those that never entered the model during the session.
  tfidf       word overlap (TF-IDF -> LSA -> MDS). Lexical, not neural; no model needed.

usage:
  python3 conversation_graph.py SESSION_DIR [--space activation] [--acts ACTS.npz] [--layer 18]
                                [--model MODEL_DIR] [--system-prompt SCRIPT_OR_TXT] [-o OUT.svg]
                                [--echo-threshold 0.86] [--png [--png-width 3440]]

activation reads the cache activation_graph.py writes (default OUT.acts.npz), building it if
--model is given. standalone caches OUT.standalone.npz and needs --model on its first run.

SESSION_DIR is a by-phase session folder containing transcript/session-complete.jsonl
(falls back to transcript/session.jsonl).
"""
import os
import sys
import json
import math
import html
import argparse
import shutil
import subprocess
import textwrap
import datetime as dt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import MDS

def e(s): return html.escape(s or "", quote=True)

def parse_ts(s):
    if not s: return None
    for f in ("%Y-%m-%dT%H:%M:%S.%f","%Y-%m-%dT%H:%M:%S","%Y%m%d-%H%M%S"):
        try: return dt.datetime.strptime(s.strip(), f)
        except ValueError: pass
    return None

def load(session_dir):
    for name in ("session-complete.jsonl","session.jsonl"):
        p=os.path.join(session_dir,"transcript",name)
        if os.path.exists(p): break
    else: sys.exit(f"no transcript jsonl under {session_dir}/transcript")
    rows=[json.loads(l) for l in open(p) if l.strip()]
    cfg=[r for r in rows if r.get("type")=="session_start"][0]["config"]
    vlm=sorted((parse_ts(r["timestamp"]), r["description"]) for r in rows
               if r.get("type")=="vlm_query" and parse_ts(r["timestamp"]))
    U=[]
    for r in rows:
        if r.get("type") in ("user","assistant","cue","startup"):
            t=parse_ts(r.get("timestamp"))
            if not t: continue
            U.append(dict(t=t, type=r["type"], speaker=r.get("speaker"),
                          text=(r.get("text") or "").strip(), logged=(r.get("logged_text") or "").strip(),
                          dur=r.get("duration_s"),
                          block=r.get("block"), cue=r.get("cue")))
    U.sort(key=lambda u:u["t"])
    for i,u in enumerate(U): u["i"]=i
    for u in U:
        prior=[v for v in vlm if v[0]<=u["t"]]
        u["vlm"]=prior[-1][1] if prior else None
    # reconstruct the payload each generation saw, mirroring chat-manager's rolling window
    toks=lambda s: max(1, round(len(s)/4)) if s else 0
    SYSP=cfg["system_prompt"]; LIMIT=cfg["history_truncation_limit"]
    hist=[]; resets=[]
    for u in U:
        if u["type"]=="cue" and u.get("cue")=="reset":
            resets.append(u["i"]); hist=[]; u["payload"]=None; continue
        if u["type"]=="user":
            hist.append(("user",u["text"])); u["payload"]=None
        elif u["type"]=="assistant":
            win=hist[-(LIMIT-1):] if len(hist)>LIMIT-1 else hist
            last_user=next((t for r_,t in reversed(win) if r_=="user"), "")
            hist_txt=" ".join(t for _,t in win[:-1]) if len(win)>1 else ""
            u["payload"]=dict(system=toks(SYSP), visual=toks(u["vlm"] or ""),
                              history=toks(hist_txt), user=toks(last_user))
            hist.append(("assistant",u["text"]))
        else: u["payload"]=None
    return cfg, U, resets, len(vlm)

def embed(U, echo_min):
    texts=[u["text"] for u in U]
    X=TfidfVectorizer(sublinear_tf=True, ngram_range=(1,2), min_df=1,
                      stop_words="english", norm="l2").fit_transform(texts)
    L=TruncatedSVD(n_components=8, random_state=7).fit_transform(X)
    L=L/np.clip(np.linalg.norm(L,axis=1,keepdims=True),1e-9,None)
    sim=np.clip(L@L.T,-1,1); np.fill_diagonal(sim,1.0)
    P=MDS(n_components=2, dissimilarity="precomputed", random_state=7, n_init=16,
          max_iter=1500, normalized_stress="auto").fit_transform(np.clip(1.0-sim,0,None))
    # orient: widest spread horizontal, block 1 on the left, the human side up
    P=P-P.mean(0); _,_,vt=np.linalg.svd(P, full_matrices=False); P=P@vt.T
    b1=[u["i"] for u in U if u.get("block")==1]
    hn=[u["i"] for u in U if u["speaker"]=="HUMAN"]
    if b1 and P[b1,0].mean()>0: P[:,0]*=-1
    if hn and P[hn,1].mean()<0: P[:,1]*=-1
    kind=lambda u: "H" if u["speaker"]=="HUMAN" else ("S" if u["type"]=="assistant" else "X")
    ed=[(i,j,float(sim[i,j]),kind(U[i])+kind(U[j]))
        for i in range(len(U)) for j in range(i+2,len(U)) if sim[i,j]>=echo_min]
    return P, sim, sorted(ed, key=lambda x:-x[2])

W,H=1720,1216
GX0,GX1,GY0,GY1=62,1118,132,1010
RX0,RX1=1166,1662
INK="#1c1917"; MUT="#78716c"; FAINT="#dcd8d2"; BG="#faf8f5"
HUM="#c2410c"; DOG="#0e7490"; SYS="#a8a29e"; VIS="#7c3aed"; RST="#b91c1c"; HIST="#64748b"
ramp=lambda f:"#%02x%02x%02x"%(int(176-146*f),int(183-152*f),int(194-165*f))

def place_labels(items,nodes,x0,x1,y0,ybot):
    """Width-aware labels, clamped to [x0,x1], relaxed against each other and against node discs.
    items: [(i,x,y,r,text)] to label; nodes: [(i,x,y,r)] every disc a label must avoid.
    Returns [(i,lx,ly,right,text)] with ly clamped to [y0+18, ybot]."""
    CW=5.05; L_,R_=x0+10,x1-10
    cx0=float(np.mean([it[1] for it in items])); cy0=float(np.mean([it[2] for it in items])); lab=[]
    for i,x,y,r,t in items:
        t=(t[:28].rstrip()+"…") if len(t)>29 else t; w=len(t)*CW
        vx,vy=x-cx0,y-cy0; Ln=math.hypot(vx,vy) or 1; right=vx>=-0.15*Ln
        if right and x+r+9+w>R_: right=False
        if (not right) and x-r-9-w<L_: right=True
        lx=x+(r+9)*(1 if right else -1)
        lx=min(max(lx,L_+(w if not right else 0)),R_-(w if right else 0))
        lab.append([i,lx,y+(vy/Ln)*10+3.5,right,t,w])
    span=lambda r:(r[1],r[1]+r[5]) if r[3] else (r[1]-r[5],r[1])
    for _ in range(400):
        lab.sort(key=lambda r:r[2]); moved=False
        for a in range(len(lab)-1):
            X_,Y_=lab[a],lab[a+1]; ax0,ax1=span(X_); bx0,bx1=span(Y_)
            if ax0<bx1 and bx0<ax1 and Y_[2]-X_[2]<12.4:
                sh=(12.4-(Y_[2]-X_[2]))/2+0.05; X_[2]-=sh; Y_[2]+=sh; moved=True
        for r in lab:
            lx0,lx1=span(r); ly=r[2]
            for j,nx,ny,nr in nodes:
                if j==r[0]: continue
                nr=nr+3.0
                if lx0-nr<nx<lx1+nr and abs(ny-(ly-3.4))<nr+5.4:
                    push=(nr+5.4)-abs(ny-(ly-3.4))+0.1
                    r[2]+= push if (ly-3.4)>=ny else -push; moved=True
        if not moved: break
    return [(i,lx,min(max(ly,y0+18),ybot),right,t) for i,lx,ly,right,t,w in lab]

def render(cfg,U,resets,n_vlm,P,sim,edges,session,echo_min,space=None):
    """space: None for tfidf, else dict(live, panel, caption, echo_max) for a model-derived layout."""
    resets=set(resets); o=[]; A=o.append
    space=space or {}; live=space.get("live") or [u["i"] for u in U]; parked=[u for u in U if u["i"] not in set(live)]
    echo_max=space.get("echo_max",1.0); say=(lambda u:u["logged"] or u["text"]) if space else (lambda u:u["text"])
    p=P[live].copy()
    # identical utterances collapse to identical coords; nudge them apart deterministically
    tol=space.get("nudge",0.012)*float(np.ptp(p,axis=0).max() or 1)
    for a in range(len(p)):
        for b in range(a+1,len(p)):
            if np.hypot(*(p[a]-p[b]))<tol:
                ang=a*2.399; r=0.055*float(np.ptp(p,axis=0).max() or 1)
                d=np.array([np.cos(ang),np.sin(ang)])*r; p[b]=p[b]+d; p[a]=p[a]-d
    p-=p.min(0); p/=np.clip(p.max(0),1e-9,None); pad=60
    top=GY0+(70 if parked else 0)                 # the parked band takes the top of the panel
    xs=np.zeros(len(U)); ys=np.zeros(len(U))
    xs[live]=GX0+pad+p[:,0]*((GX1-GX0)-2*pad); ys[live]=GY1-pad-p[:,1]*((GY1-top)-2*pad)
    kind=lambda u:"H" if u["speaker"]=="HUMAN" else ("S" if u["type"]=="assistant" else "X")
    K=[kind(u) for u in U]; col={"H":HUM,"S":DOG,"X":SYS}
    rad=[6.5+3.3*math.sqrt(u["dur"] or 1.0) for u in U]
    Sn=[i for i in live if K[i]=="S"]; Hn=[i for i in live if K[i]=="H"]
    med=lambda s:float(np.median([sim[i,j] for a,i in enumerate(s) for j in s[a+1:]])) if len(s)>1 else 0.0
    medS,medH=med(Sn),med(Hn)
    t0,t1=U[0]["t"],U[-1]["t"]

    A(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Helvetica Neue, Helvetica, Arial, sans-serif">')
    A(f'<rect width="{W}" height="{H}" fill="{BG}"/><defs>')
    for k in range(6):
        A(f'<marker id="ar{k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.2" markerHeight="6.2" orient="auto-start-reverse"><path d="M0,1.4 L9,5 L0,8.6 z" fill="{ramp(k/5)}"/></marker>')
    A(f'<marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.2" markerHeight="6.2" orient="auto-start-reverse"><path d="M0,1.4 L9,5 L0,8.6 z" fill="{RST}"/></marker>')
    A('</defs>')
    A(f'<text x="62" y="52" font-size="27" font-weight="700" fill="{INK}" letter-spacing="0.4">SNAPPER  &#215;  ROBERT</text>')
    A(f'<text x="62" y="76" font-size="12.5" fill="{MUT}">{e(session)} &#183; {t0:%Y-%m-%d}, {t0:%H:%M}&#8211;{t1:%H:%M} &#183; {e(cfg["ollama_model"])} on Go2 &#8220;snapper&#8221; &#183; {len(U)} utterances, {len(Sn)} exchanges, {len(resets)} context resets</text>')
    A(f'<line x1="62" y1="96" x2="{RX1}" y2="96" stroke="{FAINT}"/>')
    A(f'<rect x="{GX0}" y="{GY0}" width="{GX1-GX0}" height="{GY1-GY0}" fill="#ffffff" stroke="{FAINT}"/>')
    A(f'<text x="{GX0+14}" y="{GY0-8}" font-size="11.5" font-weight="700" fill="{INK}" letter-spacing="1.2">SEMANTIC SPACE</text>')
    A(f'<text x="{GX0+156}" y="{GY0-8}" font-size="11" fill="{MUT}">{space.get("panel","position = TF-IDF &#8594; LSA-8 &#8594; MDS &#183; proximity = topical similarity &#183; node area &#8733; spoken duration")}</text>')

    # echo arcs, behind everything
    for i,j,s,k in edges:
        x1,y1,x2,y2=xs[i],ys[i],xs[j],ys[j]
        mx,my=(x1+x2)/2,(y1+y2)/2; dx,dy=x2-x1,y2-y1; L=math.hypot(dx,dy) or 1
        cx,cy=mx-dy/L*L*0.20, my+dx/L*L*0.20
        c=DOG if k=="SS" else (HUM if k=="HH" else VIS); f=(s-echo_min)/max(1e-6,echo_max-echo_min)
        A(f'<path d="M{x1:.1f},{y1:.1f} Q{cx:.1f},{cy:.1f} {x2:.1f},{y2:.1f}" fill="none" stroke="{c}" stroke-width="{0.9+1.6*f:.2f}" stroke-opacity="{0.20+0.38*f:.2f}" stroke-dasharray="3.5 3"/>')
    # dialogue spine, colour-ramped by time; severed in red across a reset
    for n_,(a,b) in enumerate(zip(live,live[1:])):
        x1,y1,x2,y2=xs[a],ys[a],xs[b],ys[b]
        dx,dy=x2-x1,y2-y1; L=math.hypot(dx,dy) or 1; ux,uy=dx/L,dy/L
        x1s,y1s=x1+ux*(rad[a]+2.5),y1+uy*(rad[a]+2.5); x2s,y2s=x2-ux*(rad[b]+6.5),y2-uy*(rad[b]+6.5)
        mx,my=(x1s+x2s)/2,(y1s+y2s)/2; cx,cy=mx-uy*L*0.09,my+ux*L*0.09
        if (b in resets) or (a in resets) or any(a<c<b for c in resets):
            A(f'<path d="M{x1s:.1f},{y1s:.1f} Q{cx:.1f},{cy:.1f} {x2s:.1f},{y2s:.1f}" fill="none" stroke="{RST}" stroke-width="1.5" stroke-opacity="0.8" stroke-dasharray="2 5" marker-end="url(#arr)"/>')
        else:
            f=n_/max(1,len(live)-2)
            A(f'<path d="M{x1s:.1f},{y1s:.1f} Q{cx:.1f},{cy:.1f} {x2s:.1f},{y2s:.1f}" fill="none" stroke="{ramp(f)}" stroke-width="0.9" stroke-opacity="0.6" marker-end="url(#ar{min(5,int(f*6))})"/>')

    # labels: width-aware, clamped to the panel, relaxed against each other and against nodes
    lab=place_labels([(i,xs[i],ys[i],rad[i],say(U[i])) for i in live],
                     [(j,xs[j],ys[j],rad[j]) for j in live], GX0,GX1,top,GY1-46)
    for i,lx,ly,right,t in lab:
        A(f'<line x1="{xs[i]:.1f}" y1="{ys[i]:.1f}" x2="{lx:.1f}" y2="{ly-3.4:.1f}" stroke="{FAINT}" stroke-width="0.8"/>')
        A(f'<text x="{lx:.1f}" y="{ly:.1f}" font-size="9.3" fill="{INK}" fill-opacity="0.85" text-anchor="{"start" if right else "end"}">{e(t)}</text>')
    for i in live:
        u=U[i]; k=K[i]; c=col[k]
        if u["type"]=="cue":
            A(f'<circle cx="{xs[i]:.1f}" cy="{ys[i]:.1f}" r="{rad[i]+4.5:.1f}" fill="none" stroke="{RST}" stroke-width="1.5" stroke-dasharray="2.5 2.5"/>')
        A(f'<circle cx="{xs[i]:.1f}" cy="{ys[i]:.1f}" r="{rad[i]:.1f}" fill="{c}" fill-opacity="{0.28 if k=="X" else 0.88}" stroke="{c}" stroke-width="1.4"/>')
        A(f'<text x="{xs[i]:.1f}" y="{ys[i]+3.2:.1f}" font-size="8.2" text-anchor="middle" font-weight="700" fill="{"#ffffff" if k!="X" else MUT}" font-family="SF Mono, Menlo, monospace">{i}</text>')
    A(f'<g font-size="10.4" fill="{MUT}"><text x="{GX0+14}" y="{GY1-30}">median pairwise similarity within speaker &#8212; <tspan fill="{DOG}" font-weight="700">SNAPPER {medS:.2f}</tspan> &#183; <tspan fill="{HUM}" font-weight="700">ROBERT {medH:.2f}</tspan></text>')
    A(f'<text x="{GX0+14}" y="{GY1-14}">{space.get("caption","the dog&#39;s replies are uniformly alike; the questions split between near-repeats and genuinely new ground.")}</text></g>')
    if parked:                                    # no position in this space: a fixed spot, upper left
        A(f'<text x="{GX0+14}" y="{GY0+18}" font-size="9.2" font-weight="700" fill="{MUT}" letter-spacing="0.9">NEVER ENTERED THE MODEL &#183; NO POSITION IN THIS SPACE</text>')
        for n_,u in enumerate(parked):
            x=GX0+24+(n_%3)*218; y=GY0+36+(n_//3)*22; c=HUM if u["speaker"]=="HUMAN" else SYS
            why={"startup":"canned greeting","cue":"canned reset audio"}.get(u["type"]) or \
                ("reset request, intercepted" if any(u["i"]<r_<=u["i"]+1 for r_ in resets) else "no reply logged")
            t=u["text"]; t=(t[:19].rstrip()+"…") if len(t)>20 else t
            A(f'<circle cx="{x}" cy="{y-3}" r="8" fill="none" stroke="{c}" stroke-width="1.3" stroke-dasharray="2.5 2"/>')
            A(f'<text x="{x}" y="{y}" font-size="7.8" text-anchor="middle" font-weight="700" fill="{c}" font-family="SF Mono, Menlo, monospace">{u["i"]}</text>')
            A(f'<text x="{x+13}" y="{y-1}" font-size="9" fill="{INK}" fill-opacity="0.85">{e(t)}<tspan fill="{MUT}" font-style="italic" font-size="8.2" dx="5">{e(why)}</tspan></text>')
    return o, xs, ys, medS, medH

def render_rail(o,cfg,U,resets,n_vlm,echo_min,foot=None,sp_src=None):
    A=o.append; resets=set(resets)
    A(f'<line x1="{RX0-24}" y1="132" x2="{RX0-24}" y2="1010" stroke="{FAINT}"/>')
    A(f'<text x="{RX0}" y="124" font-size="11.5" font-weight="700" fill="{INK}" letter-spacing="1.2">PROMPT PAYLOAD</text>')
    A(f'<text x="{RX0}" y="150" font-size="10.6" fill="{MUT}">what was actually in the context window when the dog answered</text>')
    sp=cfg["system_prompt"]; spt=max(1,round(len(sp)/4))
    A(f'<rect x="{RX0}" y="164" width="{RX1-RX0}" height="60" fill="#ffffff" stroke="{FAINT}"/>')
    A(f'<rect x="{RX0}" y="164" width="4" height="60" fill="{SYS}"/>')
    A(f'<text x="{RX0+16}" y="182" font-size="9.4" font-weight="700" fill="{SYS}" letter-spacing="0.8">SYSTEM PROMPT &#183; {f"INFERRED &#183; {spt} tok &#183; "+e(sp_src.replace("performance-script","script")) if sp_src else f"PERSISTENT &#183; {spt} tok"}</text>')
    sl=textwrap.wrap(" ".join(sp.split()),width=72)
    A(f'<text x="{RX0+16}" y="199" font-size="10.2" fill="{INK}" font-style="italic">{e(sl[0])}</text>')
    if len(sl)>1: A(f'<text x="{RX0+16}" y="214" font-size="10.2" fill="{INK}" font-style="italic">{e(sl[1])}{"…" if len(sl)>2 else ""}</text>')
    asst=[u for u in U if u["type"]=="assistant"]
    BX=RX0+54; BW=286; Y0=268; RH=35.5
    peak=max(sum(u["payload"].values()) for u in asst); sc=BW/max(480.0,peak*1.08)
    A(f'<text x="{RX0}" y="250" font-size="9.6" font-weight="700" fill="{MUT}" letter-spacing="0.9">TURN</text>')
    A(f'<text x="{BX}" y="250" font-size="9.6" font-weight="700" fill="{MUT}" letter-spacing="0.9">SYSTEM &#183; VISUAL &#183; HISTORY &#183; USER</text>')
    A(f'<text x="{RX1}" y="250" font-size="9.6" font-weight="700" fill="{MUT}" letter-spacing="0.9" text-anchor="end">TOK</text>')
    rows=0; seen=set()
    for u in asst:
        y=Y0+rows*RH
        prev=[q for q in U if q["i"]<u["i"] and q["type"]=="cue"]
        if prev and prev[-1]["i"] not in seen:
            seen.add(prev[-1]["i"])
            A(f'<line x1="{RX0}" y1="{y-13:.1f}" x2="{RX1}" y2="{y-13:.1f}" stroke="{RST}" stroke-width="1.2" stroke-dasharray="3 3"/>')
            A(f'<text x="{RX1}" y="{y-17:.1f}" font-size="9" fill="{RST}" text-anchor="end" font-weight="700">&#8593; &#8220;let\'s start over&#8221; &#183; history wiped</text>')
            y+=14; Y0+=14
        p=u["payload"]; tot=sum(p.values())
        A(f'<text x="{RX0}" y="{y+4:.1f}" font-size="10" fill="{INK}" font-family="SF Mono, Menlo, monospace">{u["i"]:>2}</text>')
        A(f'<text x="{RX0+22}" y="{y+4:.1f}" font-size="9" fill="{MUT}" font-family="SF Mono, Menlo, monospace">{u["t"]:%H:%M}</text>')
        x=BX
        for k,c in (("system",SYS),("visual",VIS),("history",HIST),("user",HUM)):
            w=p[k]*sc
            if w>0.4: A(f'<rect x="{x:.1f}" y="{y-8:.1f}" width="{w:.1f}" height="13" fill="{c}" fill-opacity="0.85"/>')
            x+=w
        A(f'<rect x="{BX}" y="{y-8:.1f}" width="{BW}" height="13" fill="none" stroke="{FAINT}" stroke-width="0.7"/>')
        A(f'<text x="{RX1}" y="{y+4:.1f}" font-size="10" fill="{INK}" text-anchor="end" font-family="SF Mono, Menlo, monospace">{tot}</text>')
        rows+=1
    YB=Y0+rows*RH; ctx=cfg.get("ollama_num_ctx",2048)
    A(f'<text x="{BX}" y="{YB+6:.1f}" font-size="9.6" fill="{MUT}">payload peaks at {peak} tok &#8212; {100*peak/ctx:.0f}% of the {ctx}-token context window. It never fills.</text>')
    LY=YB+34
    A(f'<line x1="{RX0}" y1="{LY-16:.1f}" x2="{RX1}" y2="{LY-16:.1f}" stroke="{FAINT}"/>')
    A(f'<text x="{RX0}" y="{LY+2:.1f}" font-size="10.4" font-weight="700" fill="{INK}" letter-spacing="1.1">KEY</text>')
    items=[(HUM,"circle","ROBERT &#8212; spoken question"),(DOG,"circle","SNAPPER &#8212; generated reply"),
           (SYS,"circle","system utterance / reset cue"),(INK,"ramp","dialogue order &#8212; pale = first, dark = last"),
           (RST,"dash","context wipe &#8212; the thread is severed"),(DOG,"arc","echo between two replies"),
           (HUM,"arc","echo between two questions"),(VIS,"arc","echo across speakers")]
    for n,(c,shape,txt) in enumerate(items):
        yy=LY+22+n*19; x=RX0+10
        if shape=="circle": A(f'<circle cx="{x+4}" cy="{yy-3.5}" r="5.4" fill="{c}" fill-opacity="0.88" stroke="{c}"/>')
        elif shape=="ramp":
            for q in range(6): A(f'<line x1="{x-3+q*2.9:.1f}" y1="{yy-3.5}" x2="{x-3+(q+1)*2.9:.1f}" y2="{yy-3.5}" stroke="{ramp(q/5)}" stroke-width="1.6"/>')
            A(f'<line x1="{x+14}" y1="{yy-3.5}" x2="{x+16}" y2="{yy-3.5}" stroke="#1e293b" stroke-width="1.6" marker-end="url(#ar5)"/>')
        elif shape=="dash": A(f'<line x1="{x-3}" y1="{yy-3.5}" x2="{x+13}" y2="{yy-3.5}" stroke="{c}" stroke-width="1.6" stroke-dasharray="2 4"/>')
        else: A(f'<path d="M{x-3},{yy-1} Q{x+5},{yy-11} {x+13},{yy-1}" fill="none" stroke="{c}" stroke-width="1.4" stroke-opacity="0.5" stroke-dasharray="3.5 3"/>')
        A(f'<text x="{x+26}" y="{yy}" font-size="10.2" fill="{INK}" fill-opacity="0.85">{txt}</text>')
    A(f'<line x1="62" y1="1042" x2="{RX1}" y2="1042" stroke="{FAINT}"/>')
    foot=foot or [f'Structural layer of a three-part series. Source: transcript/session-complete.jsonl (re-transcribed, distil-large-v3) + {n_vlm} interleaved VLM scene captions from the same {e(cfg["vlm_model"])} weights.',
          f'Semantic positions are lexical, not neural: TF-IDF over word 1&#8211;2 grams, reduced to 8 LSA components, embedded to 2D by metric MDS. Two utterances sit close when they share vocabulary and',
          f'co-occurrence structure &#8212; not because a language model judged them alike. Echo arcs drawn at cosine &#8805; {echo_min:.2f}. Payload counts are chars/4 estimates from the logged config (history_truncation_limit {cfg["history_truncation_limit"]}, num_ctx {cfg.get("ollama_num_ctx")}).']
    for n,t in enumerate(foot): A(f'<text x="62" y="{1064+n*17}" font-size="10" fill="{MUT}">{t}</text>')
    A(f'<text x="62" y="{1064+3*17+6}" font-size="10" fill="{MUT}">Robert Twomey &#183; BFF / Dog Walk &#183; rendered {dt.date.today():%Y-%m-%d}</text>')
    A("</svg>")

def export_png(svg,png,width):
    # rasterise with whichever renderer is on PATH; inkscape is the reference
    t0=os.path.getmtime(svg)
    for tool,cmd in (("inkscape",["inkscape",svg,"-o",png,"-w",str(width)]),
                     ("rsvg-convert",["rsvg-convert","-w",str(width),"-o",png,svg])):
        if not shutil.which(tool): continue
        r=subprocess.run(cmd,capture_output=True,text=True)
        if r.returncode!=0 or not os.path.exists(png) or os.path.getmtime(png)<t0:
            sys.exit(f"{tool} failed to export {png}:\n{r.stderr.strip()}")
        return tool
    sys.exit("--png needs inkscape or rsvg-convert on PATH")

def model_space(a,cfg,U,n_vlm,out):
    """Layout, similarities and wording for --space activation or standalone."""
    import activation_graph as ag
    stem=os.path.splitext(out)[0]; n=len(U); nL=None; sp_src=None
    if a.space=="activation":
        cache=a.acts or stem+".acts.npz"
        sp,src=ag.resolve_system_prompt(a.system_prompt,cfg); z=np.load(cache) if os.path.exists(cache) else None
        if z is not None and a.system_prompt and str(z["system_prompt"])!=sp: z=None
        if z is None:
            if not a.model: sys.exit(f"{cache} is missing or built with another prompt: pass --model to replay, or --acts, or --space tfidf")
            ex,off=ag.build_replay(a.session_dir,cfg,U,system_prompt=sp); ls=ag.first_lowstate(a.session_dir)
            packet=ag.body_packet(ls) if ls else None
            acts,ntok=ag.extract(ex,a.model,packet)
            np.savez_compressed(cache,**acts,ntok=ntok,user_i=[x["user_i"] for x in ex],asst_i=[x["asst_i"] for x in ex],
                                off=off,packet=packet or "",system_prompt=sp,system_prompt_source=src)
            z=np.load(cache)
        sp=str(z["system_prompt"]) if "system_prompt" in z.files else cfg["system_prompt"].strip()
        src=str(z["system_prompt_source"]) if "system_prompt_source" in z.files else "logged config.system_prompt"
        nL=z["user"].shape[1]; layer=max(0,min(a.layer,nL-1)); r=ag.analyse(z,U,layer,[])
        live=list(r["order"]); P=np.zeros((n,2)); P[live]=r["P"]
        sim=np.full((n,n),np.nan); sim[np.ix_(live,live)]=r["sim"]
        how=f"position = hidden state at layer {layer} of {nL-1}, in context"
        what=f"Positions come from the model: each utterance&#8217;s hidden state at layer {layer} of {nL-1} inside the prompt the dog received, mean over its tokens, z-scored, compared by cosine, embedded to 2D by metric MDS."
        then="Activations in context also carry position in the conversation."
    else:
        cache=stem+".standalone.npz"
        if os.path.exists(cache): H=np.load(cache)["H"]
        else:
            if not a.model: sys.exit(f"{cache} is missing: pass --model to build it")
            H=ag.extract_standalone(U,a.model); np.savez_compressed(cache,H=H)
        nL=H.shape[1]; layer=max(0,min(a.layer,nL-1)); r=ag.space_standalone(H,U,layer)
        live=list(range(n)); P,sim=r["P"],r["sim"]; sp,src=ag.resolve_system_prompt(a.system_prompt,cfg)
        how=f"position = hidden state at layer {layer} of {nL-1}, each utterance alone"
        what=f"Positions come from the model: each utterance run alone, with no prompt or history, hidden state at layer {layer} of {nL-1}, mean over its tokens, z-scored, compared by cosine, embedded to 2D by metric MDS."
        then="This shows what each sentence means to the model on its own, not how it was met in the conversation."
    if src and src!="logged config.system_prompt": sp_src=src
    cfg=dict(cfg,system_prompt=sp)                 # the rail and its token counts follow the replayed prompt
    for u in U:
        if u.get("payload"): u["payload"]["system"]=max(1,round(len(sp)/4))
    ed=r["echoes"]; lo,hi=min(x[2] for x in ed),max(x[2] for x in ed)
    space=dict(live=live,echo_max=hi,nudge=0.012 if a.space=="activation" else 0.035,panel=f"{how} &#8594; z-score &#8594; cosine &#8594; MDS &#183; node area &#8733; spoken duration",
               caption=f"dashed arcs: the 30 most similar non-adjacent pairs &#183; the plane keeps the full-dimension distance ranks at &#961; = {r['fid']:.2f}")
    foot=[f"Structural layer of a three-part series. Source: transcript/session-complete.jsonl + {n_vlm} interleaved VLM scene captions. Replayed through google/gemma-4-E2B-it (bf16); the dog ran its Q4_K_M quantization. Labels show the text the model received.",
          what,
          f"{then} Payload counts are chars/4 estimates from the logged config (history_truncation_limit {cfg['history_truncation_limit']}, num_ctx {cfg.get('ollama_num_ctx')}) and exclude the unlogged body-state packet."]
    return cfg,space,P,sim,ed,foot,sp_src,lo

def main():
    ap=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_dir"); ap.add_argument("-o","--out", default=None)
    ap.add_argument("--space", choices=("activation","standalone","tfidf"), default="activation",
                    help="what places the nodes (default activation; see above)")
    ap.add_argument("--layer", type=int, default=18, help="model depth for activation/standalone (default 18)")
    ap.add_argument("--acts", default=None, help="activation cache (default OUT.acts.npz)")
    ap.add_argument("--model", default=None, help="local google/gemma-4-E2B-it, to build a missing cache")
    ap.add_argument("--system-prompt", default=None, metavar="SCRIPT_OR_TXT",
                    help="system prompt for an activation replay, as in activation_graph.py")
    ap.add_argument("--echo-threshold", type=float, default=0.86,
                    help="tfidf: min cosine for an echo arc (default 0.86). Model spaces draw the 30 nearest pairs")
    ap.add_argument("--png", action="store_true",
                    help="also export a PNG next to the SVG (needs inkscape or rsvg-convert)")
    ap.add_argument("--png-width", type=int, default=2*W,
                    help=f"PNG width in px (default {2*W}, twice the SVG canvas)")
    a=ap.parse_args()
    cfg,U,resets,n_vlm=load(a.session_dir)
    session=os.path.basename(a.session_dir.rstrip("/"))
    if "__" in session: session=[s for s in session.split("__") if s.startswith("session-")][0]
    out=a.out or f"{session}-dialogue-graph.svg"
    if a.space=="tfidf":
        P,sim,edges=embed(U,a.echo_threshold); space,foot,sp_src,echo_min=None,None,None,a.echo_threshold
    else:
        cfg,space,P,sim,edges,foot,sp_src,echo_min=model_space(a,cfg,U,n_vlm,out)
    o,xs,ys,medS,medH=render(cfg,U,resets,n_vlm,P,sim,edges,session,echo_min,space)
    render_rail(o,cfg,U,resets,n_vlm,echo_min,foot,sp_src)
    open(out,"w").write("\n".join(o))
    print(f"{out}  ({len(U)} utterances, {len(edges)} echo arcs, {len(resets)} resets, "
          f"median sim SNAPPER {medS:.2f} / ROBERT {medH:.2f})")
    if a.png:
        png=os.path.splitext(out)[0]+".png"
        tool=export_png(out,png,a.png_width)
        print(f"{png}  ({a.png_width} px wide, via {tool})")

if __name__=="__main__": main()
