#!/usr/bin/env python3
"""
similarity_grid.py — how alike every pair of utterances is, at every depth of Gemma 4.

A 30x30 similarity matrix per layer (36 small multiples), rows and columns ordered as
Robert's 15 questions then Snapper's 15 replies, each block in time order: speaker
structure shows as blocks, and each question's own reply falls on the off-block diagonal.
One layer is enlarged with turn labels, beside the block averages traced through depth.

usage:
  python3 similarity_grid.py SESSION_DIR --acts ACTS.npz [--model MODEL_DIR] [-o OUT.svg] [--focus 15] [--png]

ACTS.npz is the activation cache written by activation_graph.py; if it is missing, --model
is used to replay the session and write it.
"""
import os
import sys
import json
import math
import argparse
import datetime as dt
import numpy as np
import conversation_graph as cg
import activation_graph as ag

# --- diverging scale: red = more alike than this conversation's average pair, blue = less alike
def _lin(c): c=np.asarray(c,float)/255; return np.where(c<=0.04045,c/12.92,((c+0.055)/1.055)**2.4)
def _srgb(c): c=np.clip(c,0,1); return np.where(c<=0.0031308,12.92*c,1.055*np.power(c,1/2.4)-0.055)*255
_M1=np.array([[0.4122214708,0.5363325363,0.0514459929],[0.2119034982,0.6806995451,0.1073969566],[0.0883024619,0.2817188376,0.6299787005]])
_M2=np.array([[0.2104542553,0.7936177850,-0.0040720468],[1.9779984951,-2.4285922050,0.4505937099],[0.0259040371,0.7827717662,-0.8086757660]])
def oklab(h): return _M2@np.cbrt(_M1@_lin([int(h[i:i+2],16) for i in (1,3,5)]))
def hexof(lab):
    lms=(np.linalg.inv(_M2)@lab)**3; rgb=_srgb(np.linalg.inv(_M1)@lms)
    return "#%02x%02x%02x"%tuple(int(round(float(x))) for x in rgb)
MID="#f0efec"; BLUE="#1c5cab"
_Lb=oklab(BLUE)[0]; RED=hexof(np.array([_Lb,0.165*math.cos(math.radians(27)),0.165*math.sin(math.radians(27))]))
_mid,_red,_blue=oklab(MID),oklab(RED),oklab(BLUE)
def diverging(v,lim):
    t=max(-1.0,min(1.0,v/lim)); pole=_red if t>=0 else _blue
    return hexof(_mid+(pole-_mid)*abs(t))

def matrices(z):
    X=np.concatenate([z["user"],z["asst"]],0)          # [Q1..Q15 | R1..R15, layers, d]
    S=[]
    for L in range(X.shape[1]):
        f=ag.std_frame(X[:,L]); S.append(1.0-ag.cosdist(f(X[:,L])))
    return np.stack(S)                                   # [layers, 30, 30]

def profile(S,n):
    off=~np.eye(n,dtype=bool); out={k:[] for k in ("qq","rr","own","other")}
    for M in S:
        out["qq"].append(M[:n,:n][off].mean()); out["rr"].append(M[n:,n:][off].mean())
        qr=M[:n,n:]; out["own"].append(np.diag(qr).mean()); out["other"].append(qr[off].mean())
    return {k:np.array(v) for k,v in out.items()}

def render(U,z,S,prof,resets,session,focus,kv_from,lim=0.6,sp_src="logged config.system_prompt"):
    from conversation_graph import W,H,INK,MUT,FAINT,BG,e
    n=len(z["user_i"]); N=2*n; nL=S.shape[0]; o=[]; A=o.append
    turns=[int(i) for i in z["user_i"]]+[int(i) for i in z["asst_i"]]
    # a context wipe separates exchange k from k-1 when a reset cue sits between them
    cuts=[k for k in range(1,n) if any(int(z["asst_i"][k-1])<c<int(z["user_i"][k]) for c in resets)]
    t0,t1=U[0]["t"],U[-1]["t"]
    A(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Helvetica Neue, Helvetica, Arial, sans-serif">')
    A(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
    A(f'<text x="62" y="52" font-size="27" font-weight="700" fill="{INK}" letter-spacing="0.4">SNAPPER  &#215;  ROBERT</text>')
    A(f'<text x="62" y="76" font-size="12.5" fill="{MUT}">{e(session)} &#183; {t0:%Y-%m-%d}, {t0:%H:%M}&#8211;{t1:%H:%M} &#183; how alike every pair of utterances is, at each of Gemma 4&#8217;s {nL} depths</text>')
    A(f'<line x1="62" y1="96" x2="1662" y2="96" stroke="{FAINT}"/>')

    # ---------------- the grid of 36 ----------------
    GX,GY,TW,TH,m=62,132,150,146,120; c=m/N
    A(f'<text x="{GX}" y="{GY-10}" font-size="11.5" font-weight="700" fill="{INK}" letter-spacing="1.2">EVERY DEPTH</text>')
    A(f'<text x="{GX+104}" y="{GY-10}" font-size="11" fill="{MUT}">layer 0 is the token embeddings; layers {kv_from}&#8211;{nL-1} reuse keys and values computed earlier</text>')
    tags={0:"tokens",focus:"enlarged",18:"activation figure"}
    if kv_from: tags.setdefault(kv_from,"K/V shared from here")
    for L in range(nL):
        tx=GX+(L%6)*TW; ty=GY+(L//6)*TH
        if kv_from and L>=kv_from:
            A(f'<rect x="{tx-6}" y="{ty-2}" width="{TW-18}" height="{TH-4}" fill="{FAINT}" fill-opacity="0.35"/>')
        tag=f'<tspan dx="5" font-weight="400" fill="{MUT}" font-size="9">{tags[L]}</tspan>' if L in tags else ""
        A(f'<text x="{tx}" y="{ty+11}" font-size="10" font-weight="700" fill="{INK}">L{L}{tag}</text>')
        mx,my=tx,ty+17
        for a in range(N):
            for b in range(N):
                if a==b: continue
                gx=1.5 if b>=n else 0; gy=1.5 if a>=n else 0
                A(f'<rect x="{mx+b*c+gx:.2f}" y="{my+a*c+gy:.2f}" width="{c+0.05:.2f}" height="{c+0.05:.2f}" fill="{diverging(S[L,a,b],lim)}"/>')
        if L==focus:
            A(f'<rect x="{mx-2}" y="{my-2}" width="{m+5.5}" height="{m+5.5}" fill="none" stroke="{INK}" stroke-width="1.2"/>')

    # ---------------- rail: one layer enlarged ----------------
    RX=1000; A(f'<line x1="{RX-20}" y1="132" x2="{RX-20}" y2="1010" stroke="{FAINT}"/>')
    A(f'<text x="{RX}" y="{GY-10}" font-size="11.5" font-weight="700" fill="{INK}" letter-spacing="1.2">LAYER {focus}, ENLARGED</text>')
    EX,EY,ec=RX+64,GY+42,13.0; gap=3
    def ex_(b): return EX+b*ec+(gap if b>=n else 0)
    def ey_(a): return EY+a*ec+(gap if a>=n else 0)
    for a in range(N):
        for b in range(N):
            if a==b:
                A(f'<rect x="{ex_(b):.1f}" y="{ey_(a):.1f}" width="{ec-1}" height="{ec-1}" fill="#ffffff" stroke="{FAINT}" stroke-width="0.6"/>'); continue
            A(f'<rect x="{ex_(b):.1f}" y="{ey_(a):.1f}" width="{ec-1}" height="{ec-1}" fill="{diverging(S[focus,a,b],lim)}"><title>turn {turns[a]} &#215; turn {turns[b]}: {S[focus,a,b]:+.2f}</title></rect>')
    for k in range(n):   # each question's own reply
        A(f'<rect x="{ex_(n+k)-0.5:.1f}" y="{ey_(k)-0.5:.1f}" width="{ec}" height="{ec}" fill="none" stroke="{INK}" stroke-width="1.1"/>')
        A(f'<rect x="{ex_(k)-0.5:.1f}" y="{ey_(n+k)-0.5:.1f}" width="{ec}" height="{ec}" fill="none" stroke="{INK}" stroke-width="1.1"/>')
    for k in cuts:       # context wipes, drawn through both blocks
        for off in (0,n):
            A(f'<line x1="{EX-4}" y1="{ey_(off+k)-1:.1f}" x2="{ex_(N-1)+ec+3:.1f}" y2="{ey_(off+k)-1:.1f}" stroke="{INK}" stroke-width="0.8" stroke-dasharray="2 2"/>')
            A(f'<line x1="{ex_(off+k)-1:.1f}" y1="{EY-4}" x2="{ex_(off+k)-1:.1f}" y2="{ey_(N-1)+ec+3:.1f}" stroke="{INK}" stroke-width="0.8" stroke-dasharray="2 2"/>')
    for a in range(N):
        A(f'<text x="{EX-5}" y="{ey_(a)+ec-3.3:.1f}" font-size="7.8" fill="{MUT}" text-anchor="end" font-family="SF Mono, Menlo, monospace">{turns[a]}</text>')
        A(f'<text x="{ex_(a)+ec/2-0.5:.1f}" y="{EY-6}" font-size="7.8" fill="{MUT}" text-anchor="middle" font-family="SF Mono, Menlo, monospace">{turns[a]}</text>')
    blk=[("ROBERT&#8217;S QUESTIONS",0),("SNAPPER&#8217;S REPLIES",n)]
    for label,off in blk:
        cy_=ey_(off)+n*ec/2; cx_=ex_(off)+n*ec/2
        A(f'<text x="{EX-30}" y="{cy_:.1f}" font-size="9" font-weight="700" fill="{INK}" letter-spacing="0.8" text-anchor="middle" transform="rotate(-90 {EX-30} {cy_:.1f})">{label}</text>')
        A(f'<text x="{cx_:.1f}" y="{EY-18}" font-size="9" font-weight="700" fill="{INK}" letter-spacing="0.8" text-anchor="middle">{label}</text>')
    # legend: the diverging bar
    LX,LY=ex_(N-1)+ec+34,EY; LH=ec*N+gap
    for s in range(60):
        v=lim-(2*lim)*s/59
        A(f'<rect x="{LX}" y="{LY+s*LH/60:.1f}" width="14" height="{LH/60+0.4:.2f}" fill="{diverging(v,lim)}"/>')
    for v,txt in ((lim,f"+{lim:.1f} more alike"),(0,"0 an average pair"),(-lim,f"&#8722;{lim:.1f} less alike")):
        y=LY+(lim-v)/(2*lim)*LH
        A(f'<line x1="{LX+14}" y1="{y:.1f}" x2="{LX+19}" y2="{y:.1f}" stroke="{MUT}" stroke-width="0.8"/>')
        A(f'<text x="{LX+23}" y="{y+3:.1f}" font-size="9" fill="{INK}" fill-opacity="0.85">{txt}</text>')
    A(f'<rect x="{LX}" y="{LY+LH+16}" width="12" height="12" fill="none" stroke="{INK}" stroke-width="1.1"/>')
    A(f'<text x="{LX+18}" y="{LY+LH+26}" font-size="9" fill="{INK}" fill-opacity="0.85">a question and</text>')
    A(f'<text x="{LX+18}" y="{LY+LH+38}" font-size="9" fill="{INK}" fill-opacity="0.85">its own reply</text>')
    A(f'<line x1="{LX}" y1="{LY+LH+54}" x2="{LX+14}" y2="{LY+LH+54}" stroke="{INK}" stroke-width="0.8" stroke-dasharray="2 2"/>')
    A(f'<text x="{LX+18}" y="{LY+LH+57}" font-size="9" fill="{INK}" fill-opacity="0.85">context wiped</text>')

    # ---------------- rail: the block averages through depth ----------------
    cx0,cx1,cy0,cy1=RX+34,1640,660,880; ylo,yhi=-0.4,0.4
    X_=lambda L:cx0+(cx1-cx0)*L/(nL-1); Y_=lambda v:cy1-(cy1-cy0)*(min(max(v,ylo),yhi)-ylo)/(yhi-ylo)
    A(f'<text x="{RX}" y="{cy0-24}" font-size="10.4" font-weight="700" fill="{INK}" letter-spacing="1.1">THE BLOCKS, AVERAGED, THROUGH DEPTH</text>')
    if kv_from:
        A(f'<rect x="{X_(kv_from-0.5):.1f}" y="{cy0}" width="{X_(nL-1)-X_(kv_from-0.5):.1f}" height="{cy1-cy0}" fill="{FAINT}" fill-opacity="0.35"/>')
    for v in (-0.4,-0.2,0,0.2,0.4):
        A(f'<line x1="{cx0}" y1="{Y_(v):.1f}" x2="{cx1}" y2="{Y_(v):.1f}" stroke="{"#c3c2b7" if v==0 else FAINT}" stroke-width="{0.9 if v==0 else 0.6}"/>')
        A(f'<text x="{cx0-6}" y="{Y_(v)+3:.1f}" font-size="8.4" fill="{MUT}" text-anchor="end" font-family="SF Mono, Menlo, monospace">{v:+.1f}</text>')
    for L in range(0,nL,5):
        A(f'<text x="{X_(L):.1f}" y="{cy1+13}" font-size="8.4" fill="{MUT}" text-anchor="middle" font-family="SF Mono, Menlo, monospace">{L}</text>')
    A(f'<text x="{(cx0+cx1)/2:.1f}" y="{cy1+27}" font-size="8.6" fill="{MUT}" text-anchor="middle">layer</text>')
    for L_,txt in ((focus,f"L{focus}"),(18,"L18")):
        A(f'<line x1="{X_(L_):.1f}" y1="{cy0}" x2="{X_(L_):.1f}" y2="{cy1}" stroke="{INK}" stroke-width="0.7" stroke-dasharray="2 2"/>')
        A(f'<text x="{X_(L_):.1f}" y="{cy0-5}" font-size="8.6" fill="{MUT}" text-anchor="middle" font-family="SF Mono, Menlo, monospace">{txt}</text>')
    series=[("rr","reply vs reply",INK,""),("qq","question vs question",INK,"5 3"),
            ("own","question vs its own reply",INK,"1.4 2.6"),("other","question vs another reply",MUT,"")]
    for key,label,colr,dash in series:
        v=prof[key]
        A(f'<polyline points="{" ".join(f"{X_(L):.1f},{Y_(x):.1f}" for L,x in enumerate(v))}" fill="none" stroke="{colr}" stroke-width="2"{f" stroke-dasharray=\"{dash}\"" if dash else ""} stroke-linejoin="round"/>')
    # direct labels where the four lines are furthest apart, haloed so they sit legibly on the lines
    LL=9
    for key,label,colr,dash in series:
        A(f'<text x="{X_(LL):.1f}" y="{Y_(prof[key][LL])-5:.1f}" font-size="9.2" fill="{INK}" stroke="{BG}" stroke-width="3" paint-order="stroke" stroke-linejoin="round">{label}</text>')
    # legend under the chart (the lines share one ink, so dash carries identity)
    for k,(key,label,colr,dash) in enumerate(series):
        lx=RX+34+(k%2)*300; ly=cy1+48+(k//2)*16
        A(f'<line x1="{lx}" y1="{ly-3.5}" x2="{lx+24}" y2="{ly-3.5}" stroke="{colr}" stroke-width="2"{f" stroke-dasharray=\"{dash}\"" if dash else ""}/>')
        A(f'<text x="{lx+32}" y="{ly}" font-size="9.8" fill="{INK}" fill-opacity="0.85">{label}</text>')
    pk=int(np.argmax(prof["rr"]-prof["other"])); gapL=int(np.argmax(prof["own"]-prof["other"]))
    A(f'<text x="{RX}" y="{cy1+96}" font-size="9.8" fill="{MUT}">Replies are most alike each other at layer {pk}; a reply is furthest above an unrelated one</text>')
    A(f'<text x="{RX}" y="{cy1+110}" font-size="9.8" fill="{MUT}">in resembling its own question at layer {gapL} (+{(prof["own"]-prof["other"])[gapL]:.2f}), just after keys and values start being shared.</text>')

    A(f'<line x1="62" y1="1042" x2="1662" y2="1042" stroke="{FAINT}"/>')
    foot=["Each matrix: rows and columns are Robert&#8217;s 15 questions (as the model received them), then Snapper&#8217;s 15 replies, each in time order. A cell is the cosine similarity of two utterances&#8217; mean hidden states,",
          "after every dimension is z-scored across the 30 utterances &#8212; so 0 means as alike as an average pair in this conversation. Same activations as the activation-space figure: each prompt replayed as the dog received it,",
          f"the logged reply teacher-forced through google/gemma-4-E2B-it (bf16), system prompt from the {e(sp_src)}. Colour saturates at &#177;{lim:.1f}; the diagonal is left blank. Hover a cell in the enlarged matrix for its value."]
    for k,t in enumerate(foot): A(f'<text x="62" y="{1064+k*17}" font-size="10" fill="{MUT}">{t}</text>')
    A(f'<text x="62" y="{1064+3*17+6}" font-size="10" fill="{MUT}">Robert Twomey &#183; BFF / Dog Walk &#183; rendered {dt.date.today():%Y-%m-%d}</text>')
    A("</svg>")
    return o

def main():
    ap=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_dir"); ap.add_argument("--acts", required=True, help="activation cache (.npz)")
    ap.add_argument("--model", default=None, help="local google/gemma-4-E2B-it, needed only if --acts is missing")
    ap.add_argument("-o","--out", default=None); ap.add_argument("--focus", type=int, default=15,
                    help="layer to enlarge (default 15, where the speakers separate most)")
    ap.add_argument("--system-prompt", default=None, metavar="SCRIPT_OR_TXT",
                    help="system prompt for a fresh replay (see activation_graph.py); ignored when --acts exists")
    ap.add_argument("--png", action="store_true"); ap.add_argument("--png-width", type=int, default=3440)
    a=ap.parse_args()
    cfg,U,resets,_=cg.load(a.session_dir)
    session=os.path.basename(a.session_dir.rstrip("/"))
    if "__" in session: session=[s for s in session.split("__") if s.startswith("session-")][0]
    if not os.path.exists(a.acts):
        if not a.model: sys.exit(f"{a.acts} not found; pass --model to replay the session and create it")
        sp,sp_src=ag.resolve_system_prompt(a.system_prompt,cfg)
        ex,off=ag.build_replay(a.session_dir,cfg,U,system_prompt=sp); ls=ag.first_lowstate(a.session_dir)
        acts,ntok=ag.extract(ex,a.model,ag.body_packet(ls) if ls else None)
        np.savez_compressed(a.acts,**acts,ntok=ntok,user_i=[x["user_i"] for x in ex],asst_i=[x["asst_i"] for x in ex],off=off,
                            system_prompt=sp,system_prompt_source=sp_src)
    z=np.load(a.acts)
    kv_from=None
    if a.model and os.path.exists(os.path.join(a.model,"config.json")):
        tc=json.load(open(os.path.join(a.model,"config.json"))); tc=tc.get("text_config",tc)
        kv_from=tc["num_hidden_layers"]-tc["num_kv_shared_layers"]+1
    S=matrices(z); prof=profile(S,len(z["user_i"]))
    out=a.out or f"{session}-similarity-grid.svg"
    sp_src=str(z["system_prompt_source"]) if "system_prompt_source" in z.files else "logged config.system_prompt"
    open(out,"w").write("\n".join(render(U,z,S,prof,resets,session,max(0,min(a.focus,S.shape[0]-1)),kv_from,sp_src=sp_src)))
    print(f"{out}  ({S.shape[0]} layers x {S.shape[1]}x{S.shape[2]})")
    if a.png:
        png=os.path.splitext(out)[0]+".png"; print(f"{png}  ({a.png_width} px wide, via {cg.export_png(out,png,a.png_width)})")

if __name__=="__main__": main()
