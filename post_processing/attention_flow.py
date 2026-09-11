#!/usr/bin/env python3
"""
attention_flow.py — where Snapper's attention went while it produced each reply.

For every reply, the share of its attention (averaged over the reply's tokens and Gemma 4's
8 heads) that landed on each part of the prompt, at each of the 35 layers: its own words so far,
Robert's question, the visual-context packet, earlier turns, the persona prompt, the chat
template's scaffolding, and the <bos> sink. A dot plot puts question and scene on one scale;
an arc diagram shows which earlier turns each reply reached back to.

usage:
  python3 attention_flow.py SESSION_DIR --attn ATTN.json [--model MODEL_DIR] [--system-prompt SCRIPT] [-o OUT.svg] [--png]

ATTN.json is written by a replay (activation_graph.extract_attention_surprise); if it is missing,
--model (and optionally --system-prompt) are used to create it.
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

# prompt parts, stacked bottom -> top; content hues validated for CVD in this adjacency order
STACK=[("question","Robert’s question","#c2410c"),("visual","visual-context packet","#7c3aed"),
       ("history","earlier turns","#3a6aa7"),("system","persona prompt","#a16207"),
       ("reply","its own words so far","#007699"),("scaffold","template scaffolding","#d6d3d1"),
       ("sink","<bos> sink","#ebe9e6")]

def render(U,r,resets,session,sp_src):
    from conversation_graph import W,H,GX0,GX1,RX0,RX1,INK,MUT,FAINT,BG,HUM,DOG,e
    X=r["exchanges"]; P=r["parts"]; nL=r["layers"]; o=[]; A=o.append
    col={k:c for k,_,c in STACK}; order=[k for k,_,_ in STACK]
    sh=[np.array(x["shares"])[:,[P.index(k) for k in order]] for x in X]      # [layers, parts] per reply
    t0,t1=U[0]["t"],U[-1]["t"]
    A(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Helvetica Neue, Helvetica, Arial, sans-serif">')
    A(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
    A(f'<text x="62" y="52" font-size="27" font-weight="700" fill="{INK}" letter-spacing="0.4">SNAPPER  &#215;  ROBERT</text>')
    A(f'<text x="62" y="76" font-size="12.5" fill="{MUT}">{e(session)} &#183; {t0:%Y-%m-%d}, {t0:%H:%M}&#8211;{t1:%H:%M} &#183; where each reply&#8217;s attention went, replayed through google/gemma-4-E2B-it</text>')
    A(f'<line x1="62" y1="96" x2="{RX1}" y2="96" stroke="{FAINT}"/>')

    # ---------------- rows: one per reply ----------------
    SX0,SX1=300,860; DX0,DX1=896,1076; dmax=0.15; top=150; pitch=51; sh_h=38
    A(f'<text x="{GX0}" y="124" font-size="11.5" font-weight="700" fill="{INK}" letter-spacing="1.2">EACH REPLY, LAYER BY LAYER</text>')
    A(f'<text x="{SX0}" y="124" font-size="10.4" fill="{MUT}">share of attention by prompt part &#183; layer 1 at left, {nL} at right</text>')
    A(f'<text x="{DX0}" y="124" font-size="10.4" fill="{MUT}">question vs scene, mean over layers</text>')
    lx=lambda L:SX0+(SX1-SX0)*L/(nL-1); dxs=lambda v:DX0+(DX1-DX0)*min(v,dmax)/dmax
    for v in (0,0.05,0.10,0.15):
        A(f'<line x1="{dxs(v):.1f}" y1="{top-8}" x2="{dxs(v):.1f}" y2="{top+pitch*len(X)-10}" stroke="{FAINT}" stroke-width="0.6"/>')
        A(f'<text x="{dxs(v):.1f}" y="{top-11}" font-size="8.4" fill="{MUT}" text-anchor="middle" font-family="SF Mono, Menlo, monospace">{v*100:.0f}%</text>')
    for k,x in enumerate(X):
        y0=top+k*pitch; yb=y0+sh_h
        if k and any(X[k-1]["asst_i"]<c<x["user_i"] for c in resets):
            A(f'<line x1="{GX0}" y1="{y0-7}" x2="{DX1}" y2="{y0-7}" stroke="{INK}" stroke-width="0.8" stroke-dasharray="2 3"/>')
            A(f'<text x="{DX1}" y="{y0-10}" font-size="8.6" fill="{MUT}" text-anchor="end">context wiped</text>')
        q=U[x["user_i"]]["logged"] or U[x["user_i"]]["text"]; rp=U[x["asst_i"]]["logged"] or U[x["asst_i"]]["text"]
        cut=lambda s,n:(s[:n].rstrip()+"…") if len(s)>n+1 else s
        A(f'<text x="{GX0}" y="{y0+8}" font-size="8.4" fill="{MUT}" font-family="SF Mono, Menlo, monospace">{x["user_i"]}&#8594;{x["asst_i"]} &#183; {U[x["asst_i"]]["t"]:%H:%M}</text>')
        A(f'<text x="{GX0}" y="{y0+21}" font-size="9.4" fill="{INK}">{e(cut(q,40))}</text>')
        A(f'<text x="{GX0}" y="{y0+33}" font-size="9.2" fill="{MUT}" font-style="italic">{e(cut(rp,42))}</text>')
        S=sh[k]; cum=np.zeros(nL)
        for j,(key,label,c) in enumerate(STACK):
            lo=cum.copy(); cum=cum+S[:,j]
            up=" ".join(f"{lx(L):.1f},{yb-sh_h*cum[L]:.1f}" for L in range(nL))
            dn=" ".join(f"{lx(L):.1f},{yb-sh_h*lo[L]:.1f}" for L in reversed(range(nL)))
            A(f'<polygon points="{up} {dn}" fill="{c}" stroke="#ffffff" stroke-width="0.6"><title>{e(label)}: {S[:,j].mean()*100:.1f}% of attention, mean over layers</title></polygon>')
        A(f'<rect x="{SX0}" y="{y0}" width="{SX1-SX0}" height="{sh_h}" fill="none" stroke="{FAINT}" stroke-width="0.7"/>')
        qv,vv=S[:,0].mean(),S[:,1].mean(); yc=y0+sh_h/2
        A(f'<line x1="{dxs(min(qv,vv)):.1f}" y1="{yc}" x2="{dxs(max(qv,vv)):.1f}" y2="{yc}" stroke="{MUT}" stroke-width="1"/>')
        for v,c,lab in ((qv,col["question"],"question"),(vv,col["visual"],"scene")):
            A(f'<circle cx="{dxs(v):.1f}" cy="{yc}" r="4.5" fill="{c}" stroke="#ffffff" stroke-width="1.5"><title>{lab}: {v*100:.1f}%</title></circle>')
        A(f'<text x="{DX1+36}" y="{yc+3}" font-size="8.6" fill="{INK}" text-anchor="end" font-family="SF Mono, Menlo, monospace">{qv*100:.1f}</text>')
    A(f'<text x="{DX1+36}" y="{top-11}" font-size="8.4" fill="{MUT}" text-anchor="end" font-family="SF Mono, Menlo, monospace">Q %</text>')
    for L in (1,5,10,15,20,25,30,35):
        if L<=nL: A(f'<text x="{lx(L-1):.1f}" y="{top+pitch*len(X)-2}" font-size="8.4" fill="{MUT}" text-anchor="middle" font-family="SF Mono, Menlo, monospace">{L}</text>')

    # ---------------- rail: the average reply by depth ----------------
    A(f'<line x1="{RX0-24}" y1="132" x2="{RX0-24}" y2="1010" stroke="{FAINT}"/>')
    A(f'<text x="{RX0}" y="124" font-size="11.5" font-weight="700" fill="{INK}" letter-spacing="1.2">THE AVERAGE REPLY</text>')
    ax0,ax1,ay0,ay1=RX0+30,RX1-100,150,350; Sm=np.mean(sh,0)
    lxa=lambda L:ax0+(ax1-ax0)*L/(nL-1); cum=np.zeros(nL); tags=[]
    short={"question":"question","visual":"scene","history":"earlier turns","system":"persona",
           "reply":"own words","scaffold":"template","sink":"<bos>"}
    for j,(key,label,c) in enumerate(STACK):
        lo=cum.copy(); cum=cum+Sm[:,j]
        up=" ".join(f"{lxa(L):.1f},{ay1-(ay1-ay0)*cum[L]:.1f}" for L in range(nL))
        dn=" ".join(f"{lxa(L):.1f},{ay1-(ay1-ay0)*lo[L]:.1f}" for L in reversed(range(nL)))
        A(f'<polygon points="{up} {dn}" fill="{c}" stroke="#ffffff" stroke-width="1"/>')
        tags.append([ay1-(ay1-ay0)*(lo[-1]+cum[-1])/2, ay1-(ay1-ay0)*(lo[-1]+cum[-1])/2, f"{short[key]} {Sm[:,j].mean()*100:.0f}%"])
    for _ in range(50):                                          # direct labels at the right edge, pushed apart
        tags.sort(key=lambda t:t[1])
        for a_ in range(1,len(tags)):
            if tags[a_][1]-tags[a_-1][1]<11: d_=(11-(tags[a_][1]-tags[a_-1][1]))/2; tags[a_-1][1]-=d_; tags[a_][1]+=d_
    for band_y,y_,txt in tags:
        A(f'<polyline points="{ax1+2},{band_y:.1f} {ax1+6},{band_y:.1f} {ax1+9},{y_:.1f}" fill="none" stroke="{MUT}" stroke-width="0.7"/>')
        A(f'<text x="{ax1+12}" y="{y_+3:.1f}" font-size="8.8" fill="{INK}">{e(txt)}</text>')
    for v in (0,0.5,1.0):
        A(f'<text x="{ax0-5}" y="{ay1-(ay1-ay0)*v+3:.1f}" font-size="8.4" fill="{MUT}" text-anchor="end" font-family="SF Mono, Menlo, monospace">{v*100:.0f}%</text>')
    for L in (1,10,20,30,35):
        A(f'<text x="{lxa(L-1):.1f}" y="{ay1+12}" font-size="8.4" fill="{MUT}" text-anchor="middle" font-family="SF Mono, Menlo, monospace">{L}</text>')
    A(f'<text x="{(ax0+ax1)/2:.1f}" y="{ay1+25}" font-size="8.6" fill="{MUT}" text-anchor="middle">layer &#183; labels give the mean over layers</text>')

    # ---------------- rail: how far back each reply looked ----------------
    A(f'<text x="{RX0}" y="{ay1+62}" font-size="10.4" font-weight="700" fill="{INK}" letter-spacing="1.1">HOW FAR BACK EACH REPLY REACHED</text>')
    A(f'<text x="{RX0}" y="{ay1+77}" font-size="9.6" fill="{MUT}">arc width = attention to an earlier turn, mean over layers</text>')
    turns=sorted({x["user_i"] for x in X}|{x["asst_i"] for x in X}); ny0=ay1+98; step=(1000-ny0)/max(1,len(turns)-1)
    ypos={i:ny0+k*step for k,i in enumerate(turns)}; nx=RX0+36
    for k,x in enumerate(X):
        tgt=[(h["i"],float(np.mean(h["share"]))) for h in x["history"] if h["i"] is not None]
        tgt.append((x["user_i"],float(np.mean(np.array(x["shares"])[:,P.index("question")]))))
        for i,s in tgt:
            if s<0.004 or i not in ypos: continue
            y1_,y2_=ypos[i],ypos[x["asst_i"]]; rx=min(250,12+0.55*abs(y2_-y1_))
            c=HUM if U[i]["speaker"]=="HUMAN" else "#007699"
            A(f'<path d="M{nx},{y2_:.1f} C{nx+rx:.1f},{y2_:.1f} {nx+rx:.1f},{y1_:.1f} {nx},{y1_:.1f}" fill="none" stroke="{c}" stroke-width="{0.5+110*s:.2f}" stroke-opacity="0.42"><title>turn {x["asst_i"]} &#8594; turn {i}: {s*100:.1f}%</title></path>')
    for k in range(1,len(X)):
        if any(X[k-1]["asst_i"]<c<X[k]["user_i"] for c in resets):
            yy=(ypos[X[k-1]["asst_i"]]+ypos[X[k]["user_i"]])/2
            A(f'<line x1="{nx-20}" y1="{yy:.1f}" x2="{RX1}" y2="{yy:.1f}" stroke="{INK}" stroke-width="0.8" stroke-dasharray="2 3"/>')
    for i in turns:
        c=HUM if U[i]["speaker"]=="HUMAN" else "#007699"
        A(f'<circle cx="{nx}" cy="{ypos[i]:.1f}" r="3.6" fill="{c}"/>')
        A(f'<text x="{nx-8}" y="{ypos[i]+3:.1f}" font-size="8" fill="{MUT}" text-anchor="end" font-family="SF Mono, Menlo, monospace">{i}</text>')

    # ---------------- key + footer ----------------
    A(f'<line x1="62" y1="1042" x2="{RX1}" y2="1042" stroke="{FAINT}"/>')
    kx=62
    for key,label,c in STACK:
        A(f'<rect x="{kx}" y="1054" width="12" height="12" rx="2" fill="{c}"/>')
        A(f'<text x="{kx+17}" y="1064" font-size="9.8" fill="{INK}" fill-opacity="0.85">{e(label)}</text>')
        kx+=17+len(label)*5.6+22
    A(f'<circle cx="{kx+6}" cy="1060" r="4.5" fill="{HUM}"/><circle cx="{kx+18}" cy="1060" r="4.5" fill="{STACK[1][2]}"/>')
    A(f'<text x="{kx+28}" y="1064" font-size="9.8" fill="{INK}" fill-opacity="0.85">question / scene share (dot plot)</text>')
    foot=[f"Each reply was teacher-forced through google/gemma-4-E2B-it (bf16, eager attention) inside the prompt the dog received, rebuilt from the logs with the system prompt from the {e(sp_src)}.",
          "Shares are the reply tokens&#8217; attention weights summed over each part of the prompt, averaged over the reply&#8217;s tokens and the 8 heads; each layer sums to 100%. The &lt;bos&gt; sink and the template&#8217;s turn markers",
          "absorb attention that carries little content &#8212; a common transformer habit. Attention shows where information could flow, not proof of what the reply used."]
    for n,t in enumerate(foot): A(f'<text x="62" y="{1088+n*16}" font-size="10" fill="{MUT}">{t}</text>')
    A(f'<text x="62" y="{1088+3*16+8}" font-size="10" fill="{MUT}">Robert Twomey &#183; BFF / Dog Walk &#183; rendered {dt.date.today():%Y-%m-%d}</text>')
    A("</svg>")
    return o

def load_or_extract(a,cfg,U):
    """Use the cache unless a --system-prompt was asked for that the cache wasn't built with."""
    sp,src=ag.resolve_system_prompt(a.system_prompt,cfg)
    if os.path.exists(a.attn):
        r=json.load(open(a.attn))
        if a.system_prompt is None or r.get("system_prompt")==sp: return r
        print(f"{a.attn} was built with a different system prompt; replaying again")
    if not a.model: sys.exit(f"{a.attn} needs (re)building; pass --model to replay the session")
    ex,_=ag.build_replay(a.session_dir,cfg,U,system_prompt=sp)
    r=ag.extract_attention_surprise(ex,a.model); r.update(system_prompt=sp,system_prompt_source=src)
    json.dump(r,open(a.attn,"w")); return r

def main():
    ap=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_dir"); ap.add_argument("--attn", required=True, help="attention/surprise cache (.json)")
    ap.add_argument("--model", default=None); ap.add_argument("--system-prompt", default=None, metavar="SCRIPT_OR_TXT")
    ap.add_argument("-o","--out", default=None); ap.add_argument("--png", action="store_true")
    ap.add_argument("--png-width", type=int, default=3440)
    a=ap.parse_args()
    cfg,U,resets,_=cg.load(a.session_dir); r=load_or_extract(a,cfg,U)
    session=os.path.basename(a.session_dir.rstrip("/"))
    if "__" in session: session=[s for s in session.split("__") if s.startswith("session-")][0]
    out=a.out or f"{session}-attention.svg"
    open(out,"w").write("\n".join(render(U,r,resets,session,r.get("system_prompt_source","logged config.system_prompt"))))
    print(out)
    if a.png:
        png=os.path.splitext(out)[0]+".png"; print(f"{png}  ({a.png_width} px wide, via {cg.export_png(out,png,a.png_width)})")

if __name__=="__main__": main()
