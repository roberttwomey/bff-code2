#!/usr/bin/env python3
"""
dialogue_poster.py — one large sheet (22 x 30 in): the whole dialogue in Gemma 4's semantic space.

Every utterance is a card holding its complete text, exactly as the model received or produced it,
each token shaded by its surprisal. Cards sit at the utterance's position in the model's space
(or, with --space tfidf, in the lexical space of conversation_graph.py), nudged apart only enough
not to overlap. Arrows carry attention: what each of the dog's replies drew from the question and
from earlier turns, width = share of the reply's attention averaged over all layers and heads.
Each reply card also shows its whole attention budget and the scene text the camera fed it.

usage:
  python3 dialogue_poster.py SESSION_DIR --acts ACTS.npz --attn ATTN.json
                             [--space activation|tfidf] [--layer 18] [-o OUT.svg] [--png [--dpi 150]]
"""
import os
import sys
import json
import math
import argparse
import textwrap
import datetime as dt
import numpy as np
import conversation_graph as cg
import activation_graph as ag
import attention_flow as af
import surprise_ribbons as sr

PW,PH=22*72,30*72                                   # the sheet, in points
MONO="SF Mono, Menlo, monospace"; SANS="Helvetica Neue, Helvetica, Arial, sans-serif"
INK,MUT,FAINT,BG="#1c1917","#78716c","#dcd8d2","#faf8f5"
HUM,DOG,VIS,RST="#c2410c","#007699","#7c3aed","#b91c1c"
CARD_W=250; PAD=6; FS=6.8; CWID=0.602*FS; LH=9.2; SFS=5.8; SCW=0.53*SFS; SLH=7.3
FX0,FX1,FY0,FY1=54,PW-54,222,1880                   # the field the cards live in

def e(s): return cg.e(s)

def wrap_tokens(toks,width):
    lines=[[]]; x=0.0
    for t in toks:
        txt=t["t"]; w=len(txt)*CWID
        if x+w>width and lines[-1]:
            lines.append([]); x=0.0; txt=txt.lstrip(); w=len(txt)*CWID
        lines[-1].append((x,dict(t,t=txt))); x+=w
    return lines

def build_cards(U,r,ex,dog):
    X=r["exchanges"]; P=r["parts"]; cards={}
    for k,x in enumerate(X):
        for kind in ("question","reply"):
            i=x["user_i"] if kind=="question" else x["asst_i"]
            lines=wrap_tokens(x[kind],CARD_W-2*PAD)
            c=dict(i=i,kind=kind,who=("ROBERT" if kind=="question" else dog),lines=lines,
                   s=float(np.mean([t["s"] for t in x[kind]])),t=U[i]["t"],dur=U[i]["dur"])
            h=PAD+10+len(lines)*LH+PAD-1
            if kind=="reply":
                sh=np.array(x["shares"]).mean(0)
                c["budget"]={p:float(sh[P.index(p)]) for p in P}
                scene=ex[k]["vis"] or ""
                c["scene"]=textwrap.wrap("saw: "+scene,width=int((CARD_W-2*PAD)/SCW)) if scene else []
                h+=4+5+3+len(c["scene"])*SLH+2
            c["w"],c["h"]=CARD_W,h; cards[i]=c
    return cards

def layout(cards,order,P):
    """Card centres start at the semantic positions; overlaps are resolved by pushing along the axis
    of least overlap, with a weakening spring back toward the true position."""
    n=len(order); Wd=np.array([cards[i]["w"] for i in order]); Hd=np.array([cards[i]["h"] for i in order])
    lo,hi=P.min(0),P.max(0); q=(P-lo)/np.maximum(hi-lo,1e-9); mw,mh=Wd.max(),Hd.max()
    tx=FX0+mw/2+q[:,0]*((FX1-FX0)-mw); ty=FY1-mh/2-q[:,1]*((FY1-FY0)-mh)
    T=np.stack([tx,ty],1); C=T.copy(); g=9.0
    for it in range(3000):
        moved=0
        for a in range(n):
            for b in range(a+1,n):
                dx,dy=C[b,0]-C[a,0],C[b,1]-C[a,1]
                ox=(Wd[a]+Wd[b])/2+g-abs(dx); oy=(Hd[a]+Hd[b])/2+g-abs(dy)
                if ox>0 and oy>0:
                    moved+=1
                    if ox<oy: s_=1.0 if dx>=0 else -1.0; C[a,0]-=s_*ox/2; C[b,0]+=s_*ox/2
                    else:     s_=1.0 if dy>=0 else -1.0; C[a,1]-=s_*oy/2; C[b,1]+=s_*oy/2
        if it<1500: C+=0.04*(1-it/1500)*(T-C)
        C[:,0]=np.clip(C[:,0],FX0+Wd/2,FX1-Wd/2); C[:,1]=np.clip(C[:,1],FY0+Hd/2,FY1-Hd/2)
        if moved==0 and it>=1500: break
    for it in range(4000):                       # whatever the edges pinned sideways, separate vertically
        moved=0
        for a in range(n):
            for b in range(a+1,n):
                dx,dy=C[b,0]-C[a,0],C[b,1]-C[a,1]
                ox=(Wd[a]+Wd[b])/2+g-abs(dx); oy=(Hd[a]+Hd[b])/2+g-abs(dy)
                if ox>0 and oy>0:
                    moved+=1; s_=1.0 if dy>=0 else -1.0; C[a,1]-=s_*oy/2; C[b,1]+=s_*oy/2
        C[:,1]=np.clip(C[:,1],FY0+Hd/2,FY1-Hd/2)
        if moved==0: break
    return {i:(C[k],T[k]) for k,i in enumerate(order)}, moved

def edge_pt(c,w,h,t):
    dx,dy=t[0]-c[0],t[1]-c[1]
    if abs(dx)<1e-9 and abs(dy)<1e-9: return c[0],c[1]
    s=min((w/2)/abs(dx) if dx else 1e9,(h/2)/abs(dy) if dy else 1e9)
    return c[0]+dx*s,c[1]+dy*s

def render(cfg,U,resets,r,ex,cards,pos,space_label,dog):
    o=[]; A=o.append; X=r["exchanges"]; order=sorted(cards)
    A(f'<svg xmlns="http://www.w3.org/2000/svg" width="22in" height="30in" viewBox="0 0 {PW} {PH}" font-family="{SANS}">')
    A(f'<rect width="{PW}" height="{PH}" fill="{BG}"/><defs>')
    for nm,c in (("ah",HUM),("ad",DOG)):
        A(f'<marker id="{nm}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" markerUnits="userSpaceOnUse" orient="auto">'
          f'<path d="M0,1 L9,5 L0,9 z" fill="{c}"/></marker>')
    A(f'<marker id="as" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="4" markerHeight="4" markerUnits="userSpaceOnUse" orient="auto"><path d="M0,1 L9,5 L0,9 z" fill="#8a847d"/></marker>')
    A(f'<marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="4" markerHeight="4" markerUnits="userSpaceOnUse" orient="auto"><path d="M0,1 L9,5 L0,9 z" fill="{RST}"/></marker></defs>')
    t0,t1=U[0]["t"],U[-1]["t"]
    A(f'<text x="{FX0}" y="78" font-size="40" font-weight="700" fill="{INK}" letter-spacing="0.6">{e(dog)}  &#215;  ROBERT</text>')
    A(f'<text x="{FX0}" y="104" font-size="12" fill="{MUT}">{e(cfg.get("session_id",""))} &#183; {t0:%Y-%m-%d}, {t0:%H:%M}&#8211;{t1:%H:%M} &#183; {len(order)} utterances replayed through google/gemma-4-E2B-it &#183; {e(space_label)}</text>')
    how=["Each card is one utterance, placed where the model represents it: nearby cards are ones it treats as alike. The text is exactly what the model received (Robert, live",
         "transcription) or produced (the dog), every token shaded by how much it surprised the model. Arrows carry attention into each reply from Robert’s question and from",
         "earlier turns; width is the share of the reply’s attention, averaged over all 35 layers and 8 heads. Those arrows are a minority of where attention goes: the bar in each",
         "reply card is its whole attention budget, including the persona prompt, the camera’s scene text (quoted beneath it), the reply’s own words, and the chat template."]
    for k,t in enumerate(how): A(f'<text x="{FX0}" y="{136+k*15}" font-size="10.4" fill="{INK}" fill-opacity="0.85">{e(t)}</text>')
    A(f'<line x1="{FX0}" y1="{FY0-22}" x2="{FX1}" y2="{FY0-22}" stroke="{FAINT}"/>')

    # time spine, underneath everything
    for a in range(len(order)-1):
        i,j=order[a],order[a+1]; (ci,_),(cj,_)=pos[i],pos[j]
        p1=edge_pt(ci,cards[i]["w"],cards[i]["h"],cj); p2=edge_pt(cj,cards[j]["w"],cards[j]["h"],ci)
        wipe=any(i<c<j for c in resets)
        A(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" stroke="{RST if wipe else "#8a847d"}" stroke-width="{1.1 if wipe else 0.7}" stroke-opacity="0.75" stroke-dasharray="{"2 4" if wipe else "1.2 2.4"}" marker-end="url(#{"ar" if wipe else "as"})"/>')
    # attention arrows: from what was attended to, into the reply
    arrows=[]
    for x in X:
        R=x["asst_i"]; sh=np.array(x["shares"]); P_=r["parts"]
        arrows.append((x["user_i"],R,float(sh[:,P_.index("question")].mean())))
        for h in x["history"]:
            if h["i"] is not None: arrows.append((h["i"],R,float(np.mean(h["share"]))))
    arrows.sort(key=lambda a:a[2])
    for src,dst,s in arrows:
        if s<0.004 or src not in pos or dst not in pos: continue
        (cs,_),(cd,_)=pos[src],pos[dst]
        p1=edge_pt(cs,cards[src]["w"],cards[src]["h"],cd); p2=edge_pt(cd,cards[dst]["w"],cards[dst]["h"],cs)
        dx,dy=p2[0]-p1[0],p2[1]-p1[1]; L=math.hypot(dx,dy) or 1
        cx,cy=(p1[0]+p2[0])/2-dy*0.14,(p1[1]+p2[1])/2+dx*0.14
        col,mk=(HUM,"ah") if U[src]["speaker"]=="HUMAN" else (DOG,"ad")
        A(f'<path d="M{p1[0]:.1f},{p1[1]:.1f} Q{cx:.1f},{cy:.1f} {p2[0]:.1f},{p2[1]:.1f}" fill="none" stroke="{col}" stroke-width="{0.35+42*s:.2f}" stroke-opacity="0.5" marker-end="url(#{mk})"><title>turn {dst} drew {s*100:.1f}% of its attention from turn {src}</title></path>')
    # anchors: where each utterance truly sits, tied to its card when the card had to move
    for i in order:
        c,t=pos[i]
        if math.hypot(*(c-t))>14:
            A(f'<line x1="{t[0]:.1f}" y1="{t[1]:.1f}" x2="{c[0]:.1f}" y2="{c[1]:.1f}" stroke="{MUT}" stroke-width="0.4" stroke-opacity="0.7"/>')
        A(f'<circle cx="{t[0]:.1f}" cy="{t[1]:.1f}" r="1.8" fill="{HUM if cards[i]["kind"]=="question" else DOG}"/>')
    # the cards
    for i in order:
        cd=cards[i]; (c,_)=pos[i]; x0,y0=c[0]-cd["w"]/2,c[1]-cd["h"]/2; sc=HUM if cd["kind"]=="question" else DOG
        full=U[i]["text"]
        A(f'<g><title>turn {i} · {cd["who"]} · heard/said as: {e(full)}</title>')
        A(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{cd["w"]}" height="{cd["h"]:.1f}" rx="3" fill="#ffffff" stroke="{FAINT}" stroke-width="0.7"/>')
        A(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="2.6" height="{cd["h"]:.1f}" rx="1.3" fill="{sc}"/>')
        hx,hy=x0+PAD+1,y0+PAD+5.2
        A(f'<text x="{hx:.1f}" y="{hy:.1f}" font-size="6.2" fill="{INK}"><tspan font-family="{MONO}" font-weight="700">{i}</tspan><tspan dx="4" font-weight="700" letter-spacing="0.6">{e(cd["who"])}</tspan><tspan dx="4" fill="{MUT}" font-family="{MONO}">{cd["t"]:%H:%M:%S} &#183; {cd["dur"] or 0:.1f}s &#183; {cd["s"]:.1f} bits/token</tspan></text>')
        yy=hy+LH+1.2
        for line in cd["lines"]:
            for off,t in line:
                w=len(t["t"])*CWID; lead=(len(t["t"])-len(t["t"].lstrip()))*CWID; col=sr.shade(t["s"])
                tip=f'{e(t["t"].strip())}: {t["s"]:.1f} bits · model expected {e(repr(t["top"]))}'
                if col: A(f'<rect x="{hx+off+lead:.2f}" y="{yy-5.6:.2f}" width="{max(w-lead,CWID*0.6):.2f}" height="7.6" rx="1" fill="{col}"><title>{tip}</title></rect>')
                dark=col in ("#256abf","#0d366b")
                A(f'<text x="{hx+off:.2f}" y="{yy:.2f}" font-size="{FS}" font-family="{MONO}" fill="{"#ffffff" if dark else INK}" xml:space="preserve"><title>{tip}</title>{e(t["t"])}</text>')
            yy+=LH
        if cd["kind"]=="reply":
            bx=hx; bw=cd["w"]-2*PAD-1; yb=yy-3.5; acc=0.0
            for key,label,colr in af.STACK:
                v=cd["budget"][key]
                A(f'<rect x="{bx+acc*bw:.2f}" y="{yb:.2f}" width="{max(v*bw-0.5,0.2):.2f}" height="4" fill="{colr}"><title>{e(label)}: {v*100:.1f}%</title></rect>')
                acc+=v
            A(f'<text x="{bx:.1f}" y="{yb+9.6:.1f}" font-size="5.2" fill="{MUT}">attention &#8212; question {cd["budget"]["question"]*100:.1f}% &#183; scene {cd["budget"]["visual"]*100:.1f}% &#183; earlier turns {cd["budget"]["history"]*100:.1f}% &#183; persona {cd["budget"]["system"]*100:.1f}% &#183; own words {cd["budget"]["reply"]*100:.1f}%</text>')
            ys=yb+9.6+SLH
            for ln in cd["scene"]:
                A(f'<text x="{bx:.1f}" y="{ys:.1f}" font-size="{SFS}" fill="{VIS}" font-style="italic">{e(ln)}</text>'); ys+=SLH
        A('</g>')

    # ---------------- bottom band ----------------
    BY=FY1+34
    A(f'<line x1="{FX0}" y1="{BY-16}" x2="{FX1}" y2="{BY-16}" stroke="{FAINT}"/>')
    # legend
    A(f'<text x="{FX0}" y="{BY}" font-size="9" font-weight="700" fill="{INK}" letter-spacing="1">READING THE SHEET</text>')
    ly=BY+18; A(f'<text x="{FX0}" y="{ly}" font-size="7.4" fill="{INK}">token surprise (bits)</text>'); kx=FX0+86
    for (lim,c),lab in zip(sr.CLASSES,sr.EDGES):
        A(f'<rect x="{kx}" y="{ly-7.5}" width="30" height="10" rx="1.5" fill="{c or "#ffffff"}" stroke="{FAINT}" stroke-width="0.5"/>')
        A(f'<text x="{kx+15}" y="{ly}" font-size="6.6" text-anchor="middle" font-family="{MONO}" fill="{"#ffffff" if c in ("#256abf","#0d366b") else INK}">{e(lab)}</text>'); kx+=33
    ly+=20; A(f'<text x="{FX0}" y="{ly}" font-size="7.4" fill="{INK}">attention arrows</text>')
    for k,(s,lab) in enumerate(((0.01,"1%"),(0.05,"5%"),(0.10,"10%"))):
        x_=FX0+86+k*62
        A(f'<line x1="{x_}" y1="{ly-3}" x2="{x_+36}" y2="{ly-3}" stroke="{INK}" stroke-width="{0.35+42*s:.2f}" stroke-opacity="0.55"/>')
        A(f'<text x="{x_+40}" y="{ly}" font-size="6.8" fill="{INK}" font-family="{MONO}">{lab}</text>')
    ly+=16
    for k,(c,txt) in enumerate(((HUM,"drawn from Robert’s words"),(DOG,"drawn from the dog’s own earlier replies"))):
        x_=FX0+86+k*150; A(f'<line x1="{x_}" y1="{ly-3}" x2="{x_+22}" y2="{ly-3}" stroke="{c}" stroke-width="2.2" stroke-opacity="0.6"/>')
        A(f'<text x="{x_+27}" y="{ly}" font-size="6.8" fill="{INK}">{e(txt)}</text>')
    ly+=18; A(f'<text x="{FX0}" y="{ly}" font-size="7.4" fill="{INK}">attention budget</text>')
    for n,(key,label,c) in enumerate(af.STACK):                 # two rows, so it stays inside its column
        kx=FX0+86+(n%4)*108; ky=ly+(n//4)*13
        A(f'<rect x="{kx}" y="{ky-6.5}" width="8" height="8" rx="1.5" fill="{c}"/>'); A(f'<text x="{kx+11}" y="{ky}" font-size="6.8" fill="{INK}">{e(label)}</text>')
    ly+=31
    A(f'<line x1="{FX0+86}" y1="{ly-3}" x2="{FX0+110}" y2="{ly-3}" stroke="#8a847d" stroke-width="0.7" stroke-dasharray="1.2 2.4"/>')
    A(f'<text x="{FX0+115}" y="{ly}" font-size="6.8" fill="{INK}">order in time</text>')
    A(f'<line x1="{FX0+180}" y1="{ly-3}" x2="{FX0+204}" y2="{ly-3}" stroke="{RST}" stroke-width="1.1" stroke-dasharray="2 4"/>')
    A(f'<text x="{FX0+209}" y="{ly}" font-size="6.8" fill="{INK}">context wiped (“let’s start over”)</text>')
    A(f'<circle cx="{FX0+352}" cy="{ly-3}" r="1.8" fill="{DOG}"/><text x="{FX0+358}" y="{ly}" font-size="6.8" fill="{INK}">true position, when a card had to move</text>')
    # persona prompt
    PX=FX0+540
    A(f'<text x="{PX}" y="{BY}" font-size="9" font-weight="700" fill="{INK}" letter-spacing="1">THE PERSONA PROMPT</text>')
    A(f'<text x="{PX}" y="{BY+13}" font-size="6.6" fill="{MUT}">inferred: the {e(r.get("system_prompt_source",""))} (the logs record only a shorter config prompt, which a performance-script Default scene replaces)</text>')
    yy=BY+26
    for para in r.get("system_prompt","").split("\n"):
        for ln in (textwrap.wrap(para,width=128) or [""]):
            A(f'<text x="{PX}" y="{yy}" font-size="6.2" fill="{INK}" fill-opacity="0.85">{e(ln)}</text>'); yy+=7.6
    # never entered the model + method
    OX=FX1-300
    A(f'<text x="{OX}" y="{BY}" font-size="9" font-weight="700" fill="{INK}" letter-spacing="1">NEVER ENTERED THE MODEL</text>'); yy=BY+15
    for u in U:
        if u["i"] in cards: continue
        A(f'<text x="{OX}" y="{yy}" font-size="6.6" fill="{MUT}"><tspan font-family="{MONO}">{u["i"]:>2}</tspan>  {e(u["text"])}</text>'); yy+=9
    yy+=8
    notes=["Replayed from the session logs: each prompt rebuilt as chat-manager.py",
           "assembled it, the logged reply teacher-forced through gemma-4-E2B-it",
           "(bf16). The dog ran the Q4_K_M quantization via Ollama; body state was",
           "injected live but never logged, so it is absent here (a representative",
           "packet moves replies by at most 11% of the reply-to-reply distance).",
           "Attention shows where information could flow, not proof of use."]
    for t in notes: A(f'<text x="{OX}" y="{yy}" font-size="6.6" fill="{MUT}">{e(t)}</text>'); yy+=8.4
    A(f'<text x="{FX0}" y="{PH-30}" font-size="8" fill="{MUT}">Robert Twomey &#183; BFF / Dog Walk &#183; rendered {dt.date.today():%Y-%m-%d}</text>')
    A("</svg>")
    return o

def main():
    ap=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_dir"); ap.add_argument("--acts", required=True); ap.add_argument("--attn", required=True)
    ap.add_argument("--space", choices=("activation","tfidf"), default="activation")
    ap.add_argument("--layer", type=int, default=18, help="depth for --space activation (default 18)")
    ap.add_argument("-o","--out", default=None); ap.add_argument("--png", action="store_true")
    ap.add_argument("--dpi", type=int, default=150, help="PNG resolution (default 150 dpi -> 3300 px wide)")
    a=ap.parse_args()
    cfg,U,resets,_=cg.load(a.session_dir)
    rows=[json.loads(l) for l in open(os.path.join(a.session_dir,"transcript","session-complete.jsonl")) if l.strip()] \
         if os.path.exists(os.path.join(a.session_dir,"transcript","session-complete.jsonl")) else []
    sid=next((r_.get("session_id") for r_ in rows if r_.get("type")=="session_start"),None); cfg=dict(cfg,session_id=sid or "")
    r=json.load(open(a.attn)); z=np.load(a.acts); dog=cfg.get("speaker") or "SNAPPER"
    if ("system_prompt" in z.files) and str(z["system_prompt"])!=r.get("system_prompt"):
        sys.exit("the activation and attention caches were built with different system prompts")
    ex,_=ag.build_replay(a.session_dir,cfg,U,system_prompt=r.get("system_prompt"))
    if [x["user_i"] for x in ex]!=[x["user_i"] for x in r["exchanges"]]: sys.exit("attention cache is from a different replay")
    cards=build_cards(U,r,ex,dog); order=sorted(cards)
    if a.space=="activation":
        res=ag.analyse(z,U,max(0,min(a.layer,z["user"].shape[1]-1)),[])
        idx={i:k for k,i in enumerate(res["order"])}; P=np.stack([res["P"][idx[i]] for i in order])
        space_label=f"placed by hidden state at layer {a.layer} of 35 (z-scored, cosine, MDS)"
    else:
        pseudo=[dict(i=k,text=U[i]["logged"] or U[i]["text"],block=U[i]["block"],speaker=U[i]["speaker"],type=U[i]["type"]) for k,i in enumerate(order)]
        P,_,_=cg.embed(pseudo,1.01); space_label="placed by word overlap (TF-IDF, LSA, MDS)"
    pos,left=layout(cards,order,P)
    out=a.out or f"{sid or 'session'}-poster.svg"
    open(out,"w").write("\n".join(render(cfg,U,resets,r,ex,cards,pos,space_label,dog)))
    disp=[math.hypot(*(pos[i][0]-pos[i][1])) for i in order]
    touching=sum(1 for a in range(len(order)) for b in range(a+1,len(order))
                 if (cards[order[a]]["w"]+cards[order[b]]["w"])/2>abs(pos[order[b]][0][0]-pos[order[a]][0][0])
                 and (cards[order[a]]["h"]+cards[order[b]]["h"])/2>abs(pos[order[b]][0][1]-pos[order[a]][0][1]))
    print(f"{out}  ({len(order)} cards, {touching} overlapping, {left} closer than the 9 pt gap; "
          f"median card shift {np.median(disp):.0f} pt, max {max(disp):.0f} pt)")
    if a.png:
        png=os.path.splitext(out)[0]+".png"; print(f"{png}  ({22*a.dpi} px wide, via {cg.export_png(out,png,22*a.dpi)})")

if __name__=="__main__": main()
