#!/usr/bin/env python3
"""
surprise_ribbons.py — the conversation as text, every token shaded by how much it surprised the model.

Surprisal is -log2 p(token | everything before it) under google/gemma-4-E2B-it, replayed inside
the prompt the dog received: Robert's words as the model heard them (the live transcription), and
Snapper's words as the model produced them. The rail tracks surprisal turn by turn, lists the words
the model least expected (with its own guess), and shows which candidate system prompt best
explains what Snapper said.

usage:
  python3 surprise_ribbons.py SESSION_DIR --attn ATTN.json [--forensics F.json] [-o OUT.svg] [--png]
"""
import os
import sys
import json
import math
import argparse
import datetime as dt
import numpy as np
import conversation_graph as cg
import attention_flow as af

# sequential single hue (blue, light -> dark), stepped so a class reads at a glance on text
CLASSES=[(1,None),(3,"#cde2fb"),(6,"#9ec5f4"),(10,"#5598e7"),(20,"#256abf"),(1e9,"#0d366b")]
EDGES=["<1","1–3","3–6","6–10","10–20","20+"]
def shade(bits):
    for lim,c in CLASSES:
        if bits<lim: return c
    return CLASSES[-1][1]

def render(U,r,resets,session,forensics):
    from conversation_graph import W,H,GX0,GX1,RX0,RX1,INK,MUT,FAINT,BG,HUM,e
    DOG="#007699"; X=r["exchanges"]; o=[]; A=o.append
    t0,t1=U[0]["t"],U[-1]["t"]
    A(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Helvetica Neue, Helvetica, Arial, sans-serif">')
    A(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
    A(f'<text x="62" y="52" font-size="27" font-weight="700" fill="{INK}" letter-spacing="0.4">SNAPPER  &#215;  ROBERT</text>')
    A(f'<text x="62" y="76" font-size="12.5" fill="{MUT}">{e(session)} &#183; {t0:%Y-%m-%d}, {t0:%H:%M}&#8211;{t1:%H:%M} &#183; every word shaded by how much it surprised google/gemma-4-E2B-it</text>')
    A(f'<line x1="62" y1="96" x2="{RX1}" y2="96" stroke="{FAINT}"/>')

    # ---------------- the ribbons ----------------
    A(f'<text x="{GX0}" y="124" font-size="11.5" font-weight="700" fill="{INK}" letter-spacing="1.2">THE CONVERSATION, AS THE MODEL READ IT</text>')
    fs=11.2; cw=0.602*fs; TX=GX0+78; TMAX=GX1-8; lh=16.5; y=152
    inmodel={x["user_i"] for x in X}|{x["asst_i"] for x in X}
    byturn={x["user_i"]:("question",x) for x in X}; byturn.update({x["asst_i"]:("reply",x) for x in X})
    for u in U:
        i=u["i"]
        if i not in inmodel:                                   # never entered the model: shown, unshaded
            if u["type"]=="startup": continue
            A(f'<text x="{GX0}" y="{y+3}" font-size="8.4" fill="{MUT}" font-family="SF Mono, Menlo, monospace">{i}</text>')
            A(f'<text x="{TX}" y="{y+3}" font-size="9.2" fill="{MUT}" font-style="italic">{e(u["text"])} &#8212; never reached the model</text>')
            y+=lh+1
            if u["type"]=="cue":
                A(f'<line x1="{GX0}" y1="{y-6}" x2="{GX1}" y2="{y-6}" stroke="{INK}" stroke-width="0.8" stroke-dasharray="2 3"/>'); y+=5
            continue
        kind,x=byturn[i]; toks=x[kind]; who="ROBERT" if kind=="question" else "SNAPPER"
        A(f'<text x="{GX0}" y="{y+3}" font-size="8.4" fill="{MUT}" font-family="SF Mono, Menlo, monospace">{i}</text>')
        A(f'<text x="{GX0+20}" y="{y+3}" font-size="8.2" font-weight="700" fill="{INK}" letter-spacing="0.8">{who}</text>')
        A(f'<rect x="{TX-10}" y="{y-7}" width="3" height="{lh-3}" rx="1.5" fill="{HUM if kind=="question" else DOG}"/>')
        cx=TX
        for t in toks:
            w=len(t["t"])*cw
            if cx+w>TMAX and cx>TX:
                y+=lh; cx=TX; t=dict(t,t=t["t"].lstrip()); w=len(t["t"])*cw
            lead=(len(t["t"])-len(t["t"].lstrip()))*cw; c=shade(t["s"])
            tip=f'{e(t["t"].strip())}: {t["s"]:.1f} bits &#183; model expected {e(repr(t["top"]))} &#183; uncertainty {t["h"]:.1f} bits'
            if c: A(f'<rect x="{cx+lead:.1f}" y="{y-9.5:.1f}" width="{max(w-lead,cw*0.6):.1f}" height="13.5" rx="2" fill="{c}"><title>{tip}</title></rect>')
            dark=c in ("#256abf","#0d366b")
            A(f'<text x="{cx:.1f}" y="{y+1:.1f}" font-size="{fs}" fill="{"#ffffff" if dark else INK}" font-family="SF Mono, Menlo, monospace" xml:space="preserve"><title>{tip}</title>{e(t["t"])}</text>')
            cx+=w
        y+=lh+(7 if kind=="reply" else 1)

    # ---------------- rail: turn by turn ----------------
    A(f'<line x1="{RX0-24}" y1="132" x2="{RX0-24}" y2="1010" stroke="{FAINT}"/>')
    A(f'<text x="{RX0}" y="124" font-size="11.5" font-weight="700" fill="{INK}" letter-spacing="1.1">SURPRISE, TURN BY TURN</text>')
    cx0,cx1,cy0,cy1=RX0+30,RX1-70,150,330; ymax=20
    n=len(X); X_=lambda k:cx0+(cx1-cx0)*k/(n-1); Y_=lambda v:cy1-(cy1-cy0)*min(v,ymax)/ymax
    for v in (0,5,10,15,20):
        A(f'<line x1="{cx0}" y1="{Y_(v):.1f}" x2="{cx1}" y2="{Y_(v):.1f}" stroke="{"#c3c2b7" if v==0 else FAINT}" stroke-width="0.6"/>')
        A(f'<text x="{cx0-6}" y="{Y_(v)+3:.1f}" font-size="8.4" fill="{MUT}" text-anchor="end" font-family="SF Mono, Menlo, monospace">{v}</text>')
    A(f'<text x="{cx0-6}" y="{cy0-8}" font-size="8.4" fill="{MUT}" text-anchor="end">bits / token</text>')
    for k in range(1,n):
        if any(X[k-1]["asst_i"]<c<X[k]["user_i"] for c in resets):
            xx=(X_(k-1)+X_(k))/2; A(f'<line x1="{xx:.1f}" y1="{cy0}" x2="{xx:.1f}" y2="{cy1}" stroke="{INK}" stroke-width="0.8" stroke-dasharray="2 3"/>')
    for kind,colr,label in (("question",HUM,"Robert"),("reply",DOG,"Snapper")):
        v=[float(np.mean([t["s"] for t in x[kind]])) for x in X]
        A(f'<polyline points="{" ".join(f"{X_(k):.1f},{Y_(m):.1f}" for k,m in enumerate(v))}" fill="none" stroke="{colr}" stroke-width="2" stroke-linejoin="round"/>')
        for k,m in enumerate(v):
            A(f'<circle cx="{X_(k):.1f}" cy="{Y_(m):.1f}" r="4" fill="{colr}" stroke="#ffffff" stroke-width="1.5"><title>{label}, turn {X[k]["user_i" if kind=="question" else "asst_i"]}: {m:.1f} bits/token</title></circle>')
        A(f'<text x="{cx1+10}" y="{Y_(v[-1])+3:.1f}" font-size="9.4" fill="{INK}">{label} {np.mean(v):.1f}</text>')
    for k in (0,n-1):
        A(f'<text x="{X_(k):.1f}" y="{cy1+13}" font-size="8.4" fill="{MUT}" text-anchor="middle" font-family="SF Mono, Menlo, monospace">{X[k]["user_i"]}&#8594;{X[k]["asst_i"]}</text>')
    A(f'<text x="{(cx0+cx1)/2:.1f}" y="{cy1+13}" font-size="8.4" fill="{MUT}" text-anchor="middle">exchange, in time order</text>')

    # the words the model least expected
    yy=cy1+50
    A(f'<text x="{RX0}" y="{yy}" font-size="10.4" font-weight="700" fill="{INK}" letter-spacing="1.1">WHAT IT DIDN&#8217;T SEE COMING</text>'); yy+=18
    for kind,label in (("question","Robert"),("reply","Snapper")):
        top=sorted(((t,x) for x in X for t in x[kind]),key=lambda p:-p[0]["s"])[:4]
        A(f'<text x="{RX0}" y="{yy}" font-size="9.4" font-weight="700" fill="{INK}">{label}</text>'); yy+=14
        for t,x in top:
            A(f'<text x="{RX0+8}" y="{yy}" font-size="9.2" fill="{INK}" font-family="SF Mono, Menlo, monospace">{t["s"]:4.1f}b</text>')
            A(f'<text x="{RX0+54}" y="{yy}" font-size="9.2" fill="{INK}">&#8220;{e(t["t"].strip())}&#8221; <tspan fill="{MUT}">where it expected</tspan> &#8220;{e(t["top"].strip() or t["top"])}&#8221;</text>'); yy+=13
        yy+=6

    # which system prompt explains Snapper's words
    if forensics:
        yy+=8
        A(f'<text x="{RX0}" y="{yy}" font-size="10.4" font-weight="700" fill="{INK}" letter-spacing="1.1">WHICH PROMPT WAS SNAPPER RUNNING?</text>'); yy+=15
        A(f'<text x="{RX0}" y="{yy}" font-size="9.4" fill="{MUT}">surprise of Snapper&#8217;s 15 replies under each candidate system prompt</text>'); yy+=14
        cands=forensics["candidates"]; sc=forensics["scores"]
        means=[float(np.mean([m for m,_ in sc[c]])) for c in cands]; bad=[sum(k for _,k in sc[c]) for c in cands]
        bx0,bx1=RX0+170,RX1-60; vmax=max(means)*1.08
        for c,m,b in zip(cands,means,bad):
            chosen=c==r.get("system_prompt_source")
            lab=c.replace(" Default scene","").replace("performance-script","script").replace("logged config.system_prompt","logged config prompt")
            A(f'<text x="{RX0}" y="{yy+9}" font-size="9.2" fill="{INK}" font-weight="{700 if chosen else 400}">{e(lab)}</text>')
            A(f'<rect x="{bx0}" y="{yy}" width="{(bx1-bx0)*m/vmax:.1f}" height="11" rx="2" fill="{INK if chosen else "#c3c2b7"}"/>')
            A(f'<text x="{bx0+(bx1-bx0)*m/vmax+5:.1f}" y="{yy+9}" font-size="8.8" fill="{INK}" font-family="SF Mono, Menlo, monospace">{m:.2f} &#183; {b} &gt;20b</text>')
            yy+=17
        A(f'<text x="{RX0}" y="{yy+6}" font-size="9.2" fill="{MUT}">bits/token &#183; tokens above 20 bits (near-impossible to sample). The logs record only the</text>')
        A(f'<text x="{RX0}" y="{yy+19}" font-size="9.2" fill="{MUT}">config prompt; chat-manager swaps in a script&#8217;s Default scene. Replay uses the dark bar.</text>')

    # legend + footer
    A(f'<line x1="62" y1="1042" x2="{RX1}" y2="1042" stroke="{FAINT}"/>')
    A(f'<text x="62" y="1064" font-size="9.8" font-weight="700" fill="{INK}">bits of surprise</text>'); kx=160
    for (lim,c),lab in zip(CLASSES,EDGES):
        A(f'<rect x="{kx}" y="1054" width="38" height="13" rx="2" fill="{c or "#ffffff"}" stroke="{FAINT}" stroke-width="0.6"/>')
        A(f'<text x="{kx+19}" y="1064" font-size="8.8" fill="{"#ffffff" if c in ("#256abf","#0d366b") else INK}" text-anchor="middle" font-family="SF Mono, Menlo, monospace">{e(lab)}</text>'); kx+=42
    A(f'<text x="{kx+10}" y="1064" font-size="9.4" fill="{MUT}">each bit halves the probability: 10 bits is 1 in 1,024, 20 bits 1 in a million. Hover a word for the model&#8217;s own guess.</text>')
    foot=[f"Surprise is &#8722;log&#8322; p(token | everything before it) under google/gemma-4-E2B-it (bf16), each reply replayed inside the prompt the dog received, system prompt from the {e(r.get('system_prompt_source','logged config'))}.",
          "Robert&#8217;s words are the live whisper transcription the model actually got &#8212; lowercase, unpunctuated, sometimes garbled &#8212; so part of their surprise is the transcription. Snapper&#8217;s words were sampled",
          "(temperature 0.7) from the Q4_K_M quantization of these weights, which is why most of them sit near zero. Lines in italics never entered the model: reset requests are intercepted, cues are canned audio."]
    for k,t in enumerate(foot): A(f'<text x="62" y="{1088+k*16}" font-size="10" fill="{MUT}">{t}</text>')
    A(f'<text x="62" y="{1088+3*16+8}" font-size="10" fill="{MUT}">Robert Twomey &#183; BFF / Dog Walk &#183; rendered {dt.date.today():%Y-%m-%d}</text>')
    A("</svg>")
    return o, y

def main():
    ap=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_dir"); ap.add_argument("--attn", required=True, help="attention/surprise cache (.json)")
    ap.add_argument("--forensics", default=None, help="candidate-prompt scores (.json), shown in the rail if given")
    ap.add_argument("--model", default=None); ap.add_argument("--system-prompt", default=None, metavar="SCRIPT_OR_TXT")
    ap.add_argument("-o","--out", default=None); ap.add_argument("--png", action="store_true")
    ap.add_argument("--png-width", type=int, default=3440)
    a=ap.parse_args()
    cfg,U,resets,_=cg.load(a.session_dir); r=af.load_or_extract(a,cfg,U)
    fz=json.load(open(a.forensics)) if a.forensics and os.path.exists(a.forensics) else None
    session=os.path.basename(a.session_dir.rstrip("/"))
    if "__" in session: session=[s for s in session.split("__") if s.startswith("session-")][0]
    out=a.out or f"{session}-surprise.svg"
    o,ybottom=render(U,r,resets,session,fz)
    open(out,"w").write("\n".join(o)); print(f"{out}  (ribbons end at y={ybottom:.0f})")
    if a.png:
        png=os.path.splitext(out)[0]+".png"; print(f"{png}  ({a.png_width} px wide, via {cg.export_png(out,png,a.png_width)})")

if __name__=="__main__": main()
