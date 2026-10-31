#!/usr/bin/env python3
"""
plotter_drawing.py — the dialogue as a black-ink pen-plotter drawing.

Takes its manner from the housemachine trail drawings: one thin pen, no fills or
boxes, wandering trails that knot where someone lingers, density built up from
passes of the line, a form floating in white paper, and single-stroke Hershey labels
and title. Every mark is measured:

  circle       an utterance, at its place in the model's semantic space, sized by how
               long it was spoken, with its full text set in Hershey beside it. Everything in
               the dog's voice is a double ring, its canned lines included; Robert's are single
  trail        one continuous pen line through the conversation, with turn numbers
               and arrowheads for the sequence (--no-sequence removes them)
  dashed arcs  jumps to and from the utterances that never entered the model, which
               sit in a small stack outside the form, each arc ending at its own circle

Three optional layers build it toward the density of the housemachine sheets:

  --surprise   the trail wanders more on tokens that surprised the model
  --knots      scribbled knots replace the circles, sized and packed by spoken duration
  --attention  one strand per --strand-pct of a reply's attention, from Robert's
               question or an earlier turn into the reply

Labels hold the whole utterance by default; --excerpt gives its opening words instead,
and --speaker just the speaker's name. Text is placed clear of the trail where it can be.

A graph from multi_session.py draws several recordings in one space: turns are lettered (A12, B3),
and each recording has its own stack. --session-ticks marks each recording's trail with 0, 1, 2 or
3 cross-strokes, as the key then shows.

Black only, so the Hershey labels carry who is speaking. Pen trails are seeded, so a
drawing is repeatable. Reads the JSON that graph3d.py writes and the attention cache,
and needs only numpy and HersheyFonts; the housemachine conda env has both:

  /opt/homebrew/anaconda3/envs/housemachine/bin/python3 post_processing/plotter_drawing.py \\
      --graph outputs/snapper-graph3d.json --attn outputs/snapper-attention.json \\
      -o outputs/snapper-plotter.svg --png

Use graph3d.py --viewer to find a view, then pass the same --azimuth --elevation --camera.
Layers: layer_attention, layer_trail, layer_jumps, layer_sequence, layer_labels, layer_title. The output is
in millimetres of paper. Check it before plotting with send_svg_axidraw.py --inspect --overdraw.
"""
import os
import re
import sys
import json
import math
import shutil
import argparse
import subprocess
import textwrap
import zlib
import numpy as np

try:
    from HersheyFonts import HersheyFonts
except ImportError:
    sys.exit("needs HersheyFonts: run with /opt/homebrew/anaconda3/envs/housemachine/bin/python3")

CAP_HEIGHT=21.0      # Hershey glyph box, as in housemachine plotting/hershey_title.py
BASELINE=9.0
MM=25.4

def project(p,az,el,cam):
    """Same camera as graph3d.project."""
    a,b=math.radians(az),math.radians(el)
    x1=p[0]*math.cos(a)+p[2]*math.sin(a); z1=-p[0]*math.sin(a)+p[2]*math.cos(a)
    y2=p[1]*math.cos(b)-z1*math.sin(b); z2=p[1]*math.sin(b)+z1*math.cos(b)
    f=cam/(cam-z2) if cam>0 else 1.0
    return np.array([x1*f,y2*f])

def rng_for(seed,*key):
    """A generator tied to one mark, so changing one thing does not reshuffle the whole drawing."""
    return np.random.default_rng([seed,zlib.crc32(repr(key).encode())])

def smooth(pts,sub=8):
    """Catmull-Rom through the points: the housemachine trails' smoothed line."""
    P=np.asarray(pts,float)
    if len(P)<3: return P
    Q=np.vstack([2*P[0]-P[1],P,2*P[-1]-P[-2]]); out=[P[0]]
    for i in range(1,len(Q)-2):
        p0,p1,p2,p3=Q[i-1],Q[i],Q[i+1],Q[i+2]
        for t in np.linspace(0,1,sub+1)[1:]:
            out.append(0.5*((2*p1)+(-p0+p2)*t+(2*p0-5*p1+4*p2-p3)*t*t+(-p0+3*p1-3*p2+p3)*t**3))
    return np.array(out)

def knot(rng,c,r,n):
    """A jagged walk tethered to c: what the tracker draws when someone stands still."""
    p=np.array(c,float)+rng.normal(0,r*0.4,2); out=[]
    for _ in range(max(4,n)):
        p=p+rng.normal(0,r*0.3,2)+0.12*(np.array(c)-p); out.append(p.copy())
    return out

def trail(rng,a,b,toks,unit):
    """From a to b with one waypoint per token, pushed aside by that token's surprise."""
    a,b=np.array(a),np.array(b); d=b-a; L=np.linalg.norm(d) or 1; u=d/L; n=np.array([-u[1],u[0]])
    bow=rng.normal(0,0.1*L); pts=[a]; K=max(len(toks),3)
    for k in range(K):
        t=(k+1)/(K+1); s=max(0.0,toks[k]["s"]) if k<len(toks) else 3.0
        pts.append(a+d*t+n*(math.sin(math.pi*t)*bow+rng.normal(0,0.55*min(s,30))*unit)+u*rng.normal(0,0.3*min(s,30))*unit)
    pts.append(b)
    return smooth(pts)

def strand(rng,a,b,ra,rb,bow,L):
    """One attention strand: pairs share a bow, so strands bundle the way repeated walks do."""
    p0=np.array(a)+rng.normal(0,ra*0.7,2); p1=np.array(b)+rng.normal(0,rb*0.7,2); d=p1-p0
    n=np.array([-d[1],d[0]])/(np.linalg.norm(d) or 1); pts=[p0]
    drift=0.0
    for t in np.linspace(0,1,10)[1:-1]:                 # a slow drift, then a tremor: walked, not ruled
        drift=0.8*drift+rng.normal(0,0.02*L+1.0)
        pts.append(p0+d*t+n*(math.sin(math.pi*t)*(bow+drift)+rng.normal(0,1.1)))
    pts.append(p1)
    return smooth(pts,5)

def dashed_arc(a,b,on,off):
    """A jump: the same bow as graph3d's, broken into dashes. Each dash is its own pen stroke."""
    a,b=np.array(a),np.array(b); L=np.linalg.norm(b-a) or 1; m=(a+b)/2
    c=m+np.array([9.0,0]) if L<12 else m+np.array([-(b[1]-a[1]),b[0]-a[0]])*0.22
    t=np.linspace(0,1,max(40,int(L)))[:,None]; P=(1-t)**2*a+2*(1-t)*t*c+t*t*b
    s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(P,axis=0),axis=1))]; phase=np.mod(s,on+off)<on; out=[]; cur=[]
    for q,k in zip(P,phase):
        if k: cur.append(q)
        elif cur:
            if len(cur)>1: out.append(np.array(cur))
            cur=[]
    if len(cur)>1: out.append(np.array(cur))
    return out

def hershey(font,text,x,y,h,anchor="start"):
    """Single-stroke text with its baseline at y. Returns polylines; right-justified with anchor='end'."""
    k=h/CAP_HEIGHT; segs=list(font.lines_for_text(text)); paths=[]
    if not segs: return paths
    xs=[q[0] for sg in segs for q in sg]; dx=x-(max(xs)*k if anchor=="end" else min(xs)*k)
    for (x1,y1),(x2,y2) in segs:
        a=(dx+x1*k,y+(y1-BASELINE)*k); b=(dx+x2*k,y+(y2-BASELINE)*k)
        if paths and np.hypot(*(paths[-1][-1]-np.array(a)))<1e-6: paths[-1]=np.vstack([paths[-1],b])
        else: paths.append(np.array([a,b]))
    return paths

def circle(c,r):
    t=np.linspace(0,2*math.pi,max(16,int(r*8)))
    return np.stack([c[0]+r*np.cos(t),c[1]+r*np.sin(t)],1)

def chevron(p,d,size):
    """An arrowhead at p pointing along d: one stroke."""
    a=math.atan2(d[1],d[0]); wing=lambda b:np.array(p)-size*np.array([math.cos(a+b),math.sin(a+b)])
    return np.array([wing(0.45),p,wing(-0.45)])

def mid_arrow(path,size):
    m=len(path)//2; return chevron(path[m],path[min(m+2,len(path)-1)]-path[max(m-2,0)],size)

def block(font,lines,size,step):
    """Hershey lines set at the origin. Returns the strokes and their bounding box."""
    paths=[q for n,ln in enumerate(lines) for q in hershey(font,ln,0,n*step,size)]; A=np.vstack(paths)
    return paths,(A[:,0].min(),A[:,1].min(),A[:,0].max(),A[:,1].max())

def place_block(c,r,w,h,boxes,discs,ink,keep,no):
    """The nearest spot around c, as far as possible clear of other text, circles and the line."""
    ov=lambda a,b:max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    best=None
    for d in (2,6,12,20,30,44,60,80,110,150,200,280):
        if d>80 and best: break                    # the far rings are only for a block with nowhere nearer
        for k in range(16):
            t=k*math.pi/8; cx=c[0]+(r+d+w/2)*math.cos(t); cy=c[1]+(r+d+h/2)*math.sin(t)
            b=(cx-w/2,cy-h/2,cx+w/2,cy+h/2)
            if b[0]<keep[0] or b[1]<keep[1] or b[2]>keep[2] or b[3]>keep[3] or any(ov(b,q)>0 for q in no): continue
            hits=int(((ink[:,0]>b[0]-1)&(ink[:,0]<b[2]+1)&(ink[:,1]>b[1]-1)&(ink[:,1]<b[3]+1)).sum())
            sc=d*0.4+hits*5+sum(ov(b,q) for q in boxes)*3+sum(ov(b,q) for q in discs)*3+(0 if math.cos(t)>=-0.01 else 3)
            if best is None or sc<best[0]: best=(sc,b,d)
    return best[1],best[2]

def when(G):
    """Date and time of the conversation; older graph files only have the session's start."""
    if G.get("start"):
        d,t0=G["start"].split("T"); d1,t1=G["end"].split("T")
        return f"{d} {t0[:5]}-{t1[:5]}" if d==d1 else f"{d} {t0[:5]} to {d1} {t1[:5]}"
    m=re.search(r"(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})",G.get("session",""))
    return f"{m[1]}-{m[2]}-{m[3]} {m[4]}:{m[5]}" if m else ""

def hook(a,b,r,bow):
    """A short solid loop between neighbours in a stack, too short for dashes to read."""
    p0=np.array(a,float); p2=np.array(b,float); c=(p0+p2)/2+np.array([2*r+bow,0])   # centre to centre
    t=np.linspace(0,1,12)[:,None]; return (1-t)**2*p0+2*(1-t)*t*c+t*t*p2

def ticks(path,n,every,length):
    """n short cross-strokes every so often along a path: how a trail says which recording it is."""
    out=[]; d=np.diff(path,axis=0); L=np.hypot(d[:,0],d[:,1]); s=np.r_[0,np.cumsum(L)]
    for t in np.arange(every*0.6,s[-1]-every*0.3,every):
        for k in range(n):
            q=t+(k-(n-1)/2)*1.4; c=np.array([np.interp(q,s,path[:,0]),np.interp(q,s,path[:,1])])
            m=min(int(np.searchsorted(s,q)),len(d)-1); u=d[m]/(L[m] or 1); w=np.array([-u[1],u[0]])*length/2
            out.append(np.array([c-w,c+w]))
    return out

def excerpt(text,words):
    """The opening words of an utterance, marked as cut when there is more."""
    w=text.split(); return " ".join(w[:words])+("..." if len(w)>words else "")

def order(paths):
    """Greedy nearest-end ordering, reversing strokes where it helps, to keep pen travel short."""
    left=list(paths); out=[]; cur=np.zeros(2)
    while left:
        st=np.array([p[0] for p in left]); en=np.array([p[-1] for p in left])
        ds=((st-cur)**2).sum(1); de=((en-cur)**2).sum(1); i=int(np.argmin(np.minimum(ds,de)))
        p=left.pop(i); p=p if ds[i]<=de[i] else p[::-1]; out.append(p); cur=p[-1]
    return out

def overdraw(layers,w,h,cell=0.5):
    """Passes of ink per 0.5 mm cell, in the spirit of send_svg_axidraw.py --overdraw."""
    G=np.zeros((int(h/cell)+2,int(w/cell)+2),int); total=0.0
    for paths in layers:
        for p in paths:
            d=np.diff(p,axis=0); L=np.hypot(d[:,0],d[:,1]); total+=L.sum()
            if not L.sum(): continue
            s=np.r_[0,np.cumsum(L)]; t=np.arange(0,s[-1],cell*0.5)
            c=np.stack([np.interp(t,s,p[:,0]),np.interp(t,s,p[:,1])],1); ij=(c/cell).astype(int)
            keep=np.r_[True,(np.diff(ij,axis=0)!=0).any(1)]; ij=ij[keep]
            ok=(ij[:,0]>=0)&(ij[:,1]>=0)&(ij[:,0]<G.shape[1])&(ij[:,1]<G.shape[0]); ij=ij[ok]
            np.add.at(G,(ij[:,1],ij[:,0]),1)
    inked=G[G>0]
    return total/1000,int(G.max()),float(G[G>=5].sum()/max(1,inked.sum()))

def main():
    ap=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", required=True, help="the .json graph3d.py writes")
    ap.add_argument("--attn", required=True, help="attention and surprise cache (.json)")
    ap.add_argument("-o","--out", default="plotter-drawing.svg")
    ap.add_argument("--azimuth", type=float); ap.add_argument("--elevation", type=float); ap.add_argument("--camera", type=float)
    ap.add_argument("--paper", default="30x22", help="WxH in inches (default 30x22)")
    ap.add_argument("--surprise", action=argparse.BooleanOptionalAction, default=False,
                    help="let the trail wander with token surprise (default off)")
    ap.add_argument("--attention", action=argparse.BooleanOptionalAction, default=False,
                    help="draw the attention strands (default off)")
    ap.add_argument("--knots", action=argparse.BooleanOptionalAction, default=False,
                    help="scribbled knots at utterances, sized by spoken duration (default off: small open circles)")
    ap.add_argument("--sequence", action=argparse.BooleanOptionalAction, default=True,
                    help="turn numbers on labels and arrowheads along the trail (default on)")
    ap.add_argument("--session-ticks", action=argparse.BooleanOptionalAction, default=False,
                    help="with several recordings, mark each trail with cross-strokes (default off)")
    ap.add_argument("--strand-pct", type=float, default=0.5, help="attention per strand, in percent (default 0.5)")
    ap.add_argument("--stroke-mm", type=float, default=0.3); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--title", default="snapper and robert")
    ap.add_argument("--date", action=argparse.BooleanOptionalAction, default=True,
                    help="the conversation's date and time as a subtitle (default on)")
    ap.add_argument("--title-height", type=float, default=0.175, help="inches (default 0.175)")
    lab=ap.add_mutually_exclusive_group()
    lab.add_argument("--excerpt", type=int, nargs="?", const=5, default=None, metavar="WORDS",
                     help="label each knot with its first WORDS words (default 5), not the speaker")
    lab.add_argument("--full-text", action="store_true", help="label each utterance with its whole text (the default)")
    lab.add_argument("--speaker", action="store_true", help="label each utterance with just the speaker's name")
    ap.add_argument("--png", action="store_true", help="also write a proof PNG (needs inkscape)")
    a=ap.parse_args()
    a.full_text=not (a.excerpt or a.speaker)
    G=json.load(open(a.graph)); R=json.load(open(a.attn))
    sub=when(G) if a.date else ""; sub_h=0.1*MM; sub_gap=sub_h*1.9 if sub else 0   # date line under the title
    az,el,cam=[v if v is not None else d for v,d in zip((a.azimuth,a.elevation,a.camera),G["view"])]
    Wi,Hi=[float(v) for v in a.paper.lower().split("x")]; W,H=Wi*MM,Hi*MM; unit=W/762.0
    nodes={n["i"]:n for n in G["nodes"]}; seq=sorted(nodes); live=[i for i in seq if not nodes[i]["parked"]]
    # the form floats right of centre; the outside knot sits level with it, in the left margin
    Q=np.array([project(nodes[i]["p"],az,el,cam) for i in live]); lo,hi=Q.min(0),Q.max(0); span=np.maximum(hi-lo,1e-9)
    x0,x1,y0,y1=(0.22 if len(live)<len(seq) else 0.08)*W,0.92*W,0.09*H,0.86*H; s=min((x1-x0)/span[0],(y1-y0)/span[1])
    cx,cy=(x0+x1)/2,(y0+y1)/2; mid=(lo+hi)/2
    pos={i:(cx+s*(q[0]-mid[0]),cy-s*(q[1]-mid[1])) for i,q in zip(live,Q)}
    SS=G.get("sessions") or []; nG=max(1,len(SS)); ses=lambda i:nodes[i].get("session",0)
    homes={g:(0.095*W,cy if nG==1 else H*(0.1+0.8*g/nG)) for g in range(nG)}   # one stack per recording
    home=homes[0]; off=[i for i in seq if i not in pos]; gap=3.2*unit; groups={g:[i for i in off if ses(i)==g] for g in range(nG)}
    for g,ids in groups.items():
        for k,i in enumerate(ids):                 # with circles, each one its own place to land
            pos[i]=homes[g] if a.knots else (homes[g][0],homes[g][1]+(k-(len(ids)-1)/2)*gap)
    rad={i:(2.0+3.0*math.sqrt(nodes[i]["dur"] or 1.0))*unit if i in live else 7*unit for i in seq}
    dogvoice=lambda i:nodes[i]["type"]!="user"     # replies, and the canned greeting and reset cues in its voice
    if not a.knots: rad={i:(0.9+0.55*math.sqrt(nodes[i]["dur"] or 1.0))*unit if i in live else 2.2*unit for i in seq}
    num=lambda i,t:f'{nodes[i].get("label",i)} {t}' if a.sequence else t
    toks={}
    for x in R["exchanges"]: toks[x["user_i"]]=x["question"]; toks[x["asst_i"]]=x["reply"]
    # the trail: one pen line through the conversation, lifted only at a jump
    wiped={(l[0],l[1]) for l in G["links"] if l[2]=="#b91c1c"}   # a reset is a jump even between placed utterances
    trails=[]; jumps=[]; marks=[]; arrows=[]; tick=[]; cur=[]; arrow=3.2*max(unit,0.7); ring=1.1*max(unit,0.7)
    for k,i in enumerate(seq):
        if i in live and a.knots:
            r=rng_for(a.seed,"knot",i); cur+=knot(r,pos[i],rad[i],int(70*unit*unit*(nodes[i]["dur"] or 1.0)))
        elif i in live:
            cur.append(np.array(pos[i],float)); marks.append(circle(pos[i],rad[i]))
            if dogvoice(i): marks.append(circle(pos[i],rad[i]+ring))
        elif a.knots:
            jumps.append(np.array(knot(rng_for(a.seed,"out",i),pos[i],rad[i]*0.55,14)))
        else:                                      # a small stack of circles, outside the form
            marks.append(circle(pos[i],rad[i]))
            if dogvoice(i): marks.append(circle(pos[i],rad[i]+ring*0.7))
        if k+1==len(seq): break
        j=seq[k+1]
        if ses(i)!=ses(j):                         # recordings are separate: no line from one to the next
            if len(cur)>1: trails.append(np.array(cur))
            cur=[]; continue
        if i in live and j in live and (i,j) not in wiped:
            end=knot(rng_for(a.seed,"knot",j),pos[j],rad[j],1)[0] if a.knots else pos[j]
            seg=trail(rng_for(a.seed,"trail",i),cur[-1],end,toks.get(j,[]) if a.surprise else [],unit)
            cur+=list(seg); arrows.append(mid_arrow(seg,arrow))
            if SS and a.session_ticks and ses(i)>0: tick+=ticks(seg,ses(i),16*max(unit,0.7),2.4)
        else:
            if len(cur)>1: trails.append(np.array(cur))
            cur=[]
            short=not a.knots and i not in live and j not in live and np.hypot(*(np.array(pos[i])-np.array(pos[j])))<12
            arc=[hook(pos[i],pos[j],rad[i],3.5*unit)] if short else dashed_arc(pos[i],pos[j],5*unit,4*unit); jumps+=arc
            if len(arc)>2: arrows.append(mid_arrow(arc[len(arc)//2],arrow))
    if len(cur)>1: trails.append(np.array(cur))
    # attention: strands into each reply, bundled by pair
    strands=[]
    if a.attention and a.strand_pct>0:
        P=R["parts"]; per=a.strand_pct/100
        for x in R["exchanges"]:
            sh=np.array(x["shares"]); dst=x["asst_i"]
            src=[(x["user_i"],float(sh[:,P.index("question")].mean()))]
            src+=[(h["i"],float(np.mean(h["share"]))) for h in x["history"] if h["i"] is not None]
            for i,v in src:
                if i not in pos or i not in live: continue
                L=float(np.hypot(*(np.array(pos[i])-np.array(pos[dst])))); bow=rng_for(a.seed,"bow",i,dst).normal(0,0.13*L)
                for n in range(int(round(v/per))):
                    strands.append(strand(rng_for(a.seed,"strand",i,dst,n),pos[i],pos[dst],rad[i],rad[dst],bow,L))
    # labels at the knots and the title in the lower right, as on the housemachine sheets
    font=HersheyFonts(); font.load_default_font("futural"); labels=[]; lh=0.12*MM   # a fixed physical size, as on the housemachine sheets
    size=lh*(0.62 if a.full_text else 0.75); step=size*1.5; hx,hy=home; n_out=[i for i in seq if i not in live]
    wrapped=lambda i:textwrap.wrap(num(i,nodes[i]["text"]),width=34) if a.full_text else [num(i,excerpt(nodes[i]["text"],a.excerpt))]
    if a.speaker:
        for i in live:
            x,y=pos[i]; rl=rng_for(a.seed,"label",i)
            labels+=hershey(font,num(i,nodes[i]["who"].lower()),x+rad[i]*0.8+1.5,y-rad[i]*0.7+rl.normal(0,1.5),lh)
        if n_out: labels+=hershey(font,"start over "+" ".join(map(str,n_out)) if a.sequence else "start over",hx-14*unit,hy+rad[n_out[0]]+7,lh)
    else:
        def samples(paths):
            return np.vstack([p[::max(1,len(p)//400)] for p in paths]) if paths else np.zeros((0,2))
        ink=np.vstack([samples(trails),samples(jumps)]); boxes=[]
        outer=lambda i:rad[i]+(ring if not a.knots and dogvoice(i) else 0)
        discs=[(pos[i][0]-outer(i)-1.5,pos[i][1]-outer(i)-1.5,pos[i][0]+outer(i)+1.5,pos[i][1]+outer(i)+1.5) for i in live]
        keep=(0.02*W,0.03*H,0.985*W,0.945*H)
        tw=len(a.title)*a.title_height*MM*0.85 if a.title else 0; no=[(W-0.5*MM-tw,H-0.5*MM-sub_gap-a.title_height*MM*2,W,H)]
        ks=lh*0.62; kw=0.5*MM+40                   # keep every block off the key and the stacks with their lists
        for x in SS: kw=max(kw,0.5*MM+(18 if a.session_ticks else 0)+block(font,[f'{x["letter"]}  {x["start"][:10]} {x["start"][11:16]}  {x["name"]}'],ks,0)[1][2])
        no.append((0,H-0.5*MM-16-max(0,len(SS)-1)*5.2-ks,kw+3,H))
        lists=[]
        for g,ids in groups.items():               # the outside utterances, listed under their stack
            if not ids: continue
            yy=y0_=homes[g][1]+len(ids)/2*gap+rad[ids[0]]+6*unit; right=homes[g][0]+rad[ids[0]]
            for i in ids:
                ls=textwrap.wrap(num(i,nodes[i]["text"]),width=30) if a.full_text else wrapped(i)
                for ln in ls:
                    q=hershey(font,ln,0.025*W,yy+size,size); lists+=q; right=max(right,max(float(p_[:,0].max()) for p_ in q)); yy+=step
                yy+=step*0.5
            no.append((0.025*W-2,homes[g][1]-len(ids)/2*gap-rad[ids[0]]-2,right+2.5,yy+2))
        for i in sorted(live,key=lambda i:-len(nodes[i]["text"])):
            paths,(bx,by,ex,ey)=block(font,wrapped(i),size,step)
            b,d=place_block(pos[i],outer(i)+1,ex-bx,ey-by,boxes,discs,ink,keep,no); boxes.append(b)
            labels+=[p+np.array([b[0]-bx,b[1]-by]) for p in paths]
            if d>=12:                              # a leader, when the text had to sit away from its circle
                q=np.array([min(max(pos[i][0],b[0]),b[2]),min(max(pos[i][1],b[1]),b[3])]); c=np.array(pos[i],float); v=q-c; L=np.linalg.norm(v) or 1
                labels.append(np.array([c+v/L*(outer(i)+0.8),q-v/L*1.2]))
        labels+=lists
    if not a.knots:                                # a small key, on the title's baseline
        dog=next((nodes[i]["who"].lower() for i in live if nodes[i]["type"]=="assistant"),"reply")
        kx,ky,kr=0.5*MM,H-0.5*MM,1.8; ks=lh*0.75
        marks.append(circle((kx+kr,ky-kr),kr)); labels+=hershey(font,"robert",kx+2*kr+2.5,ky,ks)
        kx2=kx+2*kr+2.5+block(font,["robert"],ks,0)[1][2]+9
        marks+=[circle((kx2+kr+ring,ky-kr),kr),circle((kx2+kr+ring,ky-kr),kr+ring)]; labels+=hershey(font,dog,kx2+2*(kr+ring)+2.5,ky,ks)
    if SS:                                         # one key line per recording, above the speaker key
        ks=lh*0.62; kl=0.5*MM
        for g,x in enumerate(SS):
            ky=H-0.5*MM-9-(len(SS)-1-g)*5.2; tx=kl
            if a.session_ticks:
                sample=np.array([[kl,ky-1],[kl+14,ky-1]]); jumps.append(sample); tx=kl+18
                if g: tick+=ticks(sample,g,8,2.4)
            labels+=hershey(font,f'{x["letter"]}  {x["start"][:10]} {x["start"][11:16]}  {x["name"]}',tx,ky,ks)
    title=hershey(font,a.title,W-0.5*MM,H-0.5*MM-sub_gap,a.title_height*MM,"end") if a.title else []
    if sub: title+=hershey(font,sub,W-0.5*MM,H-0.5*MM,sub_h,"end")
    strands,trails_,jumps,labels=order(strands),trails,order(jumps),order(labels)
    seqmarks=(order(arrows) if a.sequence else [])+tick
    if marks: trails_=trails_+order(marks)
    d=lambda p:"M"+" L".join(f"{x:.2f},{y:.2f}" for x,y in p)
    layers=[("layer_attention",strands),("layer_trail",trails_),("layer_jumps",jumps),("layer_sequence",seqmarks),("layer_labels",labels),("layer_title",title)]
    o=[f'<svg xmlns="http://www.w3.org/2000/svg" version="1.2" width="{Wi:g}in" height="{Hi:g}in" viewBox="0 0 {W:.2f} {H:.2f}" overflow="visible">',
       f'<metadata><desc id="invocation">{" ".join(sys.argv).replace("&","&amp;").replace("<","&lt;")}</desc></metadata>']
    for name,paths in layers:
        o.append(f'<g id="{name}" fill="none" stroke="rgb(0,0,0)" stroke-width="{a.stroke_mm}" stroke-linecap="round" stroke-linejoin="round">')
        o+= [f'<path d="{d(p)}"/>' for p in paths]; o.append("</g>")
    o.append("</svg>"); open(a.out,"w").write("\n".join(o))
    metres,worst,heavy=overdraw([p for _,p in layers],W,H)
    stem=os.path.splitext(a.out)[0]
    json.dump(dict(command=" ".join(sys.argv),view=[az,el,cam],seed=a.seed,paper=[Wi,Hi],stroke_mm=a.stroke_mm,strand_pct=a.strand_pct,surprise=a.surprise,attention=a.attention,knots=a.knots,sequence=a.sequence,labels="full" if a.full_text else (f"excerpt {a.excerpt}" if a.excerpt else "speaker"),
                   strokes={n:len(p) for n,p in layers},pen_down_m=round(metres,2),worst_passes=worst,ink_share_at_5x_plus=round(heavy,3)),open(stem+".plot.json","w"),indent=1)
    print(f"{a.out}  ({Wi:g}x{Hi:g} in, {sum(len(p) for _,p in layers)} strokes, {metres:.1f} m of ink; "+", ".join(f"{n[6:]} {len(p)}" for n,p in layers)+")")
    print(f"overdraw: worst spot {worst} passes, {heavy*100:.0f}% of ink lands on spots hit 5x or more")
    if a.png:
        if not shutil.which("inkscape"): sys.exit("--png needs inkscape")
        r=subprocess.run(["inkscape",a.out,"-o",stem+".png","-w",str(int(Wi*100)),"-b","#ffffff"],capture_output=True,text=True)
        if r.returncode: sys.exit(r.stderr)
        print(stem+".png")

if __name__=="__main__": main()
