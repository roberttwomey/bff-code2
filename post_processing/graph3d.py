#!/usr/bin/env python3
"""
graph3d.py — the dialogue graph as a 3D node graph, and 2D projections of it.

Runs MDS to three dimensions on the same utterance distances as conversation_graph.py, then
projects the graph through a camera to an SVG (and optionally a PNG). The graph keeps its
shape from one view to the next, so renders from different angles are views of one object.

usage:
  python3 graph3d.py SESSION_DIR [--space activation] [--acts ACTS.npz] [--layer 18]
                     [--azimuth 30] [--elevation 20] [--camera 4] [-o OUT.svg] [--png]
                     [--viewer] [--obj]

Camera: --azimuth turns the graph about its vertical axis, --elevation raises the viewpoint
(degrees), --camera is the distance in graph radii (smaller = stronger perspective; 0 = orthographic).

Also writes:
  OUT.json     3D coordinates, text and edges for every utterance
  OUT.html     with --viewer: drag to rotate, scroll to zoom; prints the flags for this view
  OUT.obj      with --obj: sphere nodes and line edges, Z-up, for Rhino and other 3D tools
"""
import os
import sys
import json
import math
import argparse
import datetime as dt
import numpy as np
import conversation_graph as cg
from conversation_graph import W,H,INK,MUT,FAINT,BG,HUM,DOG,SYS,VIS,RST,ramp,e

PX0,PX1,PY0,PY1=62,1658,132,1010

def floor_of(P):
    """A reference plane just under the graph, fitted to its footprint: (y, x0, x1, z0, z1)."""
    m=0.08; return float(P[:,1].min())-m,float(P[:,0].min())-m,float(P[:,0].max())+m,float(P[:,2].min())-m,float(P[:,2].max())+m

def layout3d(D,hum,b1):
    """3D MDS, centred, axes by decreasing spread, scaled uniformly to unit radius."""
    import activation_graph as ag
    P=ag.mds2(D,3); P=P-P.mean(0); _,_,vt=np.linalg.svd(P,full_matrices=False); P=P@vt.T
    if b1 and P[b1,0].mean()>0: P[:,0]*=-1
    if P[hum,1].mean()<0: P[:,1]*=-1
    P/=np.linalg.norm(P,axis=1).max()
    for a in range(len(P)):                        # identical utterances: nudge apart, deterministically
        for b in range(a+1,len(P)):
            if np.linalg.norm(P[a]-P[b])<0.03:
                ang=a*2.399; d=np.array([math.cos(ang),math.sin(ang),0.3])*0.025; P[a]-=d; P[b]+=d
    return P

def project(X,az,el,cam):
    """Rotate about the vertical axis, tilt, then perspective. Returns screen x, y, depth (+ toward camera), scale."""
    a,b=math.radians(az),math.radians(el); X=np.atleast_2d(X)
    x1=X[:,0]*math.cos(a)+X[:,2]*math.sin(a); z1=-X[:,0]*math.sin(a)+X[:,2]*math.cos(a)
    y2=X[:,1]*math.cos(b)-z1*math.sin(b); z2=X[:,1]*math.sin(b)+z1*math.cos(b)
    f=cam/(cam-z2) if cam>0 else np.ones(len(z2))
    return x1*f,y2*f,z2,f

def render(cfg,U,resets,live,P,edges,session,a,fid3,fid2,how):
    o=[]; A=o.append; resets=set(resets); lp={i:k for k,i in enumerate(live)}; az,el,cam=a.azimuth,a.elevation,a.camera
    parked=[u for u in U if u["i"] not in lp]; top=PY0+(66 if parked else 14)
    floor,fx0,fx1,fz0,fz1=floor_of(P)
    corners=np.array([[x,floor,z] for x in (fx0,fx1) for z in (fz0,fz1)])
    sx,sy,dz,f=project(P,az,el,cam); cx,cy,_,_=project(corners,az,el,cam)
    ax=np.r_[sx,cx]; ay=np.r_[sy,cy]; pad=42
    s=min((PX1-PX0-2*pad)/np.ptp(ax),(PY1-top-pad-44)/np.ptp(ay))
    mx,my=(ax.min()+ax.max())/2,(ay.min()+ay.max())/2; ox,oy=(PX0+PX1)/2,(top+PY1-44)/2
    sc=lambda q:(ox+s*(q[0]-mx),oy-s*(q[1]-my))
    pos={i:sc((sx[k],sy[k])) for i,k in lp.items()}
    d01=(dz-dz.min())/max(1e-9,np.ptp(dz)); fog={i:float(d01[k]) for i,k in lp.items()}
    rad={i:(6.5+3.3*math.sqrt(U[i]["dur"] or 1.0))*float(np.clip(f[k],0.7,1.35)) for i,k in lp.items()}
    pt=lambda X:sc(tuple(v[0] for v in project(np.array(X,float),az,el,cam)[:2]))
    t0,t1=U[0]["t"],U[-1]["t"]
    A(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Helvetica Neue, Helvetica, Arial, sans-serif">')
    A(f'<rect width="{W}" height="{H}" fill="{BG}"/><defs>')
    for k in range(6):
        A(f'<marker id="ar{k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,1.4 L9,5 L0,8.6 z" fill="{ramp(k/5)}"/></marker>')
    A(f'<marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,1.4 L9,5 L0,8.6 z" fill="{RST}"/></marker></defs>')
    A(f'<text x="62" y="52" font-size="27" font-weight="700" fill="{INK}" letter-spacing="0.4">SNAPPER  &#215;  ROBERT</text>')
    A(f'<text x="62" y="76" font-size="12.5" fill="{MUT}">{e(session)} &#183; {t0:%Y-%m-%d}, {t0:%H:%M}&#8211;{t1:%H:%M} &#183; {len(live)} utterances &#183; {how}</text>')
    A(f'<line x1="62" y1="96" x2="{PX1}" y2="96" stroke="{FAINT}"/>')
    cam_txt=f"camera {cam:g} radii" if cam>0 else "orthographic"
    A(f'<text x="{PX0+14}" y="{PY0-8}" font-size="11.5" font-weight="700" fill="{INK}" letter-spacing="1.2">SEMANTIC SPACE, 3D</text>')
    A(f'<text x="{PX0+176}" y="{PY0-8}" font-size="11" fill="{MUT}">azimuth {az:g}&#176; &#183; elevation {el:g}&#176; &#183; {cam_txt} &#183; larger and darker = nearer the camera &#183; thin lines drop to a floor grid</text>')
    A(f'<rect x="{PX0}" y="{PY0}" width="{PX1-PX0}" height="{PY1-PY0}" fill="#ffffff" stroke="{FAINT}"/>')
    grid=[([v,floor,fz0],[v,floor,fz1]) for v in np.linspace(fx0,fx1,7)]+[([fx0,floor,v],[fx1,floor,v]) for v in np.linspace(fz0,fz1,7)]
    for p1,p2 in grid:                             # the floor: a reference plane, not data
        (x1,y1),(x2,y2)=pt(p1),pt(p2)
        A(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{FAINT}" stroke-width="0.7"/>')
    for i,k in lp.items():
        (x1,y1),(x2,y2)=pos[i],pt([P[k,0],floor,P[k,2]])
        A(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{MUT}" stroke-width="0.5" stroke-opacity="0.35"/>')
        A(f'<ellipse cx="{x2:.1f}" cy="{y2:.1f}" rx="2.4" ry="1.2" fill="{MUT}" fill-opacity="0.4"/>')
    items=[]                                       # painter's algorithm: far things first
    def link(i,j,col,w,op,dash="",marker=""):
        (x1,y1),(x2,y2)=pos[i],pos[j]; L=math.hypot(x2-x1,y2-y1) or 1; ux,uy=(x2-x1)/L,(y2-y1)/L
        x1+=ux*(rad[i]+2); y1+=uy*(rad[i]+2)
        if marker: x2-=ux*(rad[j]+5); y2-=uy*(rad[j]+5)
        d=(fog[i]+fog[j])/2
        items.append((d-0.001,f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{col}" stroke-width="{w*(0.7+0.6*d):.2f}" stroke-opacity="{op*(0.55+0.45*d):.2f}"'
                     +(f' stroke-dasharray="{dash}"' if dash else "")+(f' marker-end="url(#{marker})"' if marker else "")+'/>'))
    for i,j,v,k in edges:
        if i in lp and j in lp: link(i,j,DOG if k=="SS" else (HUM if k=="HH" else VIS),1.1,0.4,"3.5 3")
    for n,(i,j) in enumerate(zip(live,live[1:])):
        if i in resets or j in resets or any(i<c<j for c in resets): link(i,j,RST,1.5,0.85,"2 5","arr")
        else:
            q=n/max(1,len(live)-2); link(i,j,ramp(q),1.2,0.8,"",f"ar{min(5,int(q*6))}")
    for i in live:
        c=HUM if U[i]["speaker"]=="HUMAN" else (DOG if U[i]["type"]=="assistant" else SYS); x,y=pos[i]; d=fog[i]
        items.append((d,f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{rad[i]:.1f}" fill="{c}" fill-opacity="{0.5+0.45*d:.2f}" stroke="#ffffff" stroke-width="1.2"/>'
                        f'<text x="{x:.1f}" y="{y+3.2:.1f}" font-size="8.2" text-anchor="middle" font-weight="700" fill="#ffffff" font-family="SF Mono, Menlo, monospace">{i}</text>'))
    for _,svg in sorted(items,key=lambda t:t[0]): A(svg)
    say=lambda u:(u["logged"] or u["text"]) if a.space!="tfidf" else u["text"]
    for i,lx,ly,right,t in cg.place_labels([(i,*pos[i],rad[i],say(U[i])) for i in live],[(i,*pos[i],rad[i]) for i in live],PX0,PX1,top,PY1-44):
        A(f'<line x1="{pos[i][0]:.1f}" y1="{pos[i][1]:.1f}" x2="{lx:.1f}" y2="{ly-3.4:.1f}" stroke="{FAINT}" stroke-width="0.8"/>')
        A(f'<text x="{lx:.1f}" y="{ly:.1f}" font-size="9.3" fill="{INK}" fill-opacity="0.85" text-anchor="{"start" if right else "end"}" stroke="#ffffff" stroke-width="2.4" paint-order="stroke">{e(t)}</text>')
    if parked:
        A(f'<text x="{PX0+14}" y="{PY0+18}" font-size="9.2" font-weight="700" fill="{MUT}" letter-spacing="0.9">NEVER ENTERED THE MODEL &#183; NO POSITION IN THIS SPACE</text>')
        for n,u in enumerate(parked):
            x=PX0+24+(n%3)*190; y=PY0+36+(n//3)*20; c=HUM if u["speaker"]=="HUMAN" else SYS
            t=u["text"]; t=(t[:24].rstrip()+"…") if len(t)>25 else t
            A(f'<circle cx="{x}" cy="{y-3}" r="8" fill="none" stroke="{c}" stroke-width="1.3" stroke-dasharray="2.5 2"/><text x="{x}" y="{y}" font-size="7.8" text-anchor="middle" font-weight="700" fill="{c}" font-family="SF Mono, Menlo, monospace">{u["i"]}</text>')
            A(f'<text x="{x+13}" y="{y}" font-size="9" fill="{INK}" fill-opacity="0.85">{e(t)}</text>')
    fx=PX0+14; fy=PY1-14
    A(f'<text x="{fx}" y="{fy}" font-size="10.4" fill="{MUT}">3D keeps the full-dimension distance ranks at &#961; = {fid3:.2f} (2D MDS: {fid2:.2f}) &#183; on-screen distance is not model distance: depth is foreshortened, and only a rotation shows it</text>')
    kx=PX1-430
    for c,t in ((HUM,"Robert"),(DOG,"Snapper")):
        A(f'<circle cx="{kx}" cy="{fy-16}" r="4.5" fill="{c}"/><text x="{kx+9}" y="{fy-13}" font-size="9.6" fill="{INK}">{t}</text>'); kx+=68
    A(f'<line x1="{kx}" y1="{fy-16}" x2="{kx+20}" y2="{fy-16}" stroke="{ramp(.8)}" stroke-width="1.2"/><text x="{kx+25}" y="{fy-13}" font-size="9.6" fill="{INK}">time</text>'); kx+=60
    A(f'<line x1="{kx}" y1="{fy-16}" x2="{kx+20}" y2="{fy-16}" stroke="{RST}" stroke-width="1.5" stroke-dasharray="2 4"/><text x="{kx+25}" y="{fy-13}" font-size="9.6" fill="{INK}">wipe</text>'); kx+=60
    A(f'<line x1="{kx}" y1="{fy-16}" x2="{kx+20}" y2="{fy-16}" stroke="{DOG}" stroke-width="1.2" stroke-opacity="0.5" stroke-dasharray="3.5 3"/><text x="{kx+25}" y="{fy-13}" font-size="9.6" fill="{INK}">echo</text>')
    A(f'<line x1="62" y1="1042" x2="{PX1}" y2="1042" stroke="{FAINT}"/>')
    A(f'<text x="62" y="1064" font-size="10" fill="{MUT}">Metric MDS to three dimensions on the same utterance distances as the 2D dialogue graph, scaled uniformly. Axes run widest to narrowest. Dashed lines join the most similar non-adjacent pairs.</text>')
    A(f'<text x="62" y="1088" font-size="10" fill="{MUT}">Robert Twomey &#183; BFF / Dog Walk &#183; rendered {dt.date.today():%Y-%m-%d}</text></svg>')
    return o

def write_obj(path,U,live,P,edges,resets):
    """Rhino is Z-up: graph (x, up, toward viewer) -> OBJ (x, -toward viewer, up), 100 units = graph radius."""
    Q=np.stack([P[:,0],-P[:,2],P[:,1]],1)*100; lp={i:k for k,i in enumerate(live)}; lines=[]; nv=0
    def sphere(c,r,la=8,lo=12):
        nonlocal nv
        base=nv; vs=[(c[0],c[1],c[2]+r)]
        for a_ in range(1,la):
            th=math.pi*a_/la
            for b_ in range(lo):
                ph=2*math.pi*b_/lo; vs.append((c[0]+r*math.sin(th)*math.cos(ph),c[1]+r*math.sin(th)*math.sin(ph),c[2]+r*math.cos(th)))
        vs.append((c[0],c[1],c[2]-r))
        for v in vs: lines.append("v %.4f %.4f %.4f"%v)
        nv+=len(vs); ring=lambda a_,b_:base+2+(a_-1)*lo+b_%lo
        for b_ in range(lo):
            lines.append(f"f {base+1} {ring(1,b_)} {ring(1,b_+1)}")
            for a_ in range(1,la-1): lines.append(f"f {ring(a_,b_)} {ring(a_+1,b_)} {ring(a_+1,b_+1)} {ring(a_,b_+1)}")
            lines.append(f"f {base+len(vs)} {ring(la-1,b_+1)} {ring(la-1,b_)}")
    def seg(name,pairs):
        nonlocal nv
        lines.append(f"o {name}")
        for i,j in pairs:
            for v in (Q[lp[i]],Q[lp[j]]): lines.append("v %.4f %.4f %.4f"%tuple(v))
            nv+=2; lines.append(f"l {nv-1} {nv}")
    for i in live:
        u=U[i]; who="robert" if u["speaker"]=="HUMAN" else ("snapper" if u["type"]=="assistant" else "system")
        lines.append(f"o node_{i:02d}_{who}"); sphere(Q[lp[i]],1.6+0.75*math.sqrt(u["dur"] or 1.0))
    pairs=list(zip(live,live[1:])); wipe=lambda i,j:i in resets or j in resets or any(i<c<j for c in resets)
    seg("dialogue_spine",[p for p in pairs if not wipe(*p)]); seg("context_wipes",[p for p in pairs if wipe(*p)])
    seg("echoes",[(i,j) for i,j,_,_ in edges if i in lp and j in lp])
    open(path,"w").write("\n".join(lines)+"\n")

VIEWER="""<!doctype html><meta charset="utf-8"><title>__TITLE__</title>
<style>body{margin:0;background:#faf8f5;font:13px Helvetica Neue,Arial,sans-serif;color:#1c1917;overflow:hidden}
#b{position:fixed;left:14px;top:10px;right:14px;pointer-events:none}#c{font:12px Menlo,monospace;user-select:all;pointer-events:auto;color:#44403c}
#t{position:fixed;left:14px;bottom:12px;right:14px;max-width:760px;min-height:36px}</style>
<div id="b"><b>__TITLE__</b> &mdash; drag to rotate, scroll to change perspective, shift-scroll for orthographic, hover for text<br><span id="c"></span></div><div id="t"></div><canvas id="v"></canvas>
<script>const G=__DATA__;let az=G.view[0],el=G.view[1],cam=G.view[2];const v=document.getElementById('v'),x=v.getContext('2d');let hover=null,pts=[];
function proj(p){const a=az*Math.PI/180,b=el*Math.PI/180,x1=p[0]*Math.cos(a)+p[2]*Math.sin(a),z1=-p[0]*Math.sin(a)+p[2]*Math.cos(a),y2=p[1]*Math.cos(b)-z1*Math.sin(b),z2=p[1]*Math.sin(b)+z1*Math.cos(b),f=cam>0?cam/(cam-z2):1;return[x1*f,y2*f,z2,f]}
function draw(){const w=v.width=innerWidth,h=v.height=innerHeight,s=Math.min(w,h)*.38,S=q=>[w/2+s*q[0],h/2-s*q[1]];x.fillStyle='#faf8f5';x.fillRect(0,0,w,h);
const[F,x0,x1,z0,z1]=G.floor;x.strokeStyle='#dcd8d2';x.lineWidth=1;for(let i=0;i<7;i++){const u=x0+(x1-x0)*i/6,w=z0+(z1-z0)*i/6;for(const[a,b]of[[[u,F,z0],[u,F,z1]],[[x0,F,w],[x1,F,w]]]){const A=S(proj(a)),B=S(proj(b));x.beginPath();x.moveTo(...A);x.lineTo(...B);x.stroke()}}
const items=[];pts=G.nodes.map(n=>{const q=proj(n.p),o=S(q);return{n,q,o}});const by=Object.fromEntries(pts.map(p=>[p.n.i,p]));
for(const[i,j,c]of G.links){const a=by[i],b=by[j];items.push([(a.q[2]+b.q[2])/2-.001,()=>{x.strokeStyle=c;x.lineWidth=c=='#b91c1c'?1.5:1;x.setLineDash(c=='#b91c1c'?[2,5]:[]);x.beginPath();x.moveTo(...a.o);x.lineTo(...b.o);x.stroke();x.setLineDash([])}])}
for(const p of pts)items.push([p.q[2],()=>{const r=(6.5+3.3*Math.sqrt(p.n.dur||1))*Math.max(.7,Math.min(1.35,p.q[3]));x.fillStyle=p.n.c;x.globalAlpha=.9;x.beginPath();x.arc(...p.o,r,0,7);x.fill();x.globalAlpha=1;x.fillStyle='#fff';x.font='bold 9px Menlo';x.textAlign='center';x.fillText(p.n.i,p.o[0],p.o[1]+3)}]);
items.sort((a,b)=>a[0]-b[0]).forEach(t=>t[1]());
document.getElementById('c').textContent=`--azimuth ${az.toFixed(0)} --elevation ${el.toFixed(0)} --camera ${cam.toFixed(1)}`}
let d=null;v.onmousedown=e=>d=[e.clientX,e.clientY];onmouseup=()=>d=null;onresize=draw;
onmousemove=e=>{if(d){az+=(e.clientX-d[0])*.4;el=Math.max(-89,Math.min(89,el+(e.clientY-d[1])*.4));d=[e.clientX,e.clientY];draw()}else{let h=null,b=400;for(const p of pts){const q=(p.o[0]-e.clientX)**2+(p.o[1]-e.clientY)**2;if(q<b){b=q;h=p}}document.getElementById('t').textContent=h?`${h.n.i} ${h.n.who}: ${h.n.text}`:''}};
v.onwheel=e=>{e.preventDefault();cam=e.shiftKey?0:Math.max(1.5,Math.min(20,(cam||4)+e.deltaY*.01));draw()};draw()</script>"""

def main():
    ap=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_dir"); ap.add_argument("-o","--out", default=None)
    ap.add_argument("--space", choices=("activation","standalone","tfidf"), default="activation")
    ap.add_argument("--layer", type=int, default=18); ap.add_argument("--acts", default=None)
    ap.add_argument("--model", default=None); ap.add_argument("--system-prompt", default=None, metavar="SCRIPT_OR_TXT")
    ap.add_argument("--echo-threshold", type=float, default=0.86, help="tfidf only")
    ap.add_argument("--azimuth", type=float, default=30); ap.add_argument("--elevation", type=float, default=20)
    ap.add_argument("--camera", type=float, default=4, help="distance in graph radii; 0 = orthographic")
    ap.add_argument("--viewer", action="store_true", help="write an interactive OUT.html")
    ap.add_argument("--obj", action="store_true", help="write OUT.obj for 3D tools")
    ap.add_argument("--png", action="store_true"); ap.add_argument("--png-width", type=int, default=2*W)
    a=ap.parse_args()
    if 0<a.camera<=1.5: sys.exit("--camera must exceed 1.5 graph radii, or be 0 for orthographic")
    cfg,U,resets,n_vlm=cg.load(a.session_dir)
    session=os.path.basename(a.session_dir.rstrip("/"))
    if "__" in session: session=[s for s in session.split("__") if s.startswith("session-")][0]
    out=a.out or f"{session}-graph3d.svg"
    if a.space=="tfidf":
        P2,sim,edges=cg.embed(U,a.echo_threshold); live=[u["i"] for u in U]
        how="placed by word overlap (TF-IDF, LSA, MDS)"
    else:
        cfg,space,P2,sim,edges,_,_,_=cg.model_space(a,cfg,U,n_vlm,out); live=space["live"]
        how=f"hidden state at layer {a.layer}, "+("in context" if a.space=="activation" else "each utterance alone")
    D=np.clip(1-sim[np.ix_(live,live)],0,None); D=(D+D.T)/2; np.fill_diagonal(D,0)
    lp={i:k for k,i in enumerate(live)}
    hum=[lp[u["i"]] for u in U if u["i"] in lp and u["speaker"]=="HUMAN"]; b1=[lp[u["i"]] for u in U if u["i"] in lp and u.get("block")==1]
    P=layout3d(D,hum,b1)
    from scipy.stats import spearmanr
    from scipy.spatial.distance import pdist
    iu=np.triu_indices(len(live),1); fid3=float(spearmanr(pdist(P),D[iu]).statistic); fid2=float(spearmanr(pdist(P2[live]),D[iu]).statistic)
    open(out,"w").write("\n".join(render(cfg,U,resets,live,P,edges,session,a,fid3,fid2,how)))
    print(f"{out}  ({len(live)} nodes, az {a.azimuth:g} el {a.elevation:g} cam {a.camera:g}; distance ranks rho {fid3:.2f} in 3D, {fid2:.2f} in 2D)")
    stem=os.path.splitext(out)[0]; rs=set(resets); pairs=list(zip(live,live[1:]))
    who=lambda u:"ROBERT" if u["speaker"]=="HUMAN" else ("SNAPPER" if u["type"]=="assistant" else "SYSTEM")
    nodes=[dict(i=i,who=who(U[i]),type=U[i]["type"],text=U[i]["logged"] or U[i]["text"],dur=U[i]["dur"],p=[round(float(v),5) for v in P[lp[i]]],
                c=HUM if who(U[i])=="ROBERT" else (DOG if who(U[i])=="SNAPPER" else SYS)) for i in live]
    links=[[i,j,RST if (i in rs or j in rs or any(i<c<j for c in rs)) else "#9a948c"] for i,j in pairs]
    data=dict(session=session,space=a.space,nodes=nodes,links=links,floor=[round(v,4) for v in floor_of(P)],view=[a.azimuth,a.elevation,a.camera],
              echoes=[[i,j,round(s,4)] for i,j,s,_ in edges if i in lp and j in lp],unplaced=[u["i"] for u in U if u["i"] not in lp])
    json.dump(data,open(stem+".json","w"),indent=1); print(stem+".json")
    if a.viewer:
        page=VIEWER.replace("__TITLE__",e(f"{session} · 3D")).replace("__DATA__",json.dumps(data).replace("</","<\\/"))
        open(stem+".html","w").write(page); print(stem+".html")
    if a.obj: write_obj(stem+".obj",U,live,P,edges,rs); print(stem+".obj")
    if a.png:
        png=stem+".png"; print(f"{png}  ({a.png_width} px wide, via {cg.export_png(out,png,a.png_width)})")

if __name__=="__main__": main()
