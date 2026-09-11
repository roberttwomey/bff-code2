#!/usr/bin/env python3
"""
activation_graph.py — the dialogue graph re-drawn in Gemma 4's own activation space.

Replays every prompt the dog actually received (system prompt, history, the live
transcription, the visual-context packet), teacher-forces its logged reply through
google/gemma-4-E2B-it, and mean-pools each layer's hidden states over each utterance's
tokens. Utterances that never reached the model (startup, reset cues, the intercepted
"let's start over" requests, unanswered turns) are kept out of the space and shown apart.

usage:
  python3 activation_graph.py SESSION_DIR --model MODEL_DIR [-o OUT.svg] [--layer 18] [--png]

Activations are cached next to the SVG (OUT.acts.npz) so re-rendering skips the model.
"""
import os
import sys
import json
import math
import argparse
import datetime as dt
import numpy as np
import conversation_graph as cg

VISUAL_CONTEXT_PREFIX = "Visual context (what you see):"   # chat-manager.py @ 8301af8
BODY_STATE_PREFIX = "Body state:"
LEG_NAMES = ("FR", "FL", "RR", "RL")
JOINT_NAMES = tuple(f"{leg}_{joint}" for leg in LEG_NAMES for joint in ("hip", "thigh", "calf"))
TEMP_BANDS = {"cell": (38.0, 45.0), "joint": (55.0, 70.0), "chassis": (60.0, 75.0), "core": (88.0, 95.0)}

def body_packet(data):
    """The body-state summary exactly as chat-manager.py @ 8301af8 formats it, from one lowstate record."""
    bms=data.get("bms_state") or {}; sport=data.get("sport_state") or {}
    battery_pct=float(bms["soc"]) if bms.get("soc") is not None else \
        max(0.0,min(100.0,((data.get("power_v",22.0)-22.0)/(29.6-22.0))*100))
    vel=sport.get("velocity") or [0.0,0.0]; speed=math.hypot(vel[0],vel[1]) if len(vel)>1 else 0.0
    stance="standing" if (sport.get("body_height") or 0.0)>0.2 else "lying down"
    motion="still" if speed<0.05 else f"moving at {speed:.2f} m/s"
    readings=[]
    if bms.get("bq_ntc"): readings.append((float(max(bms["bq_ntc"])),"power cell","cell"))
    motors=(data.get("motor_state") or [])[:12]
    if len(motors)==12:
        temps=[m.get("temperature",0) for m in motors]
        hot=max(range(12),key=lambda i:temps[i]); cold=min(range(12),key=lambda i:temps[i])
        readings.append((float(temps[cold]),f"coolest joint {JOINT_NAMES[cold]}","joint"))
        readings.append((float(temps[hot]),f"warmest joint {JOINT_NAMES[hot]}","joint"))
    if data.get("temperature_ntc1"): readings.append((float(data["temperature_ntc1"]),"chassis","chassis"))
    if sport.get("imu_temperature"): readings.append((float(sport["imu_temperature"]),"sensing core","core"))
    warmth=""
    if readings:
        readings.sort(); items=[]; running_hot=[]
        for value,label,kind in readings:
            warm_above,hot_above=TEMP_BANDS[kind]
            verdict="hot" if value>hot_above else ("warm" if value>warm_above else "cool")
            if verdict=="hot": running_hot.append(label)
            if kind=="core" and verdict=="cool": items.append(f"{label} at its usual {value:.0f}°C")
            elif kind=="cell" or verdict!="cool": items.append(f"{label} {verdict} at {value:.0f}°C")
            else: items.append(f"{label} {value:.0f}°C")
        head=f"{', '.join(running_hot)} running hot" if running_hot else "nothing running hot"
        warmth=f", {head}: {', '.join(items)}"
    return f"charge {battery_pct:.0f}%, {stance}, {motion}{warmth}"

def resolve_system_prompt(spec, cfg):
    """Which system prompt to replay with, and a short provenance label.
    None -> the logged config.system_prompt. chat-manager replaces that with the performance script's
    "Default" scene at startup and on every reset, and never logs the replacement, so a performance-script
    JSON (its Default scene is used) or a plain text file can be given instead."""
    if not spec: return cfg["system_prompt"].strip(), "logged config.system_prompt"
    if ".json" in os.path.basename(spec):
        d=json.load(open(spec)); scenes=d if isinstance(d,list) else d.get("scenes",[])
        sc=next((x for x in scenes if x.get("name")=="Default"),None)
        if sc is None: sys.exit(f"{spec} has no Default scene")
        return sc["system_prompt"].strip(), f"{os.path.basename(spec)} Default scene"
    return open(spec).read().strip(), os.path.basename(spec)

def build_replay(session_dir, cfg, U, system_prompt=None):
    """Rebuild each prompt as chat-manager assembled it. Returns (exchanges, off_model_indices)."""
    for name in ("session-complete.jsonl","session.jsonl"):
        p=os.path.join(session_dir,"transcript",name)
        if os.path.exists(p): break
    vlm=[]
    for l in open(p):
        if not l.strip(): continue
        r=json.loads(l)
        if r.get("type")=="vlm_query" and r.get("description"):
            t=cg.parse_ts(r["timestamp"])
            if t: vlm.append((t+dt.timedelta(seconds=float(r.get("duration_seconds") or 0)), r["description"].strip()))
    vlm.sort()
    LIMIT=cfg["history_truncation_limit"]; sysmsg={"role":"system","content":(system_prompt or cfg["system_prompt"]).strip()}
    messages=[sysmsg]; ex=[]; off=[]
    for k,u in enumerate(U):
        nxt=U[k+1] if k+1<len(U) else None
        if u["type"]=="cue" and u.get("cue")=="reset":
            messages=[sysmsg]; off.append(u["i"]); continue
        if u["type"]=="user" and nxt is not None and nxt["type"]=="assistant":
            messages=[m for m in messages if not (m["role"]=="system" and
                      (m["content"].startswith(VISUAL_CONTEXT_PREFIX) or m["content"].startswith(BODY_STATE_PREFIX)))]
            messages.append({"role":"user","content":u["logged"] or u["text"],"_i":u["i"]})
            seen=[d for t,d in vlm if t<=u["t"]]
            vis=seen[-1] if seen else None
            if vis: messages.append({"role":"system","content":f"{VISUAL_CONTEXT_PREFIX} {vis}","_i":None})
            if len(messages)>LIMIT: messages=[messages[0]]+messages[-(LIMIT-1):]
            ex.append(dict(user_i=u["i"],asst_i=nxt["i"],msgs=[dict(m) for m in messages],
                           response=nxt["logged"] or nxt["text"],vis=vis))
            messages.append({"role":"assistant","content":nxt["logged"] or nxt["text"],"_i":nxt["i"]})
        elif u["type"]!="assistant":
            off.append(u["i"])
    return ex, off

def _tmpl(msgs):
    """Only role and content reach the chat template."""
    return [{"role":m["role"],"content":m["content"]} for m in msgs]

def spans_for(rendered, parts):
    """Char spans of each content string, searched right-to-left so repeats in history can't match."""
    out=[]; right=len(rendered)
    for text in reversed(parts):
        s=rendered.rfind(text,0,right)
        if s<0: raise ValueError(f"content not found in rendered prompt: {text[:60]!r}")
        out.append((s,s+len(text))); right=s
    return out[::-1]

def extract(ex, model_dir, packet):
    import torch
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
    dev="mps" if torch.backends.mps.is_available() else "cpu"
    tok=AutoTokenizer.from_pretrained(model_dir)
    model=Gemma4ForConditionalGeneration.from_pretrained(model_dir, dtype=torch.bfloat16).to(dev).eval()
    body={"role":"system","content":f"{BODY_STATE_PREFIX} {packet}"} if packet else None
    res={k:[] for k in ("user","asst","vis","asst_body","user_body")}; ntok=[]
    with torch.inference_mode():
        for n,x in enumerate(ex):
            for variant in ("logged","body") if body else ("logged",):
                msgs=[dict(m) for m in x["msgs"]]
                if variant=="body":                      # packet sits after the visual context, as injected
                    msgs.append(dict(body))
                msgs.append({"role":"assistant","content":x["response"]})
                rendered=tok.apply_chat_template(_tmpl(msgs),tokenize=False)
                parts=[x["msgs"][-2 if x["vis"] else -1]["content"]]+([x["msgs"][-1]["content"]] if x["vis"] else [])
                parts+=([body["content"]] if variant=="body" else [])+[x["response"]]
                sp=spans_for(rendered,parts)
                enc=tok(rendered,add_special_tokens=False,return_offsets_mapping=True)
                ids=enc["input_ids"]; offs=enc["offset_mapping"]
                assert ids[0]==tok.bos_token_id, "expected the template's <bos> as token 0"
                idx=[[t for t,(a,b) in enumerate(offs) if b>a and a<ce and b>cs] for cs,ce in sp]
                out=model(input_ids=torch.tensor([ids],device=dev),output_hidden_states=True)
                H=torch.stack(out.hidden_states,0)[:,0].float().cpu().numpy()   # [layers+1, T, d]
                pooled=[H[:,ii,:].mean(1) for ii in idx]
                if variant=="logged":
                    res["user"].append(pooled[0]); res["asst"].append(pooled[-1])
                    res["vis"].append(pooled[1] if x["vis"] else np.full_like(pooled[0],np.nan))
                    ntok.append((len(ids),len(idx[0]),len(idx[-1])))
                else:
                    res["user_body"].append(pooled[0]); res["asst_body"].append(pooled[-1])
            print(f"  exchange {n+1:>2}/{len(ex)}  turns {x['user_i']:>2}->{x['asst_i']:>2}  "
                  f"{ntok[-1][0]} tok (user {ntok[-1][1]}, reply {ntok[-1][2]})", flush=True)
    return {k:np.stack(v).astype(np.float32) for k,v in res.items() if v}, np.array(ntok)

def std_frame(M):
    """Per-dimension z-score fitted on the dialogue utterances: tames the late layers' outlier channels."""
    mu=M.mean(0); sd=M.std(0); sd=np.maximum(sd,1e-3*np.median(sd))
    return lambda x:(x-mu)/sd

def cosdist(Z):
    Zn=Z/np.linalg.norm(Z,axis=1,keepdims=True); return np.clip(1.0-Zn@Zn.T,0,2)

def mds2(D):
    from sklearn.manifold import MDS
    return MDS(n_components=2,dissimilarity="precomputed",random_state=7,n_init=16,
               max_iter=1500,normalized_stress="auto").fit_transform(D)

def align(P,ref):
    """Rotate/reflect/scale P onto ref so the same utterance sits in the same place across depths."""
    from scipy.linalg import orthogonal_procrustes
    a=P-P.mean(0); b=ref-ref.mean(0); Rm,_=orthogonal_procrustes(a,b); a=a@Rm
    return a*(np.linalg.norm(b)/max(np.linalg.norm(a),1e-9))+ref.mean(0)

def analyse(z,U,layer,small_layers):
    from sklearn.metrics import silhouette_score
    from scipy.stats import spearmanr
    from scipy.spatial.distance import pdist
    vec={}
    for n,i in enumerate(z["user_i"]): vec[int(i)]=z["user"][n]
    for n,i in enumerate(z["asst_i"]): vec[int(i)]=z["asst"][n]
    order=sorted(vec); X=np.stack([vec[i] for i in order]); nL=X.shape[1]
    hum=np.array([U[i]["speaker"]=="HUMAN" for i in order]); iu=np.triu_indices(len(order),1)
    pseudo=[dict(i=k,text=U[i]["logged"] or U[i]["text"],block=U[i]["block"],speaker=U[i]["speaker"],
                 type=U[i]["type"]) for k,i in enumerate(order)]
    _,tsim,_=cg.embed(pseudo,1.01); tdist=(1-tsim)[iu]
    sil,rho,shift,D={},{},{},{}
    for L in range(nL):
        f=std_frame(X[:,L]); Z=f(X[:,L]); D[L]=cosdist(Z)
        sil[L]=float(silhouette_score(Z,hum,metric="cosine"))
        rho[L]=float(spearmanr(D[L][iu],tdist).statistic)
        if "asst_body" in z.files:
            a=f(z["asst"][:,L]); b=f(z["asst_body"][:,L])
            sh=np.mean([1-np.dot(p,q)/np.linalg.norm(p)/np.linalg.norm(q) for p,q in zip(a,b)])
            shift[L]=float(sh/np.mean(pdist(a,"cosine")))
    P=mds2(D[layer]); P=P-P.mean(0); _,_,vt=np.linalg.svd(P,full_matrices=False); P=P@vt.T
    b1=[k for k,i in enumerate(order) if U[i].get("block")==1]
    if b1 and P[b1,0].mean()>0: P[:,0]*=-1
    if P[hum,1].mean()<0: P[:,1]*=-1
    fid=float(spearmanr(pdist(P),D[layer][iu]).statistic)
    small={L:align(mds2(D[L]),P) for L in small_layers}
    # how much each reply resembles the visual packet it was handed, in the same frame.
    # (Placing the packets themselves in the plane is misleading: they are near-duplicates of
    # one another, nearest one reply, and equidistant from the rest, so MDS parks them centrally.)
    f=std_frame(X[:,layer]); cos=lambda a,b: float(np.dot(a,b)/np.linalg.norm(a)/np.linalg.norm(b))
    scenes={int(ai):cos(f(z["asst"][n,layer]),f(z["vis"][n,layer]))
            for n,ai in enumerate(z["asst_i"]) if not np.isnan(z["vis"][n,layer]).all()}
    sim=1-D[layer]
    kind=lambda i:"H" if U[i]["speaker"]=="HUMAN" else "S"
    pairs=sorted(((sim[a,b],order[a],order[b]) for a in range(len(order)) for b in range(a+1,len(order))
                  if order[b]-order[a]>=2), reverse=True)
    echoes=[(i,j,float(s),kind(i)+kind(j)) for s,i,j in pairs[:30]]
    return dict(order=order,P=P,small=small,sil=sil,rho=rho,shift=shift,fid=fid,scenes=scenes,
                echoes=echoes,nL=nL,dspread=float(np.median(D[layer][iu])))

GYB=930                                   # graph area ends here; the off-model strip sits below
WHY={"startup":"canned greeting","cue":"canned reset audio"}
LBODY="#15803d"; LLEX="#a16207"

def render(cfg,U,resets,r,layer,session,kv_from,model_name,packet,sp_src="logged config.system_prompt"):
    from conversation_graph import (W,H,GX0,GX1,GY0,GY1,RX0,RX1,INK,MUT,FAINT,BG,HUM,DOG,SYS,VIS,RST,
                                    ramp,e,place_labels)
    resets=set(resets); order=r["order"]; P=r["P"]; o=[]; A=o.append
    pos={}
    lo,hi=P.min(0),P.max(0); span=np.maximum(hi-lo,1e-9); pad=60
    for k,i in enumerate(order):
        q=(P[k]-lo)/span; pos[i]=(GX0+pad+q[0]*((GX1-GX0)-2*pad), GYB-pad-q[1]*((GYB-GY0)-2*pad))
    rad={u["i"]:6.5+3.3*math.sqrt(u["dur"] or 1.0) for u in U}
    kind=lambda i:"H" if U[i]["speaker"]=="HUMAN" else "S"; col={"H":HUM,"S":DOG}
    say=lambda i:U[i]["logged"] or U[i]["text"]
    t0,t1=U[0]["t"],U[-1]["t"]; sil=r["sil"]; peak=max(sil,key=sil.get)

    A(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Helvetica Neue, Helvetica, Arial, sans-serif">')
    A(f'<rect width="{W}" height="{H}" fill="{BG}"/><defs>')
    for k in range(6):
        A(f'<marker id="ar{k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.2" markerHeight="6.2" orient="auto-start-reverse"><path d="M0,1.4 L9,5 L0,8.6 z" fill="{ramp(k/5)}"/></marker>')
    A(f'<marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.2" markerHeight="6.2" orient="auto-start-reverse"><path d="M0,1.4 L9,5 L0,8.6 z" fill="{RST}"/></marker></defs>')
    A(f'<text x="62" y="52" font-size="27" font-weight="700" fill="{INK}" letter-spacing="0.4">SNAPPER  &#215;  ROBERT</text>')
    A(f'<text x="62" y="76" font-size="12.5" fill="{MUT}">{e(session)} &#183; {t0:%Y-%m-%d}, {t0:%H:%M}&#8211;{t1:%H:%M} &#183; replayed through {e(model_name)} (bf16) &#183; {len(order)} of {len(U)} utterances entered the model</text>')
    A(f'<line x1="62" y1="96" x2="{RX1}" y2="96" stroke="{FAINT}"/>')

    # ---------------- main panel: the dialogue at one depth ----------------
    A(f'<rect x="{GX0}" y="{GY0}" width="{GX1-GX0}" height="{GY1-GY0}" fill="#ffffff" stroke="{FAINT}"/>')
    A(f'<text x="{GX0+14}" y="{GY0-8}" font-size="11.5" font-weight="700" fill="{INK}" letter-spacing="1.2">ACTIVATION SPACE &#183; LAYER {layer} OF {r["nL"]-1}</text>')
    A(f'<text x="{GX0+306}" y="{GY0-8}" font-size="11" fill="{MUT}">in-context hidden state, mean over each utterance&#8217;s tokens &#8594; z-score &#8594; cosine &#8594; MDS &#183; area &#8733; spoken duration</text>')
    ech=r["echoes"]; smin,smax=min(x[2] for x in ech),max(x[2] for x in ech)
    for i,j,s,k in ech:
        (x1,y1),(x2,y2)=pos[i],pos[j]
        mx,my=(x1+x2)/2,(y1+y2)/2; dx,dy=x2-x1,y2-y1; L=math.hypot(dx,dy) or 1
        cx,cy=mx-dy/L*L*0.20, my+dx/L*L*0.20; f=(s-smin)/max(1e-6,smax-smin)
        c=DOG if k=="SS" else (HUM if k=="HH" else VIS)
        A(f'<path d="M{x1:.1f},{y1:.1f} Q{cx:.1f},{cy:.1f} {x2:.1f},{y2:.1f}" fill="none" stroke="{c}" stroke-width="{0.9+1.6*f:.2f}" stroke-opacity="{0.20+0.38*f:.2f}" stroke-dasharray="3.5 3"/>')
    for a in range(len(order)-1):
        i,j=order[a],order[a+1]; (x1,y1),(x2,y2)=pos[i],pos[j]
        dx,dy=x2-x1,y2-y1; L=math.hypot(dx,dy) or 1; ux,uy=dx/L,dy/L
        x1s,y1s=x1+ux*(rad[i]+2.5),y1+uy*(rad[i]+2.5); x2s,y2s=x2-ux*(rad[j]+6.5),y2-uy*(rad[j]+6.5)
        mx,my=(x1s+x2s)/2,(y1s+y2s)/2; cx,cy=mx-uy*L*0.09,my+ux*L*0.09
        if any(i<c<j for c in resets):
            A(f'<path d="M{x1s:.1f},{y1s:.1f} Q{cx:.1f},{cy:.1f} {x2s:.1f},{y2s:.1f}" fill="none" stroke="{RST}" stroke-width="1.5" stroke-opacity="0.8" stroke-dasharray="2 5" marker-end="url(#arr)"/>')
        else:
            f=a/max(1,len(order)-2)
            A(f'<path d="M{x1s:.1f},{y1s:.1f} Q{cx:.1f},{cy:.1f} {x2s:.1f},{y2s:.1f}" fill="none" stroke="{ramp(f)}" stroke-width="0.9" stroke-opacity="0.6" marker-end="url(#ar{min(5,int(f*6))})"/>')
    lab=place_labels([(i,pos[i][0],pos[i][1],rad[i]+(5 if i in r["scenes"] else 0),say(i)) for i in order],
                     [(i,pos[i][0],pos[i][1],rad[i]+(5 if i in r["scenes"] else 0)) for i in order],GX0,GX1,GY0,GYB-44)
    for i,lx,ly,right,t in lab:
        A(f'<line x1="{pos[i][0]:.1f}" y1="{pos[i][1]:.1f}" x2="{lx:.1f}" y2="{ly-3.4:.1f}" stroke="{FAINT}" stroke-width="0.8"/>')
        A(f'<text x="{lx:.1f}" y="{ly:.1f}" font-size="9.3" fill="{INK}" fill-opacity="0.85" text-anchor="{"start" if right else "end"}">{e(t)}</text>')
    for i in order:
        x,y=pos[i]; c=col[kind(i)]
        sc=r["scenes"].get(i)
        if sc is not None and sc>0.03:
            A(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{rad[i]+4.5:.1f}" fill="none" stroke="{VIS}" stroke-width="{1.2+2.8*min(sc,0.6)/0.6:.2f}" stroke-opacity="{0.25+0.7*min(sc,0.6)/0.6:.2f}"/>')
        A(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{rad[i]:.1f}" fill="{c}" fill-opacity="0.88" stroke="{c}" stroke-width="1.4"/>')
        A(f'<text x="{x:.1f}" y="{y+3.2:.1f}" font-size="8.2" text-anchor="middle" font-weight="700" fill="#ffffff" font-family="SF Mono, Menlo, monospace">{i}</text>')
    A(f'<g font-size="10.4" fill="{MUT}"><text x="{GX0+14}" y="{GYB-26}">speaker separation here <tspan font-weight="700" fill="{INK}">{sil[layer]:.2f}</tspan> (silhouette) &#183; peak <tspan font-weight="700" fill="{INK}">{sil[peak]:.2f}</tspan> at layer {peak} &#183; the plane keeps the full-dimension distance ranks at &#961; = {r["fid"]:.2f}</text>')
    A(f'<text x="{GX0+14}" y="{GYB-10}">dashed arcs: the 30 most similar non-adjacent pairs &#8212; {sum(1 for x in ech if x[3]=="HH")} question&#8211;question, {sum(1 for x in ech if x[3]=="SS")} reply&#8211;reply, {sum(1 for x in ech if x[3] in ("HS","SH"))} across speakers</text></g>')
    # off-model strip
    offs=[u for u in U if u["i"] not in set(order)]
    A(f'<line x1="{GX0}" y1="{GYB+8}" x2="{GX1}" y2="{GYB+8}" stroke="{FAINT}"/>')
    A(f'<text x="{GX0+14}" y="{GYB+30}" font-size="9.6" font-weight="700" fill="{MUT}" letter-spacing="0.9">NEVER ENTERED</text>')
    A(f'<text x="{GX0+14}" y="{GYB+43}" font-size="9.6" font-weight="700" fill="{MUT}" letter-spacing="0.9">THE MODEL</text>')
    step=(GX1-GX0-150)/max(1,len(offs))
    for n,u in enumerate(offs):
        x=GX0+150+n*step+10; y=GYB+36; c=HUM if u["speaker"]=="HUMAN" else SYS
        why=WHY.get(u["type"]) or ("reset request, intercepted" if u["type"]=="user" and any(u["i"]<c_<=u["i"]+1 for c_ in resets) else "no reply logged")
        t=u["text"]; t=(t[:22].rstrip()+"…") if len(t)>23 else t
        A(f'<circle cx="{x:.1f}" cy="{y-3:.1f}" r="8" fill="none" stroke="{c}" stroke-width="1.4" stroke-dasharray="2.5 2"/>')
        A(f'<text x="{x:.1f}" y="{y:.1f}" font-size="8" text-anchor="middle" font-weight="700" fill="{c}" font-family="SF Mono, Menlo, monospace">{u["i"]}</text>')
        A(f'<text x="{x+13:.1f}" y="{y-5:.1f}" font-size="9.2" fill="{INK}" fill-opacity="0.8">{e(t)}</text>')
        A(f'<text x="{x+13:.1f}" y="{y+7:.1f}" font-size="8.6" fill="{MUT}" font-style="italic">{e(why)}</text>')

    # ---------------- rail: depth ----------------
    A(f'<line x1="{RX0-24}" y1="132" x2="{RX0-24}" y2="1010" stroke="{FAINT}"/>')
    A(f'<text x="{RX0}" y="124" font-size="11.5" font-weight="700" fill="{INK}" letter-spacing="1.2">DEPTH</text>')
    A(f'<text x="{RX0}" y="150" font-size="10.6" fill="{MUT}">the same {len(order)} utterances at six depths, aligned to layer {layer}</text>')
    bw,bh=236,138
    for n,(L,Q) in enumerate(sorted(r["small"].items())):
        bx=RX0+(n%2)*(bw+24); by=164+(n//2)*(bh+18)
        A(f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" fill="#ffffff" stroke="{FAINT}"/>')
        q=(Q-lo)/span; pts=[(bx+12+min(max(v[0],-.05),1.05)*(bw-24), by+bh-10-min(max(v[1],-.05),1.05)*(bh-30)) for v in q]
        A(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x,y in pts)}" fill="none" stroke="{INK}" stroke-width="0.6" stroke-opacity="0.22"/>')
        for (x,y),i in zip(pts,order):
            A(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.1" fill="{col[kind(i)]}" fill-opacity="0.85"/>')
        name="token embeddings" if L==0 else ("shown at left" if L==layer else "")
        A(f'<text x="{bx+8}" y="{by+15}" font-size="9.6" font-weight="700" fill="{INK}">LAYER {L}<tspan font-weight="400" fill="{MUT}">{(" &#183; "+name) if name else ""}</tspan></text>')
        A(f'<text x="{bx+bw-8}" y="{by+15}" font-size="9.4" fill="{MUT}" text-anchor="end" font-family="SF Mono, Menlo, monospace">sep {sil[L]:.2f}</text>')
    # depth profile
    cx0,cx1,cy0,cy1=RX0+30,RX1-10,672,858; nL=r["nL"]; ymax=0.5
    X_=lambda L:cx0+(cx1-cx0)*L/(nL-1); Y_=lambda v:cy1-(cy1-cy0)*max(0.0,min(v,ymax))/ymax
    A(f'<text x="{RX0}" y="{cy0-14}" font-size="10.4" font-weight="700" fill="{INK}" letter-spacing="1.1">PROFILE ACROSS ALL {nL} DEPTHS</text>')
    if kv_from is not None:
        A(f'<rect x="{X_(kv_from-0.5):.1f}" y="{cy0}" width="{X_(nL-1)-X_(kv_from-0.5):.1f}" height="{cy1-cy0}" fill="{FAINT}" fill-opacity="0.45"/>')
        A(f'<text x="{X_(nL-1)-4:.1f}" y="{cy0+11}" font-size="8.6" fill="{MUT}" text-anchor="end">layers {kv_from}&#8211;{nL-1} reuse earlier keys/values</text>')
    for v in (0,0.1,0.2,0.3,0.4,0.5):
        A(f'<line x1="{cx0}" y1="{Y_(v):.1f}" x2="{cx1}" y2="{Y_(v):.1f}" stroke="{FAINT}" stroke-width="0.6"/>')
        A(f'<text x="{cx0-5}" y="{Y_(v)+3:.1f}" font-size="8.4" fill="{MUT}" text-anchor="end" font-family="SF Mono, Menlo, monospace">{v:.1f}</text>')
    for L in range(0,nL,5):
        A(f'<text x="{X_(L):.1f}" y="{cy1+13}" font-size="8.4" fill="{MUT}" text-anchor="middle" font-family="SF Mono, Menlo, monospace">{L}</text>')
    A(f'<line x1="{X_(layer):.1f}" y1="{cy0}" x2="{X_(layer):.1f}" y2="{cy1}" stroke="{INK}" stroke-width="0.8" stroke-dasharray="2 2"/>')
    for series,c,dash,wid in ((r["rho"],LLEX,"4 3",1.4),(r["shift"],LBODY,"1.5 2.5",1.6),(sil,INK,"",1.9)):
        if not series: continue
        A(f'<polyline points="{" ".join(f"{X_(L):.1f},{Y_(v):.1f}" for L,v in sorted(series.items()))}" fill="none" stroke="{c}" stroke-width="{wid}"{f" stroke-dasharray=\"{dash}\"" if dash else ""}/>')
    A(f'<circle cx="{X_(peak):.1f}" cy="{Y_(sil[peak]):.1f}" r="3" fill="{INK}"/>')
    ly=cy1+30
    for n,(c,dash,txt) in enumerate(((INK,"","speaker separation (silhouette)"),(LLEX,"4 3","agreement with the TF-IDF space (Spearman &#961;)"),
                                     (LBODY,"1.5 2.5","shift from adding a body-state packet, &#215; reply spread"))):
        if n==2 and not r["shift"]: continue
        A(f'<line x1="{RX0+8}" y1="{ly+n*15-3.5}" x2="{RX0+30}" y2="{ly+n*15-3.5}" stroke="{c}" stroke-width="1.8"{f" stroke-dasharray=\"{dash}\"" if dash else ""}/>')
        A(f'<text x="{RX0+38}" y="{ly+n*15}" font-size="9.8" fill="{INK}" fill-opacity="0.85">{txt}</text>')
    # key
    KY=ly+56
    A(f'<line x1="{RX0}" y1="{KY-14}" x2="{RX1}" y2="{KY-14}" stroke="{FAINT}"/>')
    items=[(HUM,"dot","ROBERT &#8212; question as the model read it"),(DOG,"dot","SNAPPER &#8212; its reply"),
           (VIS,"ring","reply resembles the scene it was shown"),(RST,"dash","context wipe between replies"),
           (INK,"ramp","dialogue order &#8212; pale first, dark last"),(DOG,"arc","echo &#8212; one of the 30 nearest pairs")]
    for n,(c,shape,txt) in enumerate(items):
        x=RX0+8+(n%2)*250; y=KY+4+(n//2)*17
        if shape=="dot": A(f'<circle cx="{x+4}" cy="{y-3.5}" r="5" fill="{c}" fill-opacity="0.88"/>')
        elif shape=="ring": A(f'<circle cx="{x+4}" cy="{y-3.5}" r="5.5" fill="none" stroke="{c}" stroke-width="2.2" stroke-opacity="0.8"/>')
        elif shape=="ramp":
            for q_ in range(6): A(f'<line x1="{x-4+q_*2.9:.1f}" y1="{y-3.5}" x2="{x-4+(q_+1)*2.9:.1f}" y2="{y-3.5}" stroke="{ramp(q_/5)}" stroke-width="1.6"/>')
        elif shape=="arc": A(f'<path d="M{x-4},{y-1} Q{x+4},{y-11} {x+12},{y-1}" fill="none" stroke="{c}" stroke-width="1.4" stroke-opacity="0.55" stroke-dasharray="3.5 3"/>')
        else: A(f'<line x1="{x-4}" y1="{y-3.5}" x2="{x+12}" y2="{y-3.5}" stroke="{c}" stroke-width="1.6" stroke-dasharray="2 4"/>')
        A(f'<text x="{x+18}" y="{y}" font-size="9.8" fill="{INK}" fill-opacity="0.85">{txt}</text>')

    # ---------------- footer ----------------
    A(f'<line x1="62" y1="1042" x2="{RX1}" y2="1042" stroke="{FAINT}"/>')
    mx_shift=max(r["shift"].values()) if r["shift"] else None
    foot=[f'Each of the {len(order)//2} prompts rebuilt as chat-manager.py @ 8301af8 assembled it &#8212; system prompt, history, the live transcription the model actually received, the visual-context packet &#8212; with the logged reply teacher-forced through {e(model_name)} under Google&#8217;s chat template.',
          f'System prompt: {e(sp_src)}. The dog&#8217;s logs record only the config prompt, which a performance-script Default scene replaces at startup and on every reset; the replay uses the candidate that best explains Snapper&#8217;s words.',
          f'The dog ran the Q4_K_M quantization of these weights through Ollama&#8217;s built-in renderer. Body state was injected live but never logged; adding a representative packet (the deployed formatter on 12:31 telemetry)'
          +(f' moves the replies by at most {100*mx_shift:.0f}% of the typical reply-to-reply distance.' if mx_shift is not None else ' could not be tested: no telemetry.'),
          f'In-context activations carry position in the conversation as well as content: two first questions after a reset resemble each other partly because both follow an empty history. Labels show the text the model received.']
    for n,t in enumerate(foot): A(f'<text x="62" y="{1064+n*17}" font-size="10" fill="{MUT}">{t}</text>')
    A(f'<text x="62" y="{1064+len(foot)*17+6}" font-size="10" fill="{MUT}">Robert Twomey &#183; BFF / Dog Walk &#183; rendered {dt.date.today():%Y-%m-%d}</text>')
    A("</svg>")
    return o

def first_lowstate(session_dir):
    tdir=os.path.join(session_dir,"telemetry")
    if not os.path.isdir(tdir): return None
    chunks=sorted((d for d in os.listdir(tdir) if d.startswith("chunk_")),key=lambda d:int(d.split("_")[1]))
    for c in chunks:
        p=os.path.join(tdir,c,"lowstate.jsonl")
        if os.path.exists(p):
            with open(p) as f: return json.loads(f.readline()).get("data")
    return None

def main():
    ap=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_dir"); ap.add_argument("--model", required=True, help="local google/gemma-4-E2B-it snapshot")
    ap.add_argument("-o","--out", default=None); ap.add_argument("--layer", type=int, default=18,
                    help="depth shown in the main panel (default 18, mid-stack)")
    ap.add_argument("--reextract", action="store_true", help="ignore the cached activations and rerun the model")
    ap.add_argument("--system-prompt", default=None, metavar="SCRIPT_OR_TXT",
                    help="replay with this system prompt instead of the logged one: a performance-script JSON "
                         "(its Default scene) or a text file")
    ap.add_argument("--png", action="store_true", help="also export a PNG next to the SVG")
    ap.add_argument("--png-width", type=int, default=3440)
    a=ap.parse_args()
    cfg,U,resets,_=cg.load(a.session_dir)
    sp,sp_src=resolve_system_prompt(a.system_prompt,cfg)
    ex,off=build_replay(a.session_dir,cfg,U,system_prompt=sp)
    session=os.path.basename(a.session_dir.rstrip("/"))
    if "__" in session: session=[s for s in session.split("__") if s.startswith("session-")][0]
    out=a.out or f"{session}-activation-graph.svg"; cache=os.path.splitext(out)[0]+".acts.npz"
    stale=False
    if os.path.exists(cache) and not a.reextract:
        z=np.load(cache)
        if list(z["user_i"])!=[x["user_i"] for x in ex]: sys.exit(f"{cache} is from a different replay; pass --reextract")
        stale=(str(z["system_prompt"]) if "system_prompt" in z.files else cfg["system_prompt"].strip())!=sp
        if stale: print(f"{cache} was built with a different system prompt; replaying again")
        else: print(f"using cached activations {cache}")
    if a.reextract or stale or not os.path.exists(cache):
        ls=first_lowstate(a.session_dir); packet=body_packet(ls) if ls else None
        print(f"replaying {len(ex)} exchanges through {a.model} with the {sp_src}")
        acts,ntok=extract(ex,a.model,packet)
        np.savez_compressed(cache,**acts,ntok=ntok,user_i=[x["user_i"] for x in ex],
                            asst_i=[x["asst_i"] for x in ex],off=off,packet=packet or "",
                            system_prompt=sp,system_prompt_source=sp_src)
        z=np.load(cache)
    tc=json.load(open(os.path.join(a.model,"config.json"))); tc=tc.get("text_config",tc)
    nl,kv=tc.get("num_hidden_layers"),tc.get("num_kv_shared_layers")
    kv_from=(nl-kv+1) if nl and kv else None
    nL=z["user"].shape[1]; layer=max(0,min(a.layer,nL-1))
    small=sorted({0,8,15,17,26,nL-1})       # embeddings, the separation peak and the drop after it, the top
    r=analyse(z,U,layer,small)
    name=os.path.basename(a.model.rstrip("/")); name=f"google/{name}" if not name.startswith("google/") else name
    o=render(cfg,U,resets,r,layer,session,kv_from,name,str(z["packet"]),sp_src)
    open(out,"w").write("\n".join(o))
    pk=max(r["sil"],key=r["sil"].get)
    print(f"{out}  (layer {layer}: separation {r['sil'][layer]:.2f}, peak {r['sil'][pk]:.2f} at layer {pk}; plane rho {r['fid']:.2f})")
    if a.png:
        png=os.path.splitext(out)[0]+".png"; tool=cg.export_png(out,png,a.png_width)
        print(f"{png}  ({a.png_width} px wide, via {tool})")

if __name__=="__main__": main()

def score_prompts(ex, model_dir, candidates):
    """How surprising Snapper's logged replies are under each candidate system prompt.
    candidates: [(label, text)]. Returns {label: [(mean bits/token, tokens over 20 bits), ...per exchange]}."""
    import torch
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
    dev="mps" if torch.backends.mps.is_available() else "cpu"
    tok=AutoTokenizer.from_pretrained(model_dir)
    model=Gemma4ForConditionalGeneration.from_pretrained(model_dir,dtype=torch.bfloat16).to(dev).eval()
    out={}
    with torch.inference_mode():
        for label,text in candidates:
            out[label]=[]
            for x in ex:
                msgs=[dict(m) for m in x["msgs"]]; msgs[0]["content"]=text
                msgs.append({"role":"assistant","content":x["response"]})
                r=tok.apply_chat_template(_tmpl(msgs),tokenize=False); (cs,ce),=spans_for(r,[x["response"]])
                enc=tok(r,add_special_tokens=False,return_offsets_mapping=True); ids=enc["input_ids"]
                q=[t for t,(a,b) in enumerate(enc["offset_mapping"]) if b>a and a<ce and b>cs]
                lp=torch.log_softmax(model(input_ids=torch.tensor([ids],device=dev)).logits[0].float(),-1)
                sb=np.array([-lp[t-1,ids[t]].item()/math.log(2) for t in q])
                out[label].append((float(sb.mean()),int((sb>20).sum())))
            print(f"  scored {label}: {np.mean([m for m,_ in out[label]]):.2f} bits/token", flush=True)
    return out

PARTS=("sink","scaffold","system","history","question","visual","reply")

def extract_attention_surprise(ex, model_dir):
    """One eager-attention replay per exchange (as logged, no body packet). Returns, per exchange:
    where the reply's attention went (share of mass by prompt part and by earlier turn, per layer,
    averaged over heads and over the reply's tokens), and per-token surprisal / entropy / the model's
    own top guess for Robert's question and Snapper's reply."""
    import torch
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
    dev="mps" if torch.backends.mps.is_available() else "cpu"
    tok=AutoTokenizer.from_pretrained(model_dir)
    model=Gemma4ForConditionalGeneration.from_pretrained(model_dir,dtype=torch.bfloat16,
                                                         attn_implementation="eager").to(dev).eval()
    ln2=math.log(2); res=[]
    with torch.inference_mode():
        for n,x in enumerate(ex):
            msgs=x["msgs"]+[{"role":"assistant","content":x["response"],"_i":x["asst_i"]}]
            rendered=tok.apply_chat_template(_tmpl(msgs),tokenize=False)
            spans=[]; cur=0
            for m in msgs:                                   # every message's content, left to right
                s_=rendered.find(m["content"],cur)
                if s_<0: raise ValueError(f"content not found: {m['content'][:60]!r}")
                spans.append((s_,s_+len(m["content"]))); cur=s_+len(m["content"])
            enc=tok(rendered,add_special_tokens=False,return_offsets_mapping=True)
            ids=enc["input_ids"]; offs=enc["offset_mapping"]; T=len(ids)
            owner=np.full(T,-1)
            for k,(cs,ce) in enumerate(spans):
                for t,(a,b) in enumerate(offs):
                    if b>a and a<ce and b>cs: owner[t]=k
            ir=len(msgs)-1; iv=ir-1 if x["vis"] else None; iq=ir-(2 if x["vis"] else 1)
            role_of=lambda k:"sink" if k==-2 else ("scaffold" if k<0 else "system" if k==0 else "reply" if k==ir
                              else "question" if k==iq else "visual" if k==iv else "history")
            part=np.array([role_of(-2 if t==0 else owner[t]) for t in range(T)])
            out=model(input_ids=torch.tensor([ids],device=dev),output_attentions=True)
            lp=torch.log_softmax(out.logits[0].float(),-1)
            ent=(-(lp.exp()*lp).sum(-1)/ln2).cpu().numpy()
            nxt=torch.tensor(ids[1:],device=dev)
            sur=(-lp[:-1].gather(1,nxt[:,None])[:,0]/ln2).cpu().numpy(); top=lp[:-1].argmax(-1).cpu().numpy()
            del lp
            def tokens(k):
                cs,ce=spans[k]; o=[]
                for t in np.where(owner==k)[0]:
                    a,b=offs[t]; o.append(dict(t=rendered[max(a,cs):min(b,ce)],s=float(sur[t-1]),h=float(ent[t-1]),
                                               top=tok.decode([int(top[t-1])])))
                return o
            rq=np.where(owner==ir)[0]; hist_k=[k for k in range(1,ir) if role_of(k)=="history"]
            shares=[]; hist={k:[] for k in hist_k}
            assert len(out.attentions)==model.config.text_config.num_hidden_layers
            for att in out.attentions:                       # [1, heads, T, T]
                row=att[0].float().mean(0)[rq].mean(0).cpu().numpy()
                shares.append([float(row[part==p_].sum()) for p_ in PARTS])
                for k in hist_k: hist[k].append(float(row[owner==k].sum()))
            res.append(dict(user_i=x["user_i"],asst_i=x["asst_i"],T=T,shares=shares,
                            history=[dict(i=msgs[k].get("_i"),role=msgs[k]["role"],share=hist[k]) for k in hist_k],
                            question=tokens(iq),reply=tokens(ir),has_visual=bool(x["vis"])))
            print(f"  exchange {n+1:>2}/{len(ex)}  {T} tok  reply attention to <bos> "
                  f"{np.mean([s_[0] for s_ in shares]):.2f}, to question {np.mean([s_[4] for s_ in shares]):.3f}", flush=True)
    return dict(parts=list(PARTS),layers=len(res[0]["shares"]),exchanges=res)
