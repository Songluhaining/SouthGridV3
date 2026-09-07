"""稳健语言接地测试：对每色单按钮集抽早期帧，喂四色 prompt，
统计模型预测目标是否最接近'被指定颜色'的按钮。不需要仿真器。"""
import argparse, json, pathlib, random, collections
import av, numpy as np, pyarrow.parquet as pq
import openpi.training.config as _config
from openpi.policies import policy_config as _policy_config
CAMS=("observation.images.cam_head","observation.images.cam_wrist_r")
BTN={"red":None,"green":None,"blue":None,"yellow":None}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--ckpt",required=True); ap.add_argument("--dataset",required=True)
    ap.add_argument("--config",default="pi05_g1_button_lora"); ap.add_argument("--per_color",type=int,default=6)
    a=ap.parse_args(); ds=pathlib.Path(a.dataset)
    info=json.loads((ds/"meta/info.json").read_text()); cs,vtpl=int(info.get("chunks_size",1000)),info["video_path"]
    q=[json.loads(l) for l in (ds/"meta/quality.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    # 颜色→按钮位置(从质量记录的 cap_B 取中位)
    caps=collections.defaultdict(list); prompts={}
    for r in q:
        for p in r.get("presses",[]):
            if p.get("cap_B"): caps[p["color"]].append(p["cap_B"])
        if len(r.get("sequence",[]))==1 and r["sequence"][0] not in prompts:
            prompts[r["sequence"][0]]=r["task"]
    btn={c:np.median(np.array(v),axis=0) for c,v in caps.items()}
    colors=[c for c in ("red","green","blue","yellow") if c in btn and c in prompts]
    single=collections.defaultdict(list)
    for r in q:
        if r.get("kept") and len(r.get("sequence",[]))==1 and r.get("episode_index") is not None:
            single[r["sequence"][0]].append(r["episode_index"])
    pol=_policy_config.create_trained_policy(_config.get_config(a.config),a.ckpt)
    rnd=random.Random(0); correct=0; total=0; spreads=[]
    for tgt_color in colors:
        eps=single[tgt_color][:]; rnd.shuffle(eps)
        for ep in eps[:a.per_color]:
            ch=ep//cs
            st=np.stack(pq.read_table(ds/f"data/chunk-{ch:03d}/episode_{ep:06d}.parquet",columns=["observation.state"]).column("observation.state").to_pylist()).astype(np.float32)
            t=min(40,len(st)-11)  # 早期帧
            imgs={}
            for cam in CAMS:
                with av.open(str(ds/vtpl.format(episode_chunk=ch,video_key=cam,episode_index=ep))) as c:
                    for i,fr in enumerate(c.decode(c.streams.video[0])):
                        if i==t: imgs[cam.split(".")[-1]]=np.asarray(fr.to_image().convert("RGB")); break
            tips={}
            for c in colors:
                pr=np.asarray(pol.infer({"state":st[t],"images":imgs,"prompt":prompts[c]})["actions"])
                tips[c]=pr[-1,7:10]
            # 用指定颜色 prompt 的预测目标，看它离哪个按钮最近
            pred=tips[tgt_color]
            nearest=min(btn,key=lambda k:np.linalg.norm(pred-btn[k]))
            correct+=int(nearest==tgt_color); total+=1
            sp=max(np.linalg.norm(tips[x]-tips[y]) for x in tips for y in tips)
            spreads.append(sp)
    print(f"\n=== {a.ckpt.split('/')[-2]} ===")
    print(f"单色接地正确率: {correct}/{total} = {100*correct/total:.0f}%  (预测目标最接近被指定颜色的按钮)")
    print(f"四色 prompt 预测目标最大差(多帧均值): {np.mean(spreads)*1000:.1f} mm  (>50 语言起作用, <10 忽略)")
if __name__=="__main__": main()
