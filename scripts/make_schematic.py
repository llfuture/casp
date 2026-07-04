import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
FIG = os.path.expanduser("~/projects/casp_max/figures")
CB="#1b6ca8"; CR="#d1495b"; CG="#2a9d8f"; GRY="#6c757d"
fig, ax = plt.subplots(figsize=(10.6,4.7)); ax.set_xlim(0,100); ax.set_ylim(-5,45); ax.axis("off")
def box(x,y,w,h,t,fc,ec,fs=9.5,tc="black"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.4,rounding_size=1.2",fc=fc,ec=ec,lw=1.6))
    ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=fs,color=tc)
def arr(x1,y1,x2,y2,c=GRY,lab=None):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=16,color=c,lw=1.8,shrinkA=2,shrinkB=2))
    if lab: ax.text((x1+x2)/2,(y1+y2)/2+1.4,lab,ha="center",fontsize=8,color=c,style="italic")
ax.text(50,43.5,"Information-flow inversion: from predicting solutions to predicting verifiable certificates",ha="center",fontsize=11.5,fontweight="bold")
ax.text(1,39.5,"Positive signal (prior LAA): predict what to do",fontsize=11,color=CR,fontweight="bold")
box(1,28,17,8,"Predictor\n$\\mathcal{M}(I)$","#fff0f1",CR)
box(26,28,19,8,"candidate solution\n$\\hat S$  (do this)","#fff0f1",CR)
box(53,28,19,8,"commit $\\hat S$\n(no verifier)","#fde8ea",CR)
box(79,28,20,8,"ratio degrades with\nerror $\\eta$; no\noptimality proof","#fbdde0",CR,fs=8.8)
arr(18,32,26,32,CR); arr(45,32,53,32,CR); arr(72,32,79,32,CR)
ax.text(1,21.5,"Negative signal (CASP): predict what may be ignored, with a proof",fontsize=11,color=CB,fontweight="bold")
box(1,9,17,8,"Predictor\n$\\mathcal{M}(I)$","#eef6fb",CB)
box(26,9,19,8,"certificate $(\\phi,w)$\n(ignore region)","#eef6fb",CB)
box(53,9.5,14,7,"Verifier $V$\n(poly-time,\nsound)","#e7f5f1",CG,fs=9)
box(72,13,26,6.5,"PASS $\\Rightarrow$ prune, solve reduced","#e7f5f1",CG,fs=9)
box(72,4.5,26,6.5,"FAIL $\\Rightarrow$ classical fallback","#f1f1f1",GRY,fs=9)
arr(18,13,26,13,CB); arr(45,13,53,13,CB)
arr(67,14,72,16,CG,"valid"); arr(67,12,72,8,GRY,"rejected")
box(70,-3.2,29,5.6,"$\\Rightarrow$ bounded loss\n+ certified optimum","#dff0ea",CB,fs=9,tc=CB)
arr(85,4.5,85,2.6,CB)
fig.savefig(FIG+"/fig_idea.pdf",bbox_inches="tight"); plt.close(fig); print("wrote fig_idea.pdf")
