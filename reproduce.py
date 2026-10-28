from pathlib import Path
import os
os.environ['MPLCONFIGDIR']='/tmp/n1-research-mpl'
import numpy as np,h5py,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
parser=argparse.ArgumentParser(description='Visualize the provided pretrained cardiovascular model on case 0517.')
parser.add_argument('--input',required=True,type=Path)
parser.add_argument('--output',default=Path('results'),type=Path)
args=parser.parse_args()
src=args.input
out=args.output;out.mkdir(parents=True,exist_ok=True)
weights={}
with h5py.File(src/'iter_model.h5') as f:
 for i in range(6):
  name='dense'+('_'+str(i) if i else '')
  g=f[f'{name}/point_net/{name}']
  weights[i]=(g['kernel:0'][:],g['bias:0'][:])
def dense(x,i,relu=True):
 w,b=weights[i];y=x@w+b
 return np.maximum(y,0) if relu else y
def read_csv(path):
 rows=[]
 for line in path.open():
  try:
   row=[float(v) for v in line.strip().split(',')]
   if len(row)>=3: rows.append(row)
  except ValueError: pass
 return np.asarray(rows,dtype=np.float32)
m=read_csv(src/'test_model_csv/0517.csv')
d=read_csv(src/'test_pv_csv/0517pv.csv')
x=d[:,:3];truth=d[:,4:7]
# Match model.py: shared first two dense layers, global max, then local/global concatenation.
global_feature=np.full(512,-np.inf,dtype=np.float32)
for start in range(0,len(m),4096):
 a=dense(dense(dense(m[start:start+4096],0),1),2)
 global_feature=np.maximum(global_feature,a.max(axis=0))
pred=np.empty_like(truth)
for start in range(0,len(x),4096):
 local=dense(dense(dense(x[start:start+4096],0),1),3)
 combined=np.concatenate([local,np.broadcast_to(global_feature,(len(local),512))],axis=1)
 pred[start:start+len(local)]=dense(dense(combined,4),5,False)
err=pred-truth
metrics={'sample':'0517','geometry_points':len(m),'query_points':len(x),'velocity_component_MAE_m_s':float(np.mean(abs(err))),'velocity_component_RMSE_m_s':float(np.sqrt(np.mean(err**2))),'inference':'NumPy float32 forward pass matching provided model.py and HDF5 weights; not TensorFlow runtime validated'}
(out/'metrics.json').write_text(json.dumps(metrics,indent=2))
# Shared physical scale; do not independently normalize reference/prediction.
speed=np.linalg.norm(truth,axis=1);pspeed=np.linalg.norm(pred,axis=1);ve=np.linalg.norm(err,axis=1)
idx=np.sort(np.random.default_rng(42).choice(len(x),min(45000,len(x)),replace=False))
fig=plt.figure(figsize=(15,8),layout='constrained',facecolor='#fafbf9')
vmax=max(speed.max(),pspeed.max())
for i,(values,title,cmap,hi) in enumerate([(speed,'CFD reference speed','viridis',vmax),(pspeed,'Predicted speed','viridis',vmax),(ve,'Velocity-vector error','magma',ve.max())]):
 ax=fig.add_subplot(1,3,i+1,projection='3d')
 sc=ax.scatter(*x[idx].T,c=values[idx],s=.6,cmap=cmap,vmin=0,vmax=hi,rasterized=True,alpha=.85)
 ax.view_init(elev=12,azim=-70);ax.set_box_aspect(np.maximum(np.ptp(x,axis=0),.001))
 ax.set_title(title,pad=15,fontsize=14)
 ax.set_xlabel('X (m)');ax.set_ylabel('Y (m)');ax.set_zlabel('Z (m)')
 ax.xaxis.set_major_locator(plt.MaxNLocator(3));ax.yaxis.set_major_locator(plt.MaxNLocator(3));ax.zaxis.set_major_locator(plt.MaxNLocator(4))
 fig.colorbar(sc,ax=ax,shrink=.5,pad=.03,label='m/s')
fig.suptitle('3D cardiovascular flow | Test case 0517',fontsize=20)
fig.text(.5,.02,f"All {len(x):,} query points evaluated; {len(idx):,} shown | Component MAE {metrics['velocity_component_MAE_m_s']:.5f} m/s | RMSE {metrics['velocity_component_RMSE_m_s']:.5f} m/s",ha='center',fontsize=11)
fig.savefig(out/'flow_prediction_0517.png',dpi=200)
print(json.dumps(metrics));print(out/'flow_prediction_0517.png')
