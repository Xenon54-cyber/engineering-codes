"""He--Xe: отпуск теплоты. Свип T входа реактора и суммарных потерь газа."""
from __future__ import annotations
from pathlib import Path
from multiprocessing import Pool
import os, sys, types
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.optimize import brentq
try: from tqdm.auto import tqdm
except ImportError: tqdm=None

ROOT=Path(__file__).resolve().parent; OUT=ROOT/'Результаты режима отпуск теплоты'; N=('D','H','S','Cp','Vis','Tcx'); W=None
Q0,ER,X,P0,T0,T1=2.75e6,.99,.283,4.8e6,1060+273.15,1030+273.15
PW,DPW,TW0,TW1,EPW,EMW=1.6e6,.05,40+273.15,70+273.15,.80,.95
EC,EMC,EMM=.8706,.95,.90; WT=(.2,.4,.4) # pipe, HX, reactor

def legacy():
 n='scipy._external.array_api_compat.numpy._aliases'
 if n not in sys.modules: m=types.ModuleType(n);m.asarray=np.asarray;sys.modules[n]=m
class P:
 def __init__(s,f,x):
  legacy();s.f=dict(zip(N,np.load(ROOT/f,allow_pickle=True)));s.x=x
  for z in s.f.values():
   if not hasattr(z,'grid') and hasattr(z,'_grid'):z.grid=z._grid
   if not hasattr(z,'values') and hasattr(z,'_values'):z.values=z._values
 def v(s,n,p,t): return float(np.asarray(s.f[n]((s.x,p,t))).squeeze())
 def inv(s,n,y,p):
  g=np.asarray(s.f[n].grid[2],float);f=lambda t:s.v(n,p,t)-y
  if f(g[0])*f(g[-1])<=0:return brentq(f,g[0],g[-1])
  a=np.array([f(t) for t in g]);i=np.flatnonzero(a[:-1]*a[1:]<=0)
  if not len(i):raise ValueError('свойство вне таблицы')
  return brentq(f,g[i[0]],g[i[0]+1])
def hc(p,h,s,p3):
 tis=p.inv('S',s,p3);his=p.v('H',p3,tis);return h+(his-h)/EC
def calc(task):
 tin,dp=task; he,wa=W
 # произведение трёх падений точно равно 1-dp
 fp,fh,fr=[(1-dp)**w for w in WT]; p1=P0*fp;p2=p1*fh;p3=P0/fr
 h0,h1=he.v('H',P0,T0),he.v('H',p1,T1); pw5=PW*(1-DPW);h4,h5=wa.v('H',PW,TW0),wa.v('H',pw5,TW1);qr=Q0*ER
 h3=he.v('H',p3,tin+273.15)
 # T2 определяется из условия: компрессор должен дать заданную Tвх реактора
 grid=np.linspace(TW0+5,T1-5,60); fun=lambda t:hc(he,he.v('H',p2,t),he.v('S',p2,t),p3)-h3
 a=np.array([fun(t) for t in grid]);ii=np.flatnonzero(a[:-1]*a[1:]<=0)
 if not len(ii):raise ValueError('не найдено состояние перед компрессором')
 t2=brentq(fun,grid[ii[0]],grid[ii[0]+1]);h2=he.v('H',p2,t2);g=qr/(h0-h3);qh=g*(h1-h2)
 if min(g,qh)<=0:raise ValueError('нефизический расход/тепловой поток')
 gw=qh/(h5-h4);qpipe=g*(h0-h1);nc=g*(h3-h2);nce=nc/(EMC*EMM);rho=wa.v('D',(PW+pw5)/2,(TW0+TW1)/2);nw=gw/rho*(PW-pw5)/(EPW*EMW)
 cph=he.v('Cp',(p1+p2)/2,(T1+t2)/2);cpw=wa.v('Cp',(PW+pw5)/2,(TW0+TW1)/2);eta=qh/(min(g*cph,gw*cpw)*(T1-TW0));pin=min(T1-TW1,t2-TW0);bal=qr-qpipe-qh+nc
 return dict(valid=abs(bal)<1 and pin>=5 and 0<eta<=1,T_in_C=tin,dP_gas_percent=dp*100,T2_C=t2-273.15,G_HeXe=g,G_water=gw,Q_water_MW=qh/1e6,Q_pipe_loss_kW=qpipe/1e3,Q_reactor_loss_kW=(Q0-qr)/1e3,N_comp_fluid_kW=nc/1e3,N_comp_el_kW=nce/1e3,N_pump_kW=nw/1e3,N_aux_kW=(nce+nw)/1e3,eta_hx=eta,eta_net_percent=100*(qh-nce-nw)/Q0,pinch_K=pin,balance_W=bal,P2_MPa=p2/1e6,P3_MPa=p3/1e6)
def init():
 global W;W=(P('props.npy',X),P('props_water.npy',1.))
def work(x):
 try:return calc(x)
 except Exception as e:return dict(valid=False,T_in_C=x[0],dP_gas_percent=100*x[1],reason=str(e))
FIGSIZE=(9.0,5.4)                 # 900 x 540 px: формат графиков для презентации
COLORS=('#1f4e79','#ff7f0e','#2ca02c','#d62728','#9467bd')

def set_publication_style():
 plt.rcParams.update({
  'font.family':'Times New Roman','font.size':16,'mathtext.fontset':'stix',
  'axes.linewidth':1.6,'axes.titlesize':22,'axes.labelsize':22,
  'xtick.labelsize':16,'ytick.labelsize':16,'legend.fontsize':14,
  'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True,
  'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
 })

def format_axes(ax, grid=True):
 ax.xaxis.set_minor_locator(AutoMinorLocator())
 ax.yaxis.set_minor_locator(AutoMinorLocator())
 ax.tick_params(which='major',length=7,width=1.4,pad=5)
 ax.tick_params(which='minor',length=3.5,width=1.0)
 if grid: ax.grid(True,which='major',color='0.75',alpha=.32,lw=.8)

def legend(ax, **kwargs):
 return ax.legend(frameon=True,facecolor='white',edgecolor='0.55',framealpha=.96,
                  fancybox=False,**kwargs)

def line(ax,x,y,d,label,color):
 ax.plot(d[x],d[y],'-o',color=color,lw=2.8,ms=10,mec=color,mew=.8,label=label,zorder=3)
def savefig(fig,name):
 fig.savefig(OUT/(name+'.png'),dpi=100,facecolor='white')
 fig.savefig(OUT/(name+'.svg'),facecolor='white')
 plt.close(fig)
def figs(df):
 set_publication_style()
 best=df.loc[df.eta_net_percent.idxmax()];
 for y,yl,f in [('eta_net_percent','КПД нетто АСММ, %','eta_vs_tin.png'),('N_comp_el_kW','Мощность компрессора, кВт','comp_vs_tin.png'),('Q_water_MW','Отпускаемая теплота, МВт','heat_vs_tin.png')]:
  fig,ax=plt.subplots(figsize=FIGSIZE)
  for color,dp in zip(COLORS,[1,5,10,15]):
   d=df[np.isclose(df.dP_gas_percent,dp)].sort_values('T_in_C');line(ax,'T_in_C',y,d,f'ΔP = {dp} %',color)
  ax.axvline(best.T_in_C,color='#b23a3a',ls=(0,(4,3)),lw=2.1,zorder=2)
  ax.set(xlabel='Температура входа в реактор, °C',ylabel=yl,title='Влияние температуры и потерь давления')
  format_axes(ax);legend(ax,loc='best');fig.tight_layout(pad=.55);savefig(fig,Path(f).stem)
 piv=df.pivot(index='T_in_C',columns='dP_gas_percent',values='eta_net_percent');Xg,Yg=np.meshgrid(piv.columns,piv.index)
 fig,ax=plt.subplots(figsize=FIGSIZE);cf=ax.contourf(Xg,Yg,piv.values,levels=18,cmap='viridis',extend='both');cs=ax.contour(Xg,Yg,piv.values,levels=10,colors='black',linewidths=.7,alpha=.65);ax.clabel(cs,fmt='%.1f',fontsize=11);ax.plot(best.dP_gas_percent,best.T_in_C,'*',color='#c00000',ms=18,mec='white',mew=1.0,zorder=5);ax.set(xlabel='Суммарные потери давления газа, %',ylabel='Температура входа в реактор, °C',title='Карта КПД нетто');format_axes(ax);cb=fig.colorbar(cf,ax=ax,pad=.025);cb.set_label('КПД нетто, %',fontsize=20);cb.ax.tick_params(labelsize=14);fig.tight_layout(pad=.55);savefig(fig,'eta_map')
 # Сводные однопараметрические графики: каждая линия — фиксированное ΔP.
 for x,y,xl,yl,name in [('T2_C','eta_net_percent','Температура выхода ТО, °C','КПД нетто, %','eta_vs_T_out'),('eta_hx','eta_net_percent','Фактическая эффективность ТО','КПД нетто, %','eta_vs_hx'),('pinch_K','eta_net_percent','Минимальный температурный напор ТО, K','КПД нетто, %','eta_vs_pinch'),('dP_gas_percent','N_aux_kW','Суммарные потери давления газа, %','Суммарные СН, кВт','aux_vs_dp')]:
  fig,ax=plt.subplots(figsize=FIGSIZE)
  group='dP_gas_percent' if x!='dP_gas_percent' else 'T_in_C'
  for color,(value,d) in zip(COLORS,df.groupby(group)):
   d=d.sort_values(x);line(ax,x,y,d,(f'ΔP = {value:g} %' if group=='dP_gas_percent' else f'Tвх = {value:g} °C'),color)
  ax.set(xlabel=xl,ylabel=yl,title='Сравнение расчётных режимов');format_axes(ax);legend(ax,loc='best',ncol=2);fig.tight_layout(pad=.55);savefig(fig,name)
 # Тепловой баланс использования мощности реактора для оптимального режима.
 labels=['Потери\nреактора','Потери\nтрубопровода','Работа\nкомпрессора','Отпуск\nтеплоты','СН\nэлектрические']
 vals=[best.Q_reactor_loss_kW/1000,best.Q_pipe_loss_kW/1000,best.N_comp_fluid_kW/1000,best.Q_water_MW,best.N_aux_kW/1000]
 fig,ax=plt.subplots(figsize=FIGSIZE);bars=ax.bar(labels,vals,color=['#8cc0df','#7fb3d5','#4d8cbd','#c00000','#8cc0df'],edgecolor='0.25',linewidth=1.0)
 for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v/2 if v>.25 else v+.03,f'{v:.3f}',ha='center',va='center' if v>.25 else 'bottom',fontsize=16,fontweight='bold',color='white' if v>.25 else 'black')
 ax.set(ylabel='Мощность, МВт',title='Распределение начальной тепловой мощности');format_axes(ax);ax.text(.03,.96,f'КПД нетто АСММ = {best.eta_net_percent:.3f}%\nTвх = {best.T_in_C:.0f} °C; ΔP = {best.dP_gas_percent:.0f}%',transform=ax.transAxes,va='top',fontsize=15,bbox=dict(facecolor='white',edgecolor='0.35',boxstyle='round,pad=.3'))
 fig.tight_layout(pad=.55);savefig(fig,'thermal_balance')
def main():
 OUT.mkdir(exist_ok=True);ts=list(np.arange(650.,851.,50.));ds=[.01,.05,.10,.15];tasks=[(t,d) for t in ts for d in ds];print(f'Режимов: {len(tasks)}')
 with Pool(min(os.cpu_count()or 1,8),initializer=init) as p: rows=list(tqdm(p.imap_unordered(work,tasks),total=len(tasks),desc='Свип',unit='режим')) if tqdm else list(p.imap_unordered(work,tasks))
 df=pd.DataFrame(rows);df.to_excel(OUT/'full_sweep.xlsx',index=False);ok=df[df.valid].copy();ok.to_csv(OUT/'valid_sweep.csv',index=False,encoding='utf-8-sig')
 if ok.empty:raise RuntimeError('Нет допустимых режимов; см. full_sweep.xlsx')
 figs(ok);print(f'Допустимо: {len(ok)}/{len(df)}; максимум КПД: {ok.eta_net_percent.max():.3f}%')
if __name__=='__main__':main()
