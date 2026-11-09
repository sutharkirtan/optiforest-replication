import json,sys,numpy as np
from scipy import stats
def summ(path):
    d=json.load(open(path)); out={}
    for k,v in d.items():
        a=np.array(v['branch0']['auc']); b=np.array(v['branch2']['auc'])
        diff=a.mean()-b.mean()
        se=np.sqrt(a.var(ddof=1)/len(a)+b.var(ddof=1)/len(b))
        va,vb=a.var(ddof=1)/len(a),b.var(ddof=1)/len(b)
        if se==0: df=len(a)+len(b)-2
        else: df=(va+vb)**2/(va**2/(len(a)-1)+vb**2/(len(b)-1))
        t=stats.t.ppf(.975,df)
        out[k]=dict(mixed=a.mean(),mixed_sd=a.std(ddof=1),binary=b.mean(),binary_sd=b.std(ddof=1),
          diff=diff,lo=diff-t*se,hi=diff+t*se,n=len(a),
          mixed_pr=np.mean(v['branch0']['pr']),binary_pr=np.mean(v['branch2']['pr']))
    return out
if __name__=='__main__':
    for p in sys.argv[1:]:
        for k,v in summ(p).items():
            sig='*' if v['lo']>0 or v['hi']<0 else ' '
            print(f"{k:14s} mixed {v['mixed']:.3f}±{v['mixed_sd']:.3f} bin {v['binary']:.3f}±{v['binary_sd']:.3f} diff {v['diff']:+.3f} [{v['lo']:+.3f},{v['hi']:+.3f}]{sig}")
