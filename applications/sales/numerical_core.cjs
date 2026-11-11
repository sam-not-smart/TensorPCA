"use strict";
// Sales-specific generator, binary moments, PCA baseline, and evaluation.
// All Kikuchi recovery is supplied by the shared core.
const core = require('../../src/kikuchi.cjs');
const {rng,combs,mask,popcnt,dot,norm,unit,matvec,jacobi,structure,op,denseOperator,lanczos,density,pattern} = core._internals;
function spectral(c,d,ell,seed) {
  const result = core.recoverDirection(c,{dimension:d,level:ell,seed});
  return {u:result.direction,info:result.info};
}
const CONFIG={n_items:12,group:[0,2,4,7,9,11],theta:0.12,sample_sizes:[256,1024,4096,8192],validation_samples:8192,replicates:24,seed:20261109,levels:[2,4],permutations:199,items:["Charcoal","Oats","Meat","Milk","Marinade","Flour","Pasta","Paper plates","Apples","Firelighters","Coffee","Braai tongs"]};
function fwht(v){const a=Float64Array.from(v);for(let h=1;h<a.length;h*=2)for(let i=0;i<a.length;i+=2*h)for(let j=i;j<i+h;j++){const x=a[j],y=a[j+h];a[j]=x+y;a[j+h]=x-y;}return a;}
function moments(codes,d,n=codes.length){const h=new Float64Array(1<<d);for(let i=0;i<n;i++)h[codes[i]]++;const a=fwht(h);for(let m=0;m<a.length;m++)a[m]*=(popcnt(m)%2?-1:1)/n;return a;}
function cumulants(raw,d){
 return Float64Array.from(combs(d,4),q=>{
  const [a,b,c,d0]=q.map(i=>1<<i),all=a|b|c|d0,vs=[a,b,c,d0];let v=raw[all];
  for(const x of vs)v-=raw[x]*raw[all^x];
  v-=raw[a|b]*raw[c|d0]+raw[a|c]*raw[b|d0]+raw[a|d0]*raw[b|c];
  for(let i=0;i<4;i++)for(let j=i+1;j<4;j++){const other=vs.filter((_,k)=>k!==i&&k!==j);v+=2*raw[vs[i]|vs[j]]*raw[other[0]]*raw[other[1]];}
  return v-6*raw[a]*raw[b]*raw[c]*raw[d0];
 });
}
function distribution(d,group,theta){
 if(group.length!==6||theta<0||theta>.2)throw Error("Invalid generator configuration");
 const qs=combs(6,4).map(s=>mask(s.map(i=>group[i]))),p=new Float64Array(1<<d),cdf=new Float64Array(1<<d);let sum=0;
 for(let x=0;x<p.length;x++){let poly=0;for(const s of qs)poly+=popcnt(x&s)%2?-1:1;p[x]=(1+theta*poly)/p.length;sum+=p[x];cdf[x]=sum;if(p[x]<0)throw Error("Negative probability");}cdf[cdf.length-1]=1;return {p,cdf,qs};
}
function sample(dist,n,seed){const random=rng(seed),out=new Uint16Array(n);for(let t=0;t<n;t++){const u=random();let lo=0,hi=dist.cdf.length-1;while(lo<hi){const m=(lo+hi)>>1;if(dist.cdf[m]>=u)hi=m;else lo=m+1;}out[t]=lo;}return out;}
function fit(raw,d,seed){
 const cov=new Float64Array(d*d);for(let i=0;i<d;i++)for(let j=0;j<d;j++)cov[i*d+j]=(i===j?1:raw[(1<<i)|(1<<j)])-raw[1<<i]*raw[1<<j];
 const e=jacobi(cov,d),j=e.values.indexOf(Math.max(...e.values)),u=Float64Array.from({length:d},(_,i)=>e.vectors[i*d+j]);
 const c=cumulants(raw,d),out={PCA:{u,info:{residual:0}}};for(const ell of CONFIG.levels)out["TensorPCA l="+ell]=spectral(c,d,ell,seed);
 return {methods:out,coefficients:c,covariance:cov};
}
function metric(u,truth,group,validationRaw,d){
 const order=Array.from(u,(_,i)=>i).sort((a,b)=>Math.abs(u[b])-Math.abs(u[a])).slice(0,group.length),hit=order.filter(i=>group.includes(i)).length;
 const qs=combs(d,4),p=pattern(u,qs),raw4=qs.map(s=>validationRaw[mask(s)]);
 return {squared_cosine:dot(u,truth)**2,support_recall:hit/group.length,exact_support:+(hit===group.length),validation_contrast:p?dot(p,raw4):0};
}
function permutation(validation,u,d,count,seed){
 const qs=combs(d,4),p=pattern(u,qs),random=rng(seed),contrast=codes=>{const raw=moments(codes,d);return Math.abs(dot(p,qs.map(s=>raw[mask(s)])));};
 const observed=contrast(validation),nulls=[],n=validation.length;
 for(let r=0;r<count;r++){const codes=new Uint16Array(n);for(let j=0;j<d;j++){const bits=Uint8Array.from(validation,x=>(x>>j)&1);for(let k=n-1;k>0;k--){const t=Math.floor(random()*(k+1)),b=bits[k];bits[k]=bits[t];bits[t]=b;}for(let k=0;k<n;k++)codes[k]|=bits[k]<<j;}nulls.push(contrast(codes));}
 return {observed,p_value:(1+nulls.filter(x=>x>=observed).length)/(count+1),count,null_statistics:nulls};
}
function summary(rows){
 const keys=[...new Set(rows.map(r=>[r.condition,r.n_train,r.method].join("|")))],out=[];
 for(const key of keys){const selected=rows.filter(r=>[r.condition,r.n_train,r.method].join("|")===key),first=selected[0],s={condition:first.condition,n_train:first.n_train,method:first.method,replicates:selected.length};
  for(const metric of ["squared_cosine","support_recall","exact_support","validation_contrast"]){const vals=selected.map(r=>r[metric]),mean=vals.reduce((a,b)=>a+b,0)/vals.length,sd=Math.sqrt(vals.reduce((a,b)=>a+(b-mean)**2,0)/(vals.length-1));s[metric+"_mean"]=mean;s[metric+"_sd"]=sd;s[metric+"_min"]=Math.min(...vals);s[metric+"_max"]=Math.max(...vals);}out.push(s);
 }return out;
}

module.exports = {CONFIG,rng,combs,mask,popcnt,dot,norm,unit,matvec,jacobi,structure,op,denseOperator,lanczos,density,pattern,spectral,fwht,moments,cumulants,distribution,sample,fit,metric,permutation,summary};
