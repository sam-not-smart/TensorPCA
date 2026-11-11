"use strict";
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const m = require('./numerical_core.cjs'), c = m.CONFIG, d = c.n_items;
const results = [];
const D = m.distribution(d, c.group, c.theta), raw = m.fwht(D.p);
for (let x=0; x<raw.length; x++) raw[x] *= m.popcnt(x)%2 ? -1 : 1;
let low=0, err4=0;
for (let x=1; x<raw.length; x++) {
  if (m.popcnt(x)<=3) low=Math.max(low,Math.abs(raw[x]));
  if (m.popcnt(x)===4) err4=Math.max(err4,Math.abs(raw[x]-(D.qs.includes(x)?c.theta:0)));
}
assert(low<1e-12 && err4<1e-12);
results.push({test:'Exact enumeration of 4096 basket states', max_low_order_moment:low, max_fourth_error:err4, min_probability:Math.min(...D.p)});
const codes=m.sample(D,257,5), rs=m.moments(codes,d), cs=m.cumulants(rs,d), qs=m.combs(d,4);
const means=Array.from({length:d},(_,j)=>rs[1<<j]);
let bruteError=0;
for (let q=0;q<qs.length;q++) {
  const [a,b,c1,e]=qs[q];let fourth=0;
  const cov=(i,j)=>rs[(1<<i)|(1<<j)]-means[i]*means[j];
  for (const code of codes) fourth+=(((code>>a)&1)*2-1-means[a])*(((code>>b)&1)*2-1-means[b])*(((code>>c1)&1)*2-1-means[c1])*(((code>>e)&1)*2-1-means[e]);
  fourth=fourth/codes.length-cov(a,b)*cov(c1,e)-cov(a,c1)*cov(b,e)-cov(a,e)*cov(b,c1);
  bruteError=Math.max(bruteError,Math.abs(fourth-cs[q]));
}
assert(bruteError<1e-12);
results.push({test:'Cumulants versus direct centered calculation',max_error:bruteError});
const rand=m.rng(901),S=m.structure(9,4),v=Float64Array.from(S.quartets,()=>rand()-.5),op=m.op(S,v),dense=m.denseOperator(op,S.D);
let asym=0,fro=0;
for(let i=0;i<S.D;i++)for(let j=0;j<S.D;j++){asym=Math.max(asym,Math.abs(dense[i*S.D+j]-dense[j*S.D+i]));fro+=dense[i*S.D+j]**2;}
assert(asym===0 && Math.abs(fro-60*m.dot(v,v))<1e-9);
results.push({test:'Kikuchi symmetry and Frobenius multiplicity',dimension:S.D,max_asymmetry:asym,fro_error:Math.abs(fro-60*m.dot(v,v))});
const de=m.jacobi(dense,S.D),expected=de.values.slice().sort((a,b)=>Math.abs(b)-Math.abs(a)).slice(0,6),l=m.lanczos(op,S.D,6,123);
const err=Math.max(...expected.map((x,i)=>Math.abs(x-l.values[i])));
assert(err<1e-8 && l.residual<1e-7);
results.push({test:'Lanczos versus dense eigensolve',eigenvalue_error:err,residual:l.residual,steps:l.steps});
fs.mkdirSync(path.join(__dirname,'results'),{recursive:true});
fs.writeFileSync(path.join(__dirname,'results/numerical_tests.json'),JSON.stringify({passed:true,tests:results},null,2)+'\n');
console.log(JSON.stringify(results,null,2));
