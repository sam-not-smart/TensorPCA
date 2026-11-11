"use strict";
// Shared, application-independent distinct-index fourth-order Kikuchi solver.
function rng(seed){let a=seed>>>0;return function(){a=(a+0x6D2B79F5)>>>0;let t=a;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return ((t^(t>>>14))>>>0)/4294967296;};}
function combs(n,k){const out=[];function rec(v,start){if(v.length===k){out.push(v.slice());return;}for(let j=start;j<=n-(k-v.length);j++){v.push(j);rec(v,j+1);v.pop();}}rec([],0);return out;}
function mask(s){return s.reduce((a,b)=>a|(1<<b),0);}
function popcnt(x){let c=0;while(x){x&=x-1;c++;}return c;}
function dot(a,b){let s=0;for(let i=0;i<a.length;i++)s+=a[i]*b[i];return s;}
function norm(a){return Math.sqrt(dot(a,a));}
function unit(a){const n=norm(a);return Float64Array.from(a,x=>x/n);}
function matvec(A,x,n){const y=new Float64Array(n);for(let i=0;i<n;i++)for(let j=0;j<n;j++)y[i]+=A[i*n+j]*x[j];return y;}
function jacobi(A0,n,tol=1e-13,maxSweeps=100){
 const A=Float64Array.from(A0),V=new Float64Array(n*n);for(let i=0;i<n;i++)V[i*n+i]=1;
 for(let sweep=0;sweep<maxSweeps;sweep++){
  let max=0,scale=0;for(let i=0;i<n;i++)scale=Math.max(scale,Math.abs(A[i*n+i]));
  for(let p=0;p<n;p++)for(let q=p+1;q<n;q++){
   const b=A[p*n+q];max=Math.max(max,Math.abs(b));if(Math.abs(b)<=tol*Math.max(1,scale))continue;
   const tau=(A[q*n+q]-A[p*n+p])/(2*b),t=(tau>=0?1:-1)/(Math.abs(tau)+Math.sqrt(1+tau*tau)),c=1/Math.sqrt(1+t*t),s=t*c;
   A[p*n+p]-=t*b;A[q*n+q]+=t*b;A[p*n+q]=A[q*n+p]=0;
   for(let k=0;k<n;k++)if(k!==p&&k!==q){const u=A[k*n+p],v=A[k*n+q];A[k*n+p]=A[p*n+k]=c*u-s*v;A[k*n+q]=A[q*n+k]=s*u+c*v;}
   for(let k=0;k<n;k++){const u=V[k*n+p],v=V[k*n+q];V[k*n+p]=c*u-s*v;V[k*n+q]=s*u+c*v;}
  }
  if(max<=tol*Math.max(1,scale)){return {values:Array.from({length:n},(_,i)=>A[i*n+i]),vectors:V};}
 }
 throw Error("Jacobi eigensolver did not converge");
}
const cache=new Map();
function structure(d,ell){
 const key=d+","+ell;if(cache.has(key))return cache.get(key);
 const states=combs(d,ell),quartets=combs(d,4),sm=states.map(mask),qm=quartets.map(mask),lookup=new Map(sm.map((x,i)=>[x,i]));
 const rowptr=[0],cols=[],indices=[];
 for(let row=0;row<sm.length;row++){for(let q=0;q<qm.length;q++)if(popcnt(sm[row]&qm[q])===2){cols.push(lookup.get(sm[row]^qm[q]));indices.push(q);}rowptr.push(cols.length);}
 const lower=combs(d,ell-1),lowerlookup=new Map(lower.map((x,i)=>[mask(x),i])),dec=[];
 for(let j=0;j<states.length;j++)for(const a of states[j])dec.push([lowerlookup.get(sm[j]^(1<<a)),a,j]);
 const out={d,ell,states,quartets,qm,rows:Int32Array.from(rowptr),cols:Int32Array.from(cols),indices:Int32Array.from(indices),D:states.length,lower:lower.length,dec};cache.set(key,out);return out;
}
function op(struct,c){
 const vals=Float64Array.from(struct.indices,q=>c[q]);
 return function(x){const y=new Float64Array(struct.D);for(let row=0;row<struct.D;row++){let s=0;for(let k=struct.rows[row];k<struct.rows[row+1];k++)s+=vals[k]*x[struct.cols[k]];y[row]=s;}return y;};
}
function denseOperator(mv,n){const A=new Float64Array(n*n);for(let j=0;j<n;j++){const e=new Float64Array(n);e[j]=1;const y=mv(e);for(let i=0;i<n;i++)A[i*n+j]=y[i];}return A;}
function lanczos(mv,n,k,seed){
 if(n<=70){const e=jacobi(denseOperator(mv,n),n);const ix=e.values.map((_,i)=>i).sort((a,b)=>Math.abs(e.values[b])-Math.abs(e.values[a])).slice(0,k);let res=0;
  const vectors=ix.map(j=>Float64Array.from({length:n},(_,i)=>e.vectors[i*n+j]));for(let j=0;j<ix.length;j++){const v=mv(vectors[j]);for(let i=0;i<n;i++)v[i]-=e.values[ix[j]]*vectors[j][i];res=Math.max(res,norm(v));}return {values:ix.map(j=>e.values[j]),vectors,residual:res,steps:n};}
 const random=rng(seed),Q=[],alpha=[],beta=[];let q=unit(Float64Array.from({length:n},()=>2*random()-1)),lastBeta=0;
 const maximum=Math.min(n,200);
 for(let step=0;step<maximum;step++){
  Q.push(q);const z=mv(q);if(step)for(let i=0;i<n;i++)z[i]-=lastBeta*Q[step-1][i];
  const a=dot(q,z);alpha.push(a);for(let i=0;i<n;i++)z[i]-=a*q[i];
  for(let pass=0;pass<2;pass++)for(const prev of Q){const t=dot(prev,z);for(let i=0;i<n;i++)z[i]-=t*prev[i];}
  const b=norm(z),m=step+1;beta.push(b);
  if((m>=40&&m%20===0)||b<1e-12||m===maximum){
   const T=new Float64Array(m*m);for(let i=0;i<m;i++){T[i*m+i]=alpha[i];if(i+1<m)T[i*m+i+1]=T[(i+1)*m+i]=beta[i];}
   const e=jacobi(T,m),ix=e.values.map((_,i)=>i).sort((a,b)=>Math.abs(e.values[b])-Math.abs(e.values[a])).slice(0,k);
   const estimated=Math.max(...ix.map(j=>Math.abs(b*e.vectors[(m-1)*m+j]))),scale=Math.max(...ix.map(j=>Math.abs(e.values[j])));
   if(estimated<=1e-9*Math.max(scale,1e-12)||b<1e-12||m===n){
    const vectors=ix.map(j=>{const v=new Float64Array(n);for(let t=0;t<m;t++)for(let i=0;i<n;i++)v[i]+=Q[t][i]*e.vectors[t*m+j];return unit(v);});
    let residual=0;vectors.forEach((v,j)=>{const r=mv(v);for(let i=0;i<n;i++)r[i]-=e.values[ix[j]]*v[i];residual=Math.max(residual,norm(r));});
    if(residual>1e-7*Math.max(scale,1e-12))throw Error("Eigen residual too large: "+residual);
    return {values:ix.map(j=>e.values[j]),vectors,residual,steps:m};
   }
  }
  if(b<1e-14)throw Error("Early Lanczos breakdown");lastBeta=b;q=Float64Array.from(z,x=>x/b);
 }
 throw Error("Lanczos did not converge within 200 steps");
}
function density(v,S){
 const A=new Float64Array(S.lower*S.d);for(const [r,a,j] of S.dec)A[r*S.d+a]+=v[j];
 const rho=new Float64Array(S.d*S.d);for(let r=0;r<S.lower;r++)for(let a=0;a<S.d;a++)for(let b=0;b<S.d;b++)rho[a*S.d+b]+=A[r*S.d+a]*A[r*S.d+b]/S.ell;return rho;
}
function pattern(u,qs){const p=Float64Array.from(qs,s=>s.reduce((x,i)=>x*u[i],1));const z=norm(p);return z>1e-14?Float64Array.from(p,x=>x/z):null;}
function spectral(c,d,ell,seed){
 const S=structure(d,ell),e=lanczos(op(S,c),S.D,6,seed);let best=-Infinity,bestu;
 for(const lifted of e.vectors){const r=jacobi(density(lifted,S),d);for(let j=0;j<d;j++){const u=Float64Array.from({length:d},(_,i)=>r.vectors[i*d+j]),p=pattern(u,S.quartets);if(!p)continue;const score=Math.abs(dot(p,c));if(score>best){best=score;bestu=u;}}}
 return {u:bestu,info:{residual:e.residual,steps:e.steps,eigenvalues:e.values,score:best,dimension:S.D}};
}

function validateDimension(d) {
  if (!Number.isInteger(d) || d < 4 || d > 30) throw new RangeError("dimension must be an integer from 4 to 30 (32-bit subset masks)");
}
function quartetIndices(dimension) {
  validateDimension(dimension);
  return combs(dimension, 4);
}
/** Recover one direction from lexicographically ordered distinct fourth cumulants.
 * The array length must be choose(dimension, 4). Largest-absolute spectral ends
 * are considered, so either sign of the planted fourth cumulant is supported.
 * This prototype builds a combinatorial subset operator and is intended for small runs.
 */
function recoverDirection(coefficients, {dimension, level = 2, seed = 0} = {}) {
  validateDimension(dimension);
  if (!Number.isInteger(level) || level < 2 || level > dimension - 2) throw new RangeError("level must be between 2 and dimension - 2");
  if (!Number.isInteger(seed)) throw new TypeError("seed must be an integer");
  const expected = dimension * (dimension-1) * (dimension-2) * (dimension-3) / 24;
  if (!coefficients || coefficients.length !== expected) throw new RangeError(`Expected ${expected} distinct fourth-order coefficients`);
  const values = Float64Array.from(coefficients);
  if (!values.every(Number.isFinite)) throw new TypeError("Coefficients must be finite numbers");
  if (values.every(value => value === 0)) throw new RangeError("A zero tensor does not identify a direction");
  const result = spectral(values, dimension, level, seed);
  if (!result.u) throw new Error("No nonzero fourth-order candidate could be decoded");
  return {direction: result.u, info: result.info};
}
module.exports = {
  recoverDirection, quartetIndices,
  // Internal numerical routines are exposed for invariant checks and application adapters.
  _internals: {rng,combs,mask,popcnt,dot,norm,unit,matvec,jacobi,structure,op,denseOperator,lanczos,density,pattern,spectral}
};
