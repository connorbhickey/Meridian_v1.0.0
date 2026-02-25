import { useState, useCallback } from "react";

// ═══════════════════════════════════════════════════════════════
// A.V.I.S. STOCK & ETF PREDICTOR v6 — CLEAN REBUILD
// ═══════════════════════════════════════════════════════════════
// Architecture: Merton Jump-Diffusion + 25-method ensemble
// Key v6 fix: VOL-ADAPTIVE SIGNAL SCALING — signals produce
// meaningful estimates for ALL volatility regimes ($11 IREN
// through $415 MSFT through $5 penny stocks)
//
// Math: MJD(Student-t ν=5) · James-Stein · Bootstrap CI ·
//       Inverse-Variance · Decorrelation · Kelly Criterion ·
//       Decomposed 3-tier Prediction Intervals
//
// Sources: Merton 1976, James & Stein 1961, Efron 1979,
//   Kelly 1956, Thorp 1997, Fama-French 1993/2015,
//   Ball & Brown 1968, Marsaglia & Tsang 2000
// ═══════════════════════════════════════════════════════════════

// ────────────────────────────────────────────
// §1  PRNG & DISTRIBUTIONS
// ────────────────────────────────────────────
const lcg=s=>{let x=s%2147483647;if(x<=0)x+=2147483646;return()=>{x=(x*16807)%2147483647;return(x-1)/2147483646}};
const nrv=r=>{let u,v,s;do{u=2*r()-1;v=2*r()-1;s=u*u+v*v}while(s>=1||s===0);return u*Math.sqrt(-2*Math.log(s)/s)};
const clamp=(v,lo,hi)=>Math.max(lo,Math.min(hi,v));
const pctl=(a,p)=>{const s=Float64Array.from(a).sort();return s[Math.min(Math.floor(p/100*s.length),s.length-1)]};

// Gamma RV — Marsaglia & Tsang 2000 (for chi² → Student-t)
function gammaRV(shape,r){
  if(shape<1)return gammaRV(shape+1,r)*Math.pow(r(),1/shape);
  const d=shape-1/3,c=1/Math.sqrt(9*d);
  for(;;){let x,v;do{x=nrv(r);v=1+c*x}while(v<=0);v=v*v*v;const u=r();
    if(u<1-.0331*x*x*x*x)return d*v;if(Math.log(u)<.5*x*x+d*(1-v+Math.log(v)))return d*v}
}

// Student-t(ν) — T = Z/√(χ²/ν), normalized to unit variance
// ν=5: P(|X|>4σ) ≈ 0.5% vs 0.006% Gaussian (Welch 2024)
function studentTRV(nu,r){
  const z=nrv(r),chi2=2*gammaRV(nu/2,r);
  return z/Math.sqrt(chi2/nu)*Math.sqrt((nu-2)/nu);
}

// ────────────────────────────────────────────
// §2  MERTON JUMP-DIFFUSION (Merton 1976)
// ────────────────────────────────────────────
// dS/S = (μ − λk)dt + σ·dW + J·dN
// W uses Student-t(ν) for fat tails
// J ~ LogN(jumpMu, jumpSig²), N ~ Poisson(λ)
function mjd(S0,mu,sig,T,N,seed,opts={}){
  const{nu=5,lambda=2,jumpMu=-.02,jumpSig=.08,earningsJump=0,earningsDay=-1}=opts;
  const dt=1/252,r=lcg(seed);
  const k=Math.exp(jumpMu+.5*jumpSig*jumpSig)-1;
  const drift=(mu-.5*sig*sig-lambda*k)*dt;
  const diff=sig*Math.sqrt(dt);
  const out=new Float64Array(N);
  for(let i=0;i<N;i++){
    let logS=0;
    for(let t=0;t<T;t++){
      logS+=drift+diff*studentTRV(nu,r);
      if(r()<lambda*dt)logS+=jumpMu+jumpSig*nrv(r);
      if(t===earningsDay&&earningsJump>0)logS+=nrv(r)*earningsJump;
    }
    out[i]=S0*Math.exp(logS);
  }
  return out;
}

function buildHist(a,bins=40){
  const lo=pctl(a,2),hi=pctl(a,98),st=(hi-lo)/bins;
  if(st<=0)return[{c:lo,d:100}];
  const h=[];
  for(let i=0;i<bins;i++){
    const a0=lo+i*st,b0=a0+st;let cnt=0;
    for(let j=0;j<a.length;j++)if(a[j]>=a0&&a[j]<b0)cnt++;
    h.push({c:+((a0+b0)/2).toFixed(2),d:+((cnt/a.length)*100).toFixed(2)});
  }
  return h;
}

// ────────────────────────────────────────────
// §3  SIGNAL FUNCTIONS (S1–S20)
// ────────────────────────────────────────────
// Each returns { est, ...metadata }
// est = price estimate (raw, pre-vol-scaling)
// Vol-scaling applied LATER in runModel §5

// S1: Regime Detection (HMM-proxy)
function sigRegime(d){
  const{currentPrice:S,sma50,sma200,high52w,low52w,vix}=d;
  let sma=0;if(sma50&&sma200)sma=sma50>sma200?Math.min((sma50/sma200-1)*10,1):Math.max((sma50/sma200-1)*10,-1);
  let pp=0;if(sma50&&sma200)pp=S>sma50&&S>sma200?1:S>sma200?.3:S>sma50?-.2:-.8;
  const rng=high52w-low52w,pct=rng>0?(S-low52w)/rng:.5,rs=(pct-.5)*2;
  const v=vix||18,vp=v>30?-.3:v>25?-.15:v>20?-.05:.05;
  const comp=sma*.3+pp*.3+rs*.25+vp*.15;
  let reg,bW,mW,eW;
  if(comp>.35){reg="BULL";bW=.55;mW=.35;eW=.10}
  else if(comp>.1){reg="RECOVERY";bW=.40;mW=.40;eW=.20}
  else if(comp>-.1){reg="NEUTRAL";bW=.30;mW=.40;eW=.30}
  else if(comp>-.35){reg="DETERIORATING";bW=.15;mW=.35;eW=.50}
  else{reg="BEAR";bW=.10;mW=.25;eW=.65}
  // Drift uses fed funds from data (NOT hardcoded)
  const rf=(d.fedFundsRate||4.5)/100;
  const drift=bW*.25*Math.max(d.beta,1)+mW*(rf+d.beta*.055)+eW*(-.10*Math.max(d.beta,1));
  return{regime:reg,comp:+comp.toFixed(3),bW,mW,eW,drift,
    est:+(S*Math.exp(drift*(d.tradingDaysRemaining/252))).toFixed(2)}
}

// S2: Momentum/Reversion
function sigMom(d){
  const{currentPrice:S,sma50,rsi14}=d,rsi=rsi14||50;
  const rs=rsi>70?-.3:rsi>60?.15:rsi>40?0:rsi>30?-.1:.25;
  const m50=sma50?(S/sma50-1):0,ms=Math.abs(m50)>.2?-m50*.5:m50*.8;
  const sig=rs*.45+ms*.55;
  return{est:+(S*(1+sig*.15)).toFixed(2),signal:+sig.toFixed(3),rsi,
    label:sig>.1?"BULLISH":sig<-.1?"BEARISH":"NEUTRAL"}
}

// S3: Sector Relative Strength
function sigSRS(d){
  const sp=d.stockPerformance3m||0,sec=d.sectorPerformance3m||0,spread=sp-sec;
  const sig=spread>15?-.06:spread>5?.02:spread>-5?0:spread>-15?.04:.08;
  return{est:+(d.currentPrice*(1+sig)).toFixed(2),spread:+spread.toFixed(1),label:spread>5?"OUTPERF":spread<-5?"UNDERPERF":"INLINE"}
}

// S4: Vol Regime
function sigVol(d){
  const v=d.vix||18,b=d.beta||1,sk=v>35?-.06:v>28?-.04:v>22?-.02:v>15?.01:.03;
  const adj=sk*(1+(b-1)*.3);
  return{est:+(d.currentPrice*(1+adj)).toFixed(2),skew:+(adj*100).toFixed(2),
    regime:v>28?"CRISIS":v>22?"ELEVATED":v>15?"NORMAL":"COMPLACENT",vix:v}
}

// S5: Institutional Sentiment
function sigInst(d){
  const si=d.shortInterest||3,pcr=d.putCallRatio||.7,io=d.institutionalOwnership||65;
  const s1=si>20?.08:si>10?.03:si>5?0:-.01;
  const s2=pcr>1.2?.05:pcr>.9?.02:pcr>.6?0:-.03;
  const s3=io>85?.01:io>60?.005:-.01;
  const c=s1*.4+s2*.35+s3*.25;
  return{est:+(d.currentPrice*(1+c)).toFixed(2),comp:+(c*100).toFixed(2),si,pcr,io,
    label:c>.02?"BULLISH":c<-.01?"BEARISH":"NEUTRAL"}
}

// S6: EPS Revision Momentum
function sigEpsRevision(d){
  const rev=d.earningsRevision3m||0,sig=clamp(rev/20,-.15,.15);
  return{est:+(d.currentPrice*(1+sig)).toFixed(2),revision:+rev.toFixed(1),
    label:rev>5?"UPGRADING":rev<-5?"DOWNGRADING":"STABLE"}
}

// S7: Size Factor — Fama-French SMB
function sigSize(d){
  const cap=d.marketCapNum||50;
  let prem;if(cap<2)prem=.04;else if(cap<10)prem=.025;else if(cap<50)prem=.01;else if(cap<200)prem=0;else prem=-.005;
  return{est:+(d.currentPrice*Math.exp(prem*(d.tradingDaysRemaining/252))).toFixed(2),
    capB:+cap.toFixed(1),prem:+(prem*100).toFixed(2),label:cap<10?"SMALL-CAP":cap<50?"MID-CAP":"LARGE-CAP"}
}

// S8: Value Factor — HML
function sigValue(d){
  const pe=d.peRatio||20,secPe=d.sectorAvgPE||20,pb=d.priceToBook||3,secPb=d.sectorAvgPB||3;
  const peDisc=secPe>0?(pe/secPe-1):0,pbDisc=secPb>0?(pb/secPb-1):0;
  const valScore=-(peDisc*.5+pbDisc*.5),sig=clamp(valScore*.08,-.06,.06);
  return{est:+(d.currentPrice*(1+sig)).toFixed(2),pe,secPe,
    score:+(valScore*100).toFixed(1),label:valScore>.15?"DEEP VALUE":valScore>0?"VALUE":"GROWTH"}
}

// S9: Quality/Profitability — RMW
function sigQuality(d){
  const roe=d.roe||15,margin=d.profitMargin||10;
  let q=0;
  if(roe>25)q+=.03;else if(roe>15)q+=.015;else if(roe>8)q+=0;else q-=.02;
  if(margin>25)q+=.02;else if(margin>12)q+=.01;else if(margin>5)q+=0;else q-=.015;
  return{est:+(d.currentPrice*(1+q)).toFixed(2),roe:+roe.toFixed(1),margin:+margin.toFixed(1),
    score:+(q*100).toFixed(2),label:q>.03?"HIGH QUALITY":q>0?"ADEQUATE":"LOW QUALITY"}
}

// S10: Investment Factor — CMA
function sigInvestment(d){
  const capex=d.capexToRevenue||8;let sig;
  if(capex>25)sig=-.025;else if(capex>15)sig=-.01;else if(capex>8)sig=.005;else if(capex>3)sig=.015;else sig=.02;
  return{est:+(d.currentPrice*(1+sig)).toFixed(2),capex:+capex.toFixed(1),label:capex>15?"AGGRESSIVE":"CONSERVATIVE"}
}

// S11: Low Volatility Anomaly — BAB
function sigLowVol(d){
  const vol=d.annualizedVol||.3;let sig;
  if(vol<.15)sig=.02;else if(vol<.25)sig=.01;else if(vol<.40)sig=0;else if(vol<.60)sig=-.015;else sig=-.03;
  return{est:+(d.currentPrice*(1+sig)).toFixed(2),vol:+(vol*100).toFixed(0),label:vol<.25?"LOW VOL":vol<.45?"AVERAGE":"HIGH VOL"}
}

// S12: PEAD — Ball & Brown 1968
function sigPEAD(d){
  const surprise=d.lastEarningsSurprise||0,days=d.daysSinceEarnings||45;
  const decay=Math.max(0,1-days/60),driftPct=clamp(surprise*.003*decay,-.08,.08);
  return{est:+(d.currentPrice*(1+driftPct)).toFixed(2),surprise:+surprise.toFixed(1),days,
    label:surprise>5?"BEAT":"INLINE"}
}

// S13: Seasonality
function sigSeason(d){
  const month=new Date().getMonth()+1;
  const p={1:.02,2:.005,3:.01,4:.015,5:-.005,6:-.01,7:.005,8:-.01,9:-.015,10:.005,11:.015,12:.02};
  return{est:+(d.currentPrice*(1+(p[month]||0))).toFixed(2),month}
}

// S14: Options/IV Skew
function sigOptionsSkew(d){
  const ivRank=d.ivRank||50,skew=d.ivSkew||0;
  let sig;if(ivRank>80)sig=.03;else if(ivRank>60)sig=.01;else if(ivRank>40)sig=0;else if(ivRank>20)sig=-.005;else sig=-.02;
  sig+=clamp(skew*.01,-.02,.02);
  return{est:+(d.currentPrice*(1+sig)).toFixed(2),ivRank,label:ivRank>70?"HIGH FEAR":"NORMAL"}
}

// S15: Insider Activity
function sigInsider(d){
  const net=d.insiderNetBuying||0;let sig;
  if(net>5)sig=.04;else if(net>1)sig=.02;else if(net>-.5)sig=0;else if(net>-3)sig=-.01;else sig=-.025;
  return{est:+(d.currentPrice*(1+sig)).toFixed(2),net:+net.toFixed(1),label:net>1?"BUYING":net<-1?"SELLING":"NEUTRAL"}
}

// S16: Revenue Acceleration
function sigRevAccel(d){
  const g1=d.revenueGrowthPct||8,g0=d.priorRevenueGrowthPct||8,accel=g1-g0;
  const sig=clamp(accel*.003,-.04,.04);
  return{est:+(d.currentPrice*(1+sig)).toFixed(2),accel:+accel.toFixed(1),label:accel>3?"ACCEL":accel<-3?"DECEL":"STABLE"}
}

// S17: FCF Yield
function sigFCF(d){
  const fcfy=d.fcfYield||4;let sig;
  if(fcfy>10)sig=.04;else if(fcfy>6)sig=.02;else if(fcfy>3)sig=.005;else if(fcfy>0)sig=-.005;else sig=-.03;
  return{est:+(d.currentPrice*(1+sig)).toFixed(2),yield:+fcfy.toFixed(1),label:fcfy>6?"HIGH":"FAIR"}
}

// S18: Leverage/Health
function sigLeverage(d){
  const de=d.debtToEquity||.8,ic=d.interestCoverage||10;let sig=0;
  if(de<.3)sig+=.015;else if(de<.8)sig+=.005;else if(de<2)sig-=.005;else sig-=.02;
  if(ic>15)sig+=.01;else if(ic>5)sig+=.005;else if(ic>2)sig-=.005;else sig-=.02;
  return{est:+(d.currentPrice*(1+sig)).toFixed(2),de:+de.toFixed(2),ic:+ic.toFixed(1),
    label:de<.5&&ic>10?"FORTRESS":de>2||ic<3?"FRAGILE":"MODERATE"}
}

// S19: Buyback Signal
function sigBuyback(d){
  const sc=d.shareCountChange||0;let sig;
  if(sc<-5)sig=.035;else if(sc<-2)sig=.02;else if(sc<1)sig=.005;else if(sc<5)sig=-.01;else sig=-.03;
  return{est:+(d.currentPrice*(1+sig)).toFixed(2),change:+sc.toFixed(1),label:sc<-2?"BUYBACK":sc>2?"DILUTING":"STABLE"}
}

// S20: Macro Regime — Adrian & Shin 2010
function sigMacro(d){
  const vix=d.vix||18,ffr=d.fedFundsRate||4.5,yc=d.yieldCurve2s10s||0.2,cs=d.creditSpreadHY||3.5;
  let vS;if(vix>35)vS=-.04;else if(vix>28)vS=-.025;else if(vix>22)vS=-.01;else if(vix>15)vS=.01;else vS=.02;
  let ycS;if(yc<-.5)ycS=-.04;else if(yc<0)ycS=-.02;else if(yc<.5)ycS=0;else if(yc<1.5)ycS=.01;else ycS=.015;
  let csS;if(cs>6)csS=-.04;else if(cs>5)csS=-.02;else if(cs>4)csS=-.01;else if(cs>3)csS=.005;else csS=.015;
  let fS;if(ffr>6)fS=-.02;else if(ffr>5)fS=-.01;else if(ffr>3)fS=0;else if(ffr>1)fS=.01;else fS=.02;
  const comp=vS*.3+ycS*.3+csS*.2+fS*.2;
  return{est:+(d.currentPrice*(1+comp)).toFixed(2),comp:+(comp*100).toFixed(2),
    label:comp>.01?"SUPPORTIVE":comp<-.015?"RESTRICTIVE":"NEUTRAL"}
}

// ────────────────────────────────────────────
// §4  ENSEMBLE MACHINERY
// ────────────────────────────────────────────

// Horizon-Adaptive Weights (25 methods)
function horizonWeights(T,isETF){
  const S=[.07,.05,.04,.05,.05,.09,.04,.06,.06,.07,.01,.01,.01,.01,.02,.09,.03,.07,.04,.03,.01,.02,.02,.03,.02];
  const M=[.10,.08,.05,.07,.06,.04,.04,.03,.04,.04,.02,.03,.03,.02,.02,.04,.02,.03,.02,.03,.03,.03,.03,.04,.04];
  const L=[.12,.10,.06,.05,.04,.01,.02,.02,.02,.02,.04,.07,.06,.04,.03,.00,.01,.01,.02,.02,.06,.04,.03,.05,.03];
  let a;if(T<=63)a=0;else if(T<=252)a=(T-63)/189;else if(T<=504)a=1+(T-252)/252;else a=2;
  let w;if(a<=1)w=S.map((s,i)=>s+a*(M[i]-s));else w=M.map((m,i)=>m+(a-1)*(L[i]-m));
  const sum=w.reduce((x,y)=>x+y,0);return w.map(x=>x/sum);
}

// Signal Interactions (multiplicative weight boosts)
function signalInteractions(sig){
  const m=new Array(25).fill(1);
  // Momentum × Short Interest → squeeze
  if(sig.mom.signal>.1&&(sig.inst.si||3)>15){m[5]*=1.5;m[8]*=1.3}
  // Value × Quality → Buffett (threshold=2, NOT 200)
  if(sig.val.score>0&&sig.qual.score>2){m[11]*=1.4;m[12]*=1.4}
  // PEAD × EPS Revision → confirmation
  if(sig.pead.surprise>5&&sig.epsRev.revision>3){m[15]*=1.5;m[9]*=1.3}
  // Macro tight × High leverage → amplified downside
  if(sig.macro.comp<-1.5&&(sig.lev.de||.8)>2){m[21]*=1.5;m[23]*=1.3}
  // FCF strong × Buyback → capital return
  if((sig.fcf.yield||4)>6&&(sig.buyback.change||0)<-2){m[20]*=1.3;m[22]*=1.3}
  return m;
}

// Earnings Event Detection
function detectEarningsEvent(d){
  const ned=d.nextEarningsDate;
  if(!ned)return{inWindow:false,earningsDay:-1,expectedMove:0};
  try{
    const eDate=new Date(ned),now=new Date();
    const tradDays=Math.round((eDate-now)/864e5*252/365);
    if(tradDays>0&&tradDays<d.tradingDaysRemaining){
      return{inWindow:true,earningsDay:tradDays,expectedMove:(d.historicalEarningsMoveAvg||6)/100}
    }
  }catch(e){}
  return{inWindow:false,earningsDay:-1,expectedMove:0}
}

// Kelly Confidence — Kelly 1956, Thorp 1997
// f* = edge/σ², horizon-adjusted
function kellyConfidence(methods,S,vol,T){
  const ests=methods.map(m=>m.est);
  const mean=ests.reduce((a,b)=>a+b,0)/ests.length;
  const variance=ests.reduce((a,e)=>{const d=e-mean;return a+d*d},0)/ests.length;
  const cv=Math.sqrt(variance)/Math.abs(mean);
  const edge=mean/S-1;
  // σ² scaled to horizon: vol² × (T/252) — NOT raw annualized vol²
  const sigSq=vol*vol*Math.max(T||252,1)/252;
  const kellyFull=sigSq>0?edge/sigSq:0;
  const kellyHalf=kellyFull*.5;
  const absK=Math.abs(kellyHalf);
  const conf=clamp(absK*200,15,95);
  const agreement=Math.max(.2,1-cv*3);
  const bayesEst=S*(1-agreement)+mean*agreement;
  return{est:+bayesEst.toFixed(2),cv:+cv.toFixed(4),confidence:+conf.toFixed(1),
    kellyFull:+(kellyFull*100).toFixed(1),kellyHalf:+(kellyHalf*100).toFixed(1),
    agreement:+(agreement*100).toFixed(1),mean:+mean.toFixed(2),edge:+(edge*100).toFixed(2),
    label:conf>75?"HIGH":conf>50?"MODERATE":conf>30?"LOW":"VERY LOW"}
}

// James-Stein Shrinkage — James & Stein 1961
// θ̂_JS = ν + (1 − (m−2)σ²/‖y−ν‖²)₊ · (y − ν)
function jamesShrink(estimates,target){
  const m=estimates.length;if(m<3)return{estimates:[...estimates],coeff:1};
  const diffs=estimates.map(e=>e-target);
  const ssq=diffs.reduce((a,d)=>a+d*d,0);
  if(ssq===0)return{estimates:[...estimates],coeff:1};
  const mean=estimates.reduce((a,b)=>a+b,0)/m;
  const s2=estimates.reduce((a,e)=>{const d=e-mean;return a+d*d},0)/(m-1);
  const coeff=Math.max(0,1-(m-2)*s2/ssq);
  return{estimates:estimates.map((_,i)=>target+coeff*diffs[i]),coeff};
}

// Bootstrap CI — Efron 1979
// IMPORTANT: uses HORIZON weights (not IVW) to preserve signal disagreement
function bootstrapCI(methods,nBoot,seed){
  const r=lcg(seed),n=methods.length;
  const bootEns=new Float64Array(nBoot);
  for(let b=0;b<nBoot;b++){
    let sumW=0,sumWE=0;
    for(let i=0;i<n;i++){
      const j=Math.floor(r()*n);
      sumW+=methods[j].w;sumWE+=methods[j].w*methods[j].est;
    }
    bootEns[b]=sumW>0?sumWE/sumW:methods[0].est;
  }
  const sorted=Float64Array.from(bootEns).sort();
  const p=q=>sorted[Math.min(Math.floor(q/100*sorted.length),sorted.length-1)];
  const mean=bootEns.reduce((a,b)=>a+b,0)/nBoot;
  const vari=bootEns.reduce((a,e)=>{const d=e-mean;return a+d*d},0)/nBoot;
  return{mean:+mean.toFixed(2),std:+Math.sqrt(vari).toFixed(2),
    ci68:[+p(16).toFixed(2),+p(84).toFixed(2)],
    ci90:[+p(5).toFixed(2),+p(95).toFixed(2)],
    ci95:[+p(2.5).toFixed(2),+p(97.5).toFixed(2)]}
}

// Inverse-Variance Weighting (for point estimate only)
function inverseVarWeights(estimates){
  const mean=estimates.reduce((a,b)=>a+b,0)/estimates.length;
  const minVar=mean*mean*1e-6;
  const vars=estimates.map(e=>Math.max((e-mean)*(e-mean),minVar));
  const ivs=vars.map(v=>1/v);
  const sum=ivs.reduce((a,b)=>a+b,0);
  return ivs.map(iv=>iv/sum);
}

// Signal Decorrelation — discount correlated pairs
function decorrelateWeights(weights){
  const pairs=[[11,20,.6],[12,21,.5],[9,15,.5],[5,6,.4],[7,14,.5],[23,7,.3]];
  const adj=[...weights];
  for(const[i,j,rho]of pairs){
    if(i<adj.length&&j<adj.length){const d=1-rho*.5;adj[i]*=d;adj[j]*=d}
  }
  const sum=adj.reduce((a,b)=>a+b,0);
  return adj.map(w=>w/sum);
}

// Decomposed Prediction Interval
// Total PI = √(MC_tail² + boot_tail²) per tail (quadrature)
function predictionInterval(bootCI,mcDist,ensPoint){
  const modelStd=parseFloat(bootCI.std);
  const mcP5=parseFloat(mcDist.p5),mcP95=parseFloat(mcDist.p95);
  const mcP25=parseFloat(mcDist.p25),mcP75=parseFloat(mcDist.p75);
  const marketStd=+((mcP75-mcP25)/1.349).toFixed(2);
  const totalStd=+(Math.sqrt(modelStd*modelStd+marketStd*marketStd)).toFixed(2);

  // Per-tail quadrature (handles asymmetric distributions)
  const bLo90=ensPoint-bootCI.ci90[0], bHi90=bootCI.ci90[1]-ensPoint;
  const mLo90=ensPoint-mcP5, mHi90=mcP95-ensPoint;

  return{
    modelStd:+modelStd.toFixed(2), marketStd, totalStd,
    model90:bootCI.ci90,
    market90:[+mcP5.toFixed(2),+mcP95.toFixed(2)],
    total90:[+(ensPoint-Math.sqrt(mLo90*mLo90+bLo90*bLo90)).toFixed(2),
             +(ensPoint+Math.sqrt(mHi90*mHi90+bHi90*bHi90)).toFixed(2)],
    modelPct:+((modelStd*modelStd)/(totalStd*totalStd)*100).toFixed(1),
    marketPct:+((marketStd*marketStd)/(totalStd*totalStd)*100).toFixed(1),
    // Signal fidelity: how much do signals matter vs market noise?
    fidelity:modelStd/(totalStd||1)
  }
}

// ────────────────────────────────────────────
// §5  ENSEMBLE MODEL
// ────────────────────────────────────────────
// KEY v6 CHANGE: Vol-Adaptive Signal Scaling
// Signals produce ±1-6% adjustments calibrated for σ≈25% stocks.
// For high-vol stocks (σ=90%), we scale signal adjustments by
// volScale = max(σ, 0.15) / 0.25 so signals remain informative.
// This is ONLY applied to factor signals (S2-S20), NOT to:
//   - MJD Monte Carlo (already vol-aware)
//   - Earnings/Yield valuation (price-level, not return-based)
//   - Mean Reversion targets (absolute price levels)
//   - Analyst Consensus (external, already calibrated)
//   - Kelly Calibrator (computed from ensemble)
function runModel(d,isETF){
  const{currentPrice:S,annualizedVol:vol,beta,forwardEps:e1,forwardEps2:e2,
    high52w,low52w,analystAvgPt,analystHigh,analystLow,recentAnalystAvg,
    tradingDaysRemaining:T,dividendYield,expenseRatio}=d;
  const rf=(d.fedFundsRate||4.5)/100,mrp=.055,N=10000;
  const bEps=e1&&e2?.42*e1+.58*e2:e1?e1*1.05:0;

  // ═══════════════════════════════════════
  // VOL-ADAPTIVE SCALING (v6 core fix)
  // ═══════════════════════════════════════
  const VOL_REF=0.25; // calibration baseline (~average stock vol)
  const volScale=Math.max(vol,0.15)/VOL_REF;

  // ── Earnings Event ──
  const earnings=detectEarningsEvent(d);

  // ── S1: MJD Monte Carlo ──
  const reg=sigRegime(d);
  const mjdOpts={nu:5,lambda:2,jumpMu:-.02,jumpSig:.08,
    earningsJump:earnings.inWindow?earnings.expectedMove:0,
    earningsDay:earnings.inWindow?earnings.earningsDay:-1};
  const bullDrift=.25*Math.max(beta,1);
  const bearDrift=-.10*Math.max(beta,1);
  const bull=mjd(S,bullDrift,vol,T,N,42,mjdOpts);
  const base=mjd(S,rf+beta*mrp,vol,T,N,137,mjdOpts);
  const bear=mjd(S,bearDrift,vol,T,N,491,mjdOpts);
  const nB=Math.floor(N*reg.bW),nM=Math.floor(N*reg.mW),nE=Math.floor(N*reg.eW);
  const blend=new Float64Array(nB+nM+nE);let k=0;
  for(let i=0;i<nB;i++)blend[k++]=bull[i];
  for(let i=0;i<nM;i++)blend[k++]=base[i];
  for(let i=0;i<nE;i++)blend[k++]=bear[i];
  const mcR={est:+pctl(blend,50).toFixed(2),mean:+(blend.reduce((a,b)=>a+b,0)/blend.length).toFixed(2),
    p5:+pctl(blend,5).toFixed(2),p10:+pctl(blend,10).toFixed(2),p25:+pctl(blend,25).toFixed(2),
    p50:+pctl(blend,50).toFixed(2),p75:+pctl(blend,75).toFixed(2),p90:+pctl(blend,90).toFixed(2),
    p95:+pctl(blend,95).toFixed(2)};
  const pA=t=>{let c=0;for(let i=0;i<blend.length;i++)if(blend[i]>t)c++;return+((c/blend.length)*100).toFixed(1)};

  // ── S2: Earnings/Yield Valuation ──
  let earnEst,peS;
  if(!isETF&&bEps>0){
    const fp=Math.max(e1>0?S/e1:15,5);
    peS=[{label:"Bear("+Math.round(fp*.7)+"x)",m:Math.round(fp*.7),p:.20},
      {label:"Base("+Math.round(fp)+"x)",m:Math.round(fp),p:.45},
      {label:"Bull("+Math.round(fp*1.3)+"x)",m:Math.round(fp*1.3),p:.25},
      {label:"Mega("+Math.round(fp*1.6)+"x)",m:Math.round(fp*1.6),p:.10}];
    peS.forEach(s=>{s.price=+(bEps*s.m).toFixed(2)});
    earnEst=+peS.reduce((a,s)=>a+s.price*s.p,0).toFixed(2);
  } else {
    const dy=(dividendYield||1.5)/100,er=(expenseRatio||.2)/100,ny=dy-er,tY=T/252;
    peS=[{label:"Bear(5%)",m:5,p:.25,price:+(S*Math.pow(1.05+ny,tY)).toFixed(2)},
      {label:"Base(8%)",m:8,p:.45,price:+(S*Math.pow(1.08+ny,tY)).toFixed(2)},
      {label:"Bull(12%)",m:12,p:.20,price:+(S*Math.pow(1.12+ny,tY)).toFixed(2)},
      {label:"Mega(15%)",m:15,p:.10,price:+(S*Math.pow(1.15+ny,tY)).toFixed(2)}];
    earnEst=+peS.reduce((a,s)=>a+s.price*s.p,0).toFixed(2);
  }

  // ── S3: Mean Reversion ── uses SMA200 (NOT broken midpoint formula)
  const meanTarget=d.sma200||((high52w+low52w)/2);
  const revS=[{target:+(S*1.05).toFixed(2),prob:.15},{target:+meanTarget.toFixed(2),prob:.35},
    {target:+(low52w+.5*(high52w-low52w)).toFixed(2),prob:.30},
    {target:+(low52w+.75*(high52w-low52w)).toFixed(2),prob:.15},
    {target:+(high52w*.95).toFixed(2),prob:.05}];
  const revEst=+revS.reduce((a,s)=>a+s.target*s.prob,0).toFixed(2);

  // ── S4: Analyst Consensus ──
  const rec=recentAnalystAvg||(analystAvgPt?analystAvgPt*.85:S*1.05);
  const broad=analystAvgPt||S*1.08;
  const anEst=+(rec*.6+broad*.4).toFixed(2);

  // ── S5-S20: Factor signals ──
  const mom=sigMom(d),srs=sigSRS(d),volR=sigVol(d),inst=sigInst(d);
  const epsRev=sigEpsRevision(d),size=sigSize(d),val=sigValue(d),qual=sigQuality(d);
  const inv=sigInvestment(d),lowV=sigLowVol(d),pead=sigPEAD(d),season=sigSeason(d);
  const opts=sigOptionsSkew(d),insider=sigInsider(d),revAcc=sigRevAccel(d),fcf=sigFCF(d);
  const lev=sigLeverage(d),buyback=sigBuyback(d),macro=sigMacro(d);

  // ═══════════════════════════════════════
  // BUILD METHOD ARRAY
  // Methods 0-4: NO vol-scaling (already calibrated)
  // Methods 5-23: VOL-SCALED
  // ═══════════════════════════════════════
  const signalResults=[mom,srs,volR,inst,epsRev,size,val,qual,inv,lowV,pead,season,opts,insider,revAcc,fcf,lev,buyback,macro];
  const signalNames=["Momentum","Sector RS","Vol Regime","Inst. Sentiment","EPS Revision","Size Factor",
    "Value Factor","Quality","Investment","Low Vol","PEAD","Seasonality","Options Skew","Insider",
    "Rev Accel","FCF Yield","Leverage","Buyback","Macro Regime"];
  const signalColors=["#F97316","#14B8A6","#EF4444","#A78BFA","#06B6D4","#84CC16","#D946EF","#F43F5E",
    "#0EA5E9","#22D3EE","#FB923C","#A3E635","#C084FC","#2DD4BF","#FBBF24","#34D399","#F87171","#818CF8","#4ADE80"];

  // Vol-scale factor signals: amplify their deviation from S by volScale
  const scaledSignalEsts=signalResults.map(sig=>{
    const rawPctAdj=(sig.est-S)/S; // e.g. +0.03 for +3%
    const scaledPctAdj=rawPctAdj*volScale;
    return +(S*(1+scaledPctAdj)).toFixed(2);
  });

  const preMethods=[
    {n:"MJD Monte Carlo",est:mcR.est,c:"#3B82F6",src:"Merton 1976 + Student-t(ν=5)",volScaled:false},
    {n:isETF?"Yield & Growth":"Earnings Val",est:earnEst,c:"#10B981",src:"DCF / Gordon Growth",volScaled:false},
    {n:"Mean Reversion",est:revEst,c:"#F59E0B",src:"Ornstein-Uhlenbeck",volScaled:false},
    {n:"Analyst Consensus",est:anEst,c:"#8B5CF6",src:"Thomson Reuters I/B/E/S",volScaled:false},
    {n:"Regime Drift",est:reg.est,c:"#EC4899",src:"HMM proxy",volScaled:false},
    ...signalNames.map((n,i)=>({n,est:scaledSignalEsts[i],c:signalColors[i],
      src:n,volScaled:true,rawEst:signalResults[i].est}))
  ];

  // ── Winsorize at P5/P95 (Tukey 1977) ──
  const rawArr=preMethods.map(m=>m.est);
  const wLo=pctl(new Float64Array(rawArr),5),wHi=pctl(new Float64Array(rawArr),95);
  preMethods.forEach(m=>{m.est=+clamp(m.est,wLo,wHi).toFixed(2)});

  // ── Save RAW estimates for bootstrap BEFORE shrinkage ──
  const rawEstsForBoot=preMethods.map(m=>m.est);

  // ── James-Stein Shrinkage toward analyst consensus ──
  const js=jamesShrink(preMethods.map(m=>m.est),anEst);
  preMethods.forEach((m,i)=>{m.rawEst=m.rawEst||m.est;m.est=+js.estimates[i].toFixed(2)});

  // ── Kelly Calibrator (method 25) ──
  const kelly=kellyConfidence(preMethods,S,vol,T);
  const allMethods=[...preMethods,{n:"Kelly Calibrator",est:kelly.est,c:"#E879F9",src:"Kelly 1956 / Thorp 1997"}];

  // ── Weights: 40% horizon + 30% IVW + 30% interaction ──
  const all25Ests=allMethods.map(m=>m.est);
  const horizW=horizonWeights(T,isETF);
  const ivW=inverseVarWeights(all25Ests);
  const interMults=signalInteractions({mom,inst,val,qual,pead,epsRev,macro,lev,fcf,buyback});
  const interW=horizW.map((w,i)=>w*(interMults[i]||1));
  const iSum=interW.reduce((a,b)=>a+b,0);
  const interWN=interW.map(x=>x/iSum);
  const blendW=horizW.map((_,i)=>horizW[i]*.40+ivW[i]*.30+interWN[i]*.30);
  const finalW=decorrelateWeights(blendW);
  allMethods.forEach((m,i)=>{m.w=finalW[i]});

  // ── Ensemble Point Estimate ──
  const ens=+allMethods.reduce((a,m)=>a+m.est*m.w,0).toFixed(2);

  // ── Bootstrap CI on RAW estimates with HORIZON weights ──
  // Uses horizon weights (not IVW) to capture true signal disagreement
  const rawForBoot=rawEstsForBoot.map((est,i)=>({est,w:horizW[i]||1/rawEstsForBoot.length}));
  rawForBoot.push({est:kelly.mean,w:horizW[horizW.length-1]||.04});
  const boot=bootstrapCI(rawForBoot,5000,777);

  // ── Prediction Interval ──
  const pi=predictionInterval(boot,mcR,ens);

  // ── Probability Table ──
  const probs=[
    {label:"P(>$"+Math.round(S*1.25)+")",value:pA(S*1.25)},
    {label:"P(>$"+Math.round(S*1.5)+")",value:pA(S*1.5)},
    {label:"P(>$"+Math.round(S*2)+")",value:pA(S*2)},
    {label:"P(<$"+Math.round(S*.66)+")",value:+(100-pA(S*.66)).toFixed(1)},
    {label:"P(<$"+Math.round(S)+")",value:+(100-pA(S)).toFixed(1)}];

  return{isETF,volScale:+volScale.toFixed(2),
    ensemble:{point:ens,ret:((ens/S-1)*100).toFixed(1)},
    methods:allMethods,probs,peS,revS,mc:mcR,blendEps:+(bEps).toFixed(2),
    earnings,boot,pi,jsCoeff:+js.coeff.toFixed(4),
    analystData:{broad:+broad.toFixed(2),recent:+rec.toFixed(2),weighted:anEst,
      high:analystHigh||broad*1.3,low:analystLow||broad*.7},
    hist:buildHist(blend),
    signals:{reg,mom,srs,volR,inst,epsRev,size,val,qual,inv,lowV,pead,season,opts,insider,revAcc,fcf,lev,buyback,macro,kelly}};
}

// ────────────────────────────────────────────
// §6  DATA FETCH (Haiku + server-side web_search for LIVE data)
// ────────────────────────────────────────────
// Haiku is given the web_search tool so it can look up current
// prices server-side. This is NOT client-side web_search (which
// hangs in artifact sandbox). The search happens on Anthropic's
// servers during the API call — the artifact just waits for the
// HTTP response like any normal fetch.
async function fetchData(ticker,onS){
  onS("Fetching live "+ticker+" data...");
  const sys="You are a financial data API. FIRST use web_search to look up the current stock price and key metrics for the requested ticker. Then return ONLY a raw JSON object — no markdown, no backticks, no explanation. Start your final answer with { and end with }. Use null for fields you cannot find. The currentPrice MUST be from your web search, not from memory.";
  const prompt='Search for the current stock price and financial data for "'+ticker+'", then return the data as this exact JSON structure:\n'+
    '{"ticker":"'+ticker+'","companyName":"str","isETF":bool,'+
    '"currentPrice":n,"annualizedVol":decimal,"beta":n,'+
    '"forwardEps":n_or_null,"forwardEps2":n_or_null,'+
    '"high52w":n,"low52w":n,'+
    '"analystAvgPt":n_or_null,"analystHigh":n_or_null,"analystLow":n_or_null,'+
    '"recentAnalystAvg":n_or_null,"marketCap":"str","marketCapNum":n_billions,'+
    '"peRatio":n_or_null,"sectorAvgPE":n_or_null,'+
    '"priceToBook":n_or_null,"sectorAvgPB":n_or_null,'+
    '"roe":n_pct,"profitMargin":n_pct,'+
    '"capexToRevenue":n_pct,"revenueGrowthPct":n,"priorRevenueGrowthPct":n,'+
    '"fcfYield":n_pct,"debtToEquity":n,"interestCoverage":n,'+
    '"shareCountChange":n_yoy_pct,'+
    '"dividendYield":n,"expenseRatio":n,'+
    '"rsi14":n,"sma50":n,"sma200":n,'+
    '"shortInterest":n_pct,"putCallRatio":n,'+
    '"institutionalOwnership":n_pct,"vix":n,'+
    '"ivRank":n_0to100,"ivSkew":n,'+
    '"earningsRevision3m":n_pct,'+
    '"lastEarningsSurprise":n_pct,"daysSinceEarnings":n,'+
    '"insiderNetBuying":n_pct,'+
    '"stockPerformance3m":n_pct,"sectorPerformance3m":n_pct,'+
    '"sector":"str","aum":"str_or_null","topHoldings":"str_or_null",'+
    '"fedFundsRate":n,"yieldCurve2s10s":n,"creditSpreadHY":n,'+
    '"nextEarningsDate":"YYYY-MM-DD_or_null","historicalEarningsMoveAvg":n_pct,'+
    '"keyRisks":["r1","r2","r3"],"keyCatalysts":["c1","c2","c3"]}\n'+
    "isETF=true for ETFs (SPY,QQQ,XLK,ARKK,VTI,IWM). marketCapNum in billions. After searching, output ONLY the JSON object.";

  const res=await fetch("https://api.anthropic.com/v1/messages",{method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify({model:"claude-haiku-4-5-20251001",max_tokens:4096,system:sys,
      tools:[{type:"web_search_20250305",name:"web_search"}],
      messages:[{role:"user",content:prompt}]})});
  if(!res.ok){const e=await res.text().catch(()=>"?");throw new Error("API "+res.status+": "+e.slice(0,200))}

  const result=await res.json();onS("Parsing response...");
  // Haiku response may contain mixed blocks: text, web_search_tool_result, etc.
  // Extract ALL text blocks and concatenate to find the JSON
  const texts=(result.content||[]).filter(b=>b.type==="text"&&b.text).map(b=>b.text);
  if(!texts.length)throw new Error("No text in API response — Haiku may have only returned search results");
  const raw=texts.join("\n").trim();
  let depth=0,start=-1,end=-1;
  for(let i=0;i<raw.length;i++){if(raw[i]==="{"){if(depth===0)start=i;depth++}if(raw[i]==="}"){depth--;if(depth===0&&start!==-1){end=i;break}}}
  if(start===-1||end===-1)throw new Error("No JSON object found");
  let js=raw.substring(start,end+1).replace(/,(\s*[}\]])/g,"$1");
  let data;try{data=JSON.parse(js)}catch(e){throw new Error("JSON parse: "+e.message)}

  ["currentPrice","high52w","low52w"].forEach(f=>{if(data[f]==null||isNaN(Number(data[f])))throw new Error("Missing: "+f)});
  const numFields=["currentPrice","annualizedVol","beta","forwardEps","forwardEps2","high52w","low52w",
    "analystAvgPt","analystHigh","analystLow","recentAnalystAvg","peRatio","sectorAvgPE",
    "priceToBook","sectorAvgPB","roe","profitMargin","capexToRevenue","revenueGrowthPct",
    "priorRevenueGrowthPct","fcfYield","debtToEquity","interestCoverage","shareCountChange",
    "dividendYield","expenseRatio","rsi14","sma50","sma200","shortInterest","putCallRatio",
    "institutionalOwnership","vix","ivRank","ivSkew","earningsRevision3m","lastEarningsSurprise",
    "daysSinceEarnings","insiderNetBuying","stockPerformance3m","sectorPerformance3m","marketCapNum",
    "fedFundsRate","yieldCurve2s10s","creditSpreadHY","historicalEarningsMoveAvg"];
  numFields.forEach(k=>{if(data[k]!=null)data[k]=Number(data[k])});

  // Defaults for missing data
  if(!data.annualizedVol||data.annualizedVol<=0)data.annualizedVol=Math.min(Math.log(data.high52w/Math.max(data.low52w,.01))/1.67,1.5);
  if(!data.beta||data.beta<=0)data.beta=1;
  if(!data.vix)data.vix=18;
  if(!data.analystAvgPt)data.analystAvgPt=data.currentPrice*1.08;
  if(!data.analystHigh)data.analystHigh=data.analystAvgPt*1.3;
  if(!data.analystLow)data.analystLow=data.analystAvgPt*.7;
  if(!data.isETF&&data.forwardEps&&!data.forwardEps2)data.forwardEps2=data.forwardEps*1.1;
  if(!data.marketCapNum)data.marketCapNum=50;
  if(!data.fedFundsRate)data.fedFundsRate=4.5;
  if(data.yieldCurve2s10s==null)data.yieldCurve2s10s=.2;
  if(!data.creditSpreadHY)data.creditSpreadHY=3.5;

  onS("Running MJD ensemble...");
  return data;
}

// ────────────────────────────────────────────
// §7  USER INTERFACE
// ────────────────────────────────────────────
const F="'JetBrains Mono',Consolas,monospace";
const G="'DM Sans',-apple-system,BlinkMacSystemFont,sans-serif";
const FONT_URL="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600;700&display=swap";

function Badge({children,tone}){
  const colors={
    red:{bg:"rgba(239,68,68,.12)",fg:"#f87171",bd:"rgba(239,68,68,.2)"},
    amber:{bg:"rgba(245,158,11,.12)",fg:"#fbbf24",bd:"rgba(245,158,11,.2)"},
    green:{bg:"rgba(16,185,129,.12)",fg:"#34d399",bd:"rgba(16,185,129,.2)"},
    cyan:{bg:"rgba(6,182,212,.12)",fg:"#22d3ee",bd:"rgba(6,182,212,.2)"},
    purple:{bg:"rgba(139,92,246,.12)",fg:"#a78bfa",bd:"rgba(139,92,246,.2)"},
  };
  const c=colors[tone]||colors.cyan;
  return <span style={{fontSize:10,fontFamily:F,padding:"2px 8px",borderRadius:4,fontWeight:600,
    background:c.bg,color:c.fg,border:"1px solid "+c.bd,letterSpacing:".03em"}}>{children}</span>
}

function Card({title,children,accent}){
  return <div style={{borderRadius:12,padding:16,
    border:accent?"1px solid rgba(6,182,212,.2)":"1px solid rgba(51,65,85,.35)",
    background:accent?"linear-gradient(135deg,rgba(6,182,212,.04),rgba(15,23,42,.97))":"rgba(30,41,59,.4)",
    backdropFilter:"blur(6px)"}}>
    {title&&<div style={{fontSize:10,fontFamily:F,fontWeight:600,color:"#64748b",letterSpacing:".08em",
      textTransform:"uppercase",marginBottom:10}}>{title}</div>}
    {children}
  </div>
}

function CIBox({label,lo,hi,color,sub}){
  return <div style={{flex:1,textAlign:"center",borderRadius:10,padding:"10px 6px",
    border:"1px solid "+color+"33",background:color+"0a"}}>
    <div style={{fontSize:9,fontFamily:F,color:"#94a3b8",letterSpacing:".06em",fontWeight:600,marginBottom:4}}>{label}</div>
    <div style={{fontFamily:F,fontSize:13,fontWeight:700,color}}>
      ${typeof lo==="number"?lo.toFixed(2):lo} – ${typeof hi==="number"?hi.toFixed(2):hi}
    </div>
    {sub&&<div style={{fontSize:8,fontFamily:F,color:"#475569",marginTop:3}}>{sub}</div>}
  </div>
}

export default function StockPredictor(){
  const[ticker,setTicker]=useState("");
  const[targetDate,setTargetDate]=useState(()=>{const d=new Date();d.setFullYear(d.getFullYear()+1);return d.toISOString().slice(0,10)});
  const[status,setStatus]=useState("");
  const[result,setResult]=useState(null);
  const[error,setError]=useState("");
  const[running,setRunning]=useState(false);
  const[tab,setTab]=useState("overview");

  const run=useCallback(async()=>{
    if(!ticker.trim()||running)return;
    setRunning(true);setResult(null);setError("");setStatus("Initializing...");
    try{
      const data=await fetchData(ticker.trim().toUpperCase(),setStatus);
      const today=new Date(),target=new Date(targetDate);
      const calDays=Math.max(Math.round((target-today)/864e5),30);
      data.tradingDaysRemaining=Math.round(calDays*252/365);
      const r=runModel(data,!!data.isETF);
      setResult({...r,data});setStatus("");
    }catch(e){setError(e.message);setStatus("")}
    finally{setRunning(false)}
  },[ticker,targetDate,running]);

  const R=result, D=R?.data;

  return(
    <div style={{minHeight:"100vh",background:"linear-gradient(180deg,#0a0f1a 0%,#0f172a 40%,#111827 100%)",
      color:"#e2e8f0",fontFamily:G,padding:0}}>
      <link href={FONT_URL} rel="stylesheet"/>

      {/* ── HEADER ── */}
      <div style={{padding:"16px 20px",borderBottom:"1px solid rgba(51,65,85,.3)",
        display:"flex",alignItems:"center",gap:12}}>
        <div style={{width:36,height:36,borderRadius:8,background:"linear-gradient(135deg,#0ea5e9,#6366f1)",
          display:"flex",alignItems:"center",justifyContent:"center",fontSize:11,fontFamily:F,fontWeight:700,color:"#fff"}}>AV</div>
        <div>
          <div style={{fontSize:15,fontWeight:700,letterSpacing:".02em"}}>A.V.I.S. <span style={{color:"#64748b",fontWeight:400}}>Stock & ETF Predictor v6</span></div>
          <div style={{fontSize:10,fontFamily:F,color:"#475569"}}>MJD · Student-t(ν=5) · James-Stein · Bootstrap CI · Vol-Adaptive Signals · Live Web Search · Decorrelated Ensemble</div>
        </div>
      </div>

      {/* ── INPUT ── */}
      <div style={{padding:"16px 20px",display:"flex",gap:12,alignItems:"flex-end",flexWrap:"wrap"}}>
        <div style={{flex:"1 1 200px"}}>
          <div style={{fontSize:10,fontFamily:F,color:"#64748b",marginBottom:4,letterSpacing:".06em",fontWeight:600}}>TICKER</div>
          <input value={ticker} onChange={e=>setTicker(e.target.value.toUpperCase())}
            onKeyDown={e=>e.key==="Enter"&&run()}
            placeholder="MSFT, IREN, SPY..."
            style={{width:"100%",boxSizing:"border-box",padding:"10px 14px",borderRadius:8,border:"1px solid rgba(51,65,85,.5)",
              background:"rgba(15,23,42,.8)",color:"#f1f5f9",fontFamily:F,fontSize:14,fontWeight:600,outline:"none"}}/>
        </div>
        <div style={{flex:"1 1 200px"}}>
          <div style={{fontSize:10,fontFamily:F,color:"#64748b",marginBottom:4,letterSpacing:".06em",fontWeight:600}}>TARGET DATE</div>
          <input type="date" value={targetDate} onChange={e=>setTargetDate(e.target.value)}
            style={{width:"100%",boxSizing:"border-box",padding:"10px 14px",borderRadius:8,border:"1px solid rgba(51,65,85,.5)",
              background:"rgba(15,23,42,.8)",color:"#f1f5f9",fontFamily:F,fontSize:14,outline:"none"}}/>
        </div>
        <button onClick={run} disabled={running}
          style={{padding:"10px 28px",borderRadius:8,border:"none",fontFamily:F,fontSize:13,fontWeight:700,
            cursor:running?"wait":"pointer",letterSpacing:".04em",
            background:running?"#334155":"linear-gradient(135deg,#ef4444,#dc2626)",color:"#fff",
            boxShadow:running?"none":"0 4px 14px rgba(239,68,68,.25)"}}>
          {running?"Running...":"Predict"}
        </button>
      </div>

      {status&&<div style={{padding:"0 20px",fontFamily:F,fontSize:11,color:"#22d3ee",marginBottom:8}}>⟳ {status}</div>}
      {error&&<div style={{padding:"0 20px",fontFamily:F,fontSize:11,color:"#f87171",marginBottom:8}}>✗ {error}</div>}

      {R&&D&&(()=>{
        const S=D.currentPrice,ens=R.ensemble.point,ret=R.ensemble.ret;
        const pi=R.pi,boot=R.boot,mc=R.mc;
        const sig=R.signals;
        const fidelity=pi.fidelity;
        const fidelityLabel=fidelity>.15?"HIGH":fidelity>.05?"MODERATE":"LOW";
        const fidelityTone=fidelity>.15?"green":fidelity>.05?"amber":"red";

        return <div style={{padding:"0 20px 40px"}}>
          {/* ── STOCK HEADER ── */}
          <div style={{display:"flex",justifyContent:"space-between",alignItems:"baseline",marginBottom:20,flexWrap:"wrap",gap:8}}>
            <div>
              <span style={{fontSize:28,fontWeight:700}}>{D.ticker}</span>
              <span style={{fontSize:14,color:"#94a3b8",marginLeft:10}}>{D.companyName}</span>
              <div style={{fontSize:11,fontFamily:F,color:"#475569",marginTop:2}}>
                {targetDate} · {D.tradingDaysRemaining}d · {D.sector}
                {R.volScale>1.5&&<span style={{color:"#f59e0b"}}> · VolScale: {R.volScale}×</span>}
              </div>
            </div>
            <div style={{textAlign:"right"}}>
              <div style={{fontSize:28,fontWeight:700,fontFamily:F}}>${S.toFixed(2)}</div>
              <div style={{fontSize:11,fontFamily:F,color:"#64748b"}}>{D.marketCap} mkt cap</div>
            </div>
          </div>

          {/* ── PREDICTION HERO ── */}
          <Card accent>
            <div style={{textAlign:"center"}}>
              <div style={{fontSize:10,fontFamily:F,color:"#64748b",letterSpacing:".08em",marginBottom:8}}>
                25-METHOD MJD ENSEMBLE · JAMES-STEIN SHRINKAGE · VOL-ADAPTIVE</div>
              <div style={{fontSize:48,fontWeight:700,fontFamily:F,
                color:ret>=0?"#22d3ee":"#f87171",lineHeight:1.1}}>
                ${ens.toFixed(2)}
              </div>
              <div style={{fontSize:16,fontFamily:F,color:ret>=0?"#34d399":"#f87171",marginTop:4}}>
                {ret>=0?"+":""}{ret}%
              </div>

              {/* ── 3-TIER CI/PI ── */}
              <div style={{display:"flex",gap:8,marginTop:16,flexWrap:"wrap"}}>
                <CIBox label="MODEL CI (90%)" lo={pi.model90[0]} hi={pi.model90[1]}
                  color="#22d3ee" sub="Signal disagreement"/>
                <CIBox label="MARKET (P5–P95)" lo={mc.p5} hi={mc.p95}
                  color="#f59e0b" sub="MJD path randomness"/>
                <CIBox label="TOTAL PI (90%)" lo={pi.total90[0]} hi={pi.total90[1]}
                  color="#a78bfa" sub="Model ⊕ market combined"/>
              </div>

              {/* ── SIGMA DECOMPOSITION ── */}
              <div style={{fontSize:10,fontFamily:F,color:"#64748b",marginTop:10}}>
                σ_model: ${pi.modelStd} ({pi.modelPct}%) · σ_market: ${pi.marketStd} ({pi.marketPct}%) · σ_total: ${pi.totalStd}
              </div>

              {/* ── BADGES ── */}
              <div style={{display:"flex",gap:6,justifyContent:"center",marginTop:10,flexWrap:"wrap"}}>
                <Badge tone={sig.reg.regime==="BULL"||sig.reg.regime==="RECOVERY"?"green":
                  sig.reg.regime==="DETERIORATING"||sig.reg.regime==="BEAR"?"red":"amber"}>
                  {sig.reg.regime}</Badge>
                <Badge tone="cyan">Mom: {sig.mom.label}</Badge>
                <Badge tone="purple">Macro: {sig.macro.label}</Badge>
                <Badge tone="amber">Kelly: {sig.kelly.confidence}%</Badge>
                <Badge tone={fidelityTone}>Fidelity: {fidelityLabel}</Badge>
              </div>

              {/* ── VOL-SCALE NOTICE ── */}
              {R.volScale>1.5&&<div style={{fontSize:10,fontFamily:F,color:"#f59e0b",marginTop:8}}>
                ⚡ Vol-Adaptive: signals scaled {R.volScale}× (σ={((D.annualizedVol||.3)*100).toFixed(0)}% vs 25% baseline)
              </div>}

              <div style={{fontSize:9,fontFamily:F,color:"#334155",marginTop:10}}>
                MJD · Student-t(ν=5) · James-Stein · 5K bootstrap · Decorrelated · Vol-Scaled ·
                Live data via web search · Not investment advice
              </div>
            </div>
          </Card>

          {/* ── TABS ── */}
          <div style={{display:"flex",gap:0,marginTop:20,borderBottom:"1px solid rgba(51,65,85,.3)"}}>
            {["overview","methods","engine","signals","risks"].map(t=>
              <button key={t} onClick={()=>setTab(t)}
                style={{flex:1,padding:"10px 0",background:"none",border:"none",
                  borderBottom:tab===t?"2px solid #22d3ee":"2px solid transparent",
                  color:tab===t?"#f1f5f9":"#64748b",fontFamily:F,fontSize:11,fontWeight:600,
                  cursor:"pointer",letterSpacing:".04em",textTransform:"capitalize"}}>{
                t==="methods"?"All 25 Methods":t==="engine"?"Engine Details":t==="signals"?"Signal Deep-Dive":
                t==="risks"?"Risks":t.charAt(0).toUpperCase()+t.slice(1)
              }</button>
            )}
          </div>

          {/* ── TAB: OVERVIEW ── */}
          {tab==="overview"&&<div style={{marginTop:16}}>
            {/* MJD Distribution */}
            <Card title={"MJD Distribution ("+((D.tradingDaysRemaining<=252?"10K":"10K"))+"×3 · Student-t · Poisson jumps)"}>
              <div style={{display:"flex",alignItems:"flex-end",gap:1,height:100}}>
                {R.hist.map((b,i)=>{
                  const maxD=Math.max(...R.hist.map(x=>parseFloat(x.d)));
                  const h=maxD>0?parseFloat(b.d)/maxD*100:0;
                  const bc=b.c;
                  const col=bc<S?"#ef4444":bc<ens?"#f59e0b":"#22d3ee";
                  return <div key={i} style={{flex:1,height:h+"%",background:col,borderRadius:"2px 2px 0 0",
                    opacity:.7,minWidth:1}}/>
                })}
              </div>
              <div style={{display:"flex",justifyContent:"space-between",marginTop:4}}>
                <span style={{fontSize:10,fontFamily:F,color:"#64748b"}}>${R.hist[0]?.c}</span>
                <span style={{fontSize:10,fontFamily:F,color:"#475569"}}>${S} now</span>
                <span style={{fontSize:10,fontFamily:F,color:"#22d3ee"}}>${ens.toFixed(0)} pred</span>
                <span style={{fontSize:10,fontFamily:F,color:"#64748b"}}>${R.hist[R.hist.length-1]?.c}</span>
              </div>
              {/* Probability table */}
              <div style={{display:"flex",gap:4,marginTop:12,flexWrap:"wrap"}}>
                {R.probs.map((p,i)=><div key={i} style={{flex:1,minWidth:70,textAlign:"center",
                  padding:"6px 4px",borderRadius:6,background:"rgba(15,23,42,.5)",border:"1px solid rgba(51,65,85,.3)"}}>
                  <div style={{fontSize:9,fontFamily:F,color:"#64748b"}}>{p.label}</div>
                  <div style={{fontSize:14,fontFamily:F,fontWeight:700,
                    color:parseFloat(p.value)>50?"#34d399":parseFloat(p.value)>25?"#fbbf24":"#f87171"}}>{p.value}%</div>
                </div>)}
              </div>
            </Card>
          </div>}

          {/* ── TAB: ALL 25 METHODS ── */}
          {tab==="methods"&&<div style={{marginTop:16}}>
            <Card title="25-Method Estimates (sorted by weight)">
              {[...R.methods].sort((a,b)=>b.w-a.w).map((m,i)=>
                <div key={i} style={{display:"flex",alignItems:"center",gap:8,padding:"5px 0",
                  borderBottom:"1px solid rgba(51,65,85,.15)"}}>
                  <div style={{width:4,height:20,borderRadius:2,background:m.c}}/>
                  <div style={{flex:1,fontSize:12,fontFamily:G,fontWeight:500}}>{m.n}</div>
                  <div style={{fontSize:11,fontFamily:F,color:"#94a3b8",minWidth:40,textAlign:"right"}}>
                    {(m.w*100).toFixed(1)}%
                  </div>
                  <div style={{fontSize:13,fontFamily:F,fontWeight:700,minWidth:65,textAlign:"right",
                    color:m.est>S?"#34d399":m.est<S?"#f87171":"#94a3b8"}}>
                    ${m.est}
                  </div>
                  <div style={{fontSize:10,fontFamily:F,color:m.est>S?"#34d399":"#f87171",minWidth:45,textAlign:"right"}}>
                    {((m.est/S-1)*100).toFixed(1)}%
                  </div>
                  {m.volScaled&&<span style={{fontSize:8,fontFamily:F,color:"#f59e0b"}}>V</span>}
                </div>
              )}
              <div style={{fontSize:9,fontFamily:F,color:"#475569",marginTop:8}}>
                "V" = vol-scaled signal (×{R.volScale}). Weights: 40% horizon + 30% IVW + 30% interaction · decorrelated.
              </div>
            </Card>
          </div>}

          {/* ── TAB: ENGINE DETAILS ── */}
          {tab==="engine"&&<div style={{marginTop:16,display:"flex",flexDirection:"column",gap:12}}>
            <Card title="MJD Parameters">
              <div style={{fontFamily:F,fontSize:11,color:"#94a3b8",lineHeight:1.8}}>
                <div>Price: ${S} · Vol: {((D.annualizedVol||.3)*100).toFixed(0)}% · Beta: {D.beta}</div>
                <div>Drift (base): rf({((D.fedFundsRate||4.5)/100*100).toFixed(1)}%) + β({D.beta})×MRP(5.5%) = {((D.fedFundsRate||4.5)/100+D.beta*.055).toFixed(3)}</div>
                <div>Bull: +{(.25*Math.max(D.beta,1)*100).toFixed(1)}% · Bear: {(-.10*Math.max(D.beta,1)*100).toFixed(1)}%</div>
                <div>Student-t: ν=5 · Jumps: λ=2, μ_J=−2%, σ_J=8%</div>
                <div>Paths: 10K×3 scenarios · Blend: {sig.reg.regime} ({(sig.reg.bW*100).toFixed(0)}/{(sig.reg.mW*100).toFixed(0)}/{(sig.reg.eW*100).toFixed(0)})</div>
              </div>
            </Card>
            <Card title="James-Stein Shrinkage">
              <div style={{fontFamily:F,fontSize:11,color:"#94a3b8",lineHeight:1.8}}>
                <div>θ̂_JS = ν + {R.jsCoeff} × (y − ν)</div>
                <div>Target (analyst consensus): ${R.analystData.weighted}</div>
                <div>Shrinkage coefficient: {R.jsCoeff} {R.jsCoeff>.9?"(light)":R.jsCoeff>.5?"(moderate)":"(heavy)"}</div>
              </div>
            </Card>
            <Card title="Vol-Adaptive Scaling (v6)">
              <div style={{fontFamily:F,fontSize:11,color:"#94a3b8",lineHeight:1.8}}>
                <div>Stock vol: {((D.annualizedVol||.3)*100).toFixed(0)}% · Baseline: 25% · Scale: {R.volScale}×</div>
                <div>Signals 5–23 adjusted by ×{R.volScale} to match vol regime</div>
                <div>Methods 0–4 (MJD, Earnings, MeanRev, Analyst, Regime) unscaled</div>
                {R.volScale>2&&<div style={{color:"#f59e0b"}}>⚠ High-vol stock: signals amplified significantly. Model CI wider than usual.</div>}
              </div>
            </Card>
            <Card title="Bootstrap CI (5K resamples)">
              <div style={{fontFamily:F,fontSize:11,color:"#94a3b8",lineHeight:1.8}}>
                <div>Bootstrap mean: ${boot.mean} · std: ${boot.std}</div>
                <div>CI 68%: ${boot.ci68[0]} – ${boot.ci68[1]}</div>
                <div>CI 90%: ${boot.ci90[0]} – ${boot.ci90[1]}</div>
                <div>CI 95%: ${boot.ci95[0]} – ${boot.ci95[1]}</div>
                <div style={{color:"#475569",fontSize:10}}>Bootstrap uses RAW estimates with horizon weights (not IVW)</div>
              </div>
            </Card>
            <Card title="Kelly Criterion">
              <div style={{fontFamily:F,fontSize:11,color:"#94a3b8",lineHeight:1.8}}>
                <div>Edge: {sig.kelly.edge}% · σ²_horizon: {(D.annualizedVol*D.annualizedVol*D.tradingDaysRemaining/252).toFixed(4)}</div>
                <div>Full Kelly: {sig.kelly.kellyFull}% · Half Kelly: {sig.kelly.kellyHalf}%</div>
                <div>Confidence: {sig.kelly.confidence}% · Agreement: {sig.kelly.agreement}%</div>
              </div>
            </Card>
          </div>}

          {/* ── TAB: SIGNAL DEEP-DIVE ── */}
          {tab==="signals"&&<div style={{marginTop:16,display:"flex",flexDirection:"column",gap:8}}>
            {[
              ["Regime",sig.reg,"Comp: "+sig.reg.comp+" · Blend: "+sig.reg.bW+"/"+sig.reg.mW+"/"+sig.reg.eW],
              ["Momentum",sig.mom,"RSI: "+sig.mom.rsi+" · Signal: "+sig.mom.signal],
              ["Macro",sig.macro,"Score: "+sig.macro.comp+"% · "+sig.macro.label],
              ["Kelly",sig.kelly,"Conf: "+sig.kelly.confidence+"% · CV: "+sig.kelly.cv],
              ["Value",sig.val,"Score: "+sig.val.score+" · PE: "+sig.val.pe+"/"+sig.val.secPe],
              ["Quality",sig.qual,"ROE: "+sig.qual.roe+"% · Margin: "+sig.qual.margin+"%"],
              ["EPS Rev",sig.epsRev,"Revision: "+sig.epsRev.revision+"%"],
              ["PEAD",sig.pead,"Surprise: "+sig.pead.surprise+"% · Days: "+sig.pead.days],
              ["Leverage",sig.lev,"D/E: "+sig.lev.de+" · IC: "+sig.lev.ic+"×"],
              ["Insider",sig.insider,"Net: "+sig.insider.net+"%"],
              ["FCF",sig.fcf,"Yield: "+sig.fcf.yield+"%"],
              ["Buyback",sig.buyback,"Δ shares: "+sig.buyback.change+"%"],
              ["Rev Accel",sig.revAcc,"Accel: "+sig.revAcc.accel+"%"],
            ].map(([name,s,detail],i)=>
              <div key={i} style={{display:"flex",alignItems:"center",gap:8,padding:"6px 10px",
                borderRadius:6,background:"rgba(15,23,42,.4)",border:"1px solid rgba(51,65,85,.2)"}}>
                <div style={{fontFamily:G,fontSize:12,fontWeight:600,minWidth:80}}>{name}</div>
                <Badge tone={s.label==="BULLISH"||s.label==="BULL"||s.label==="SUPPORTIVE"||s.label==="HIGH QUALITY"||s.label==="FORTRESS"?"green":
                  s.label==="BEARISH"||s.label==="BEAR"||s.label==="RESTRICTIVE"||s.label==="FRAGILE"?"red":"amber"}>
                  {s.label||"—"}</Badge>
                <div style={{flex:1,fontFamily:F,fontSize:10,color:"#64748b",textAlign:"right"}}>{detail}</div>
                <div style={{fontFamily:F,fontSize:12,fontWeight:700,color:"#e2e8f0",minWidth:55,textAlign:"right"}}>${s.est}</div>
              </div>
            )}
          </div>}

          {/* ── TAB: RISKS ── */}
          {tab==="risks"&&<div style={{marginTop:16,display:"flex",flexDirection:"column",gap:12}}>
            <Card title="Key Risks">
              {(D.keyRisks||["No risk data"]).map((r,i)=>
                <div key={i} style={{fontFamily:G,fontSize:12,color:"#f87171",padding:"4px 0",
                  borderBottom:"1px solid rgba(51,65,85,.15)"}}>⚠ {r}</div>
              )}
            </Card>
            <Card title="Key Catalysts">
              {(D.keyCatalysts||["No catalyst data"]).map((c,i)=>
                <div key={i} style={{fontFamily:G,fontSize:12,color:"#34d399",padding:"4px 0",
                  borderBottom:"1px solid rgba(51,65,85,.15)"}}>◆ {c}</div>
              )}
            </Card>
            <Card title="Earnings Event">
              <div style={{fontFamily:F,fontSize:11,color:"#94a3b8"}}>
                {R.earnings.inWindow
                  ?`Next earnings in ~${R.earnings.calDays} calendar days. Expected move: ±${(R.earnings.expectedMove*100).toFixed(1)}%. Modeled as MJD jump.`
                  :"No earnings event within forecast window."}
              </div>
            </Card>
            <Card title="Model Limitations">
              <div style={{fontFamily:F,fontSize:10,color:"#475569",lineHeight:1.8}}>
                <div>• Live data fetched via web search — may have slight delay vs real-time feeds</div>
                <div>• Signal fidelity: {fidelityLabel} — model uncertainty is {pi.modelPct}% of total</div>
                <div>• Student-t(ν=5) fat tails increase tail probabilities vs Gaussian</div>
                <div>• Poisson jumps (λ=2) model rare events but not black swans</div>
                <div>• James-Stein shrinkage (coeff={R.jsCoeff}) biases toward analyst consensus</div>
                {R.volScale>1.5&&<div style={{color:"#f59e0b"}}>• Vol-scaling ({R.volScale}×) amplifies signal estimates for this high-vol name</div>}
                <div>• NOT investment advice. This is a quantitative research tool.</div>
              </div>
            </Card>
          </div>}
        </div>
      })()}
    </div>
  );
}
