/*
 * The original code is under the following copyright:
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE_GS.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 *
 * The modifications of the code are under the following copyright:
 * Copyright (C) 2024, University of Liege, KAUST and University of Oxford
 * TELIM research group, http://www.telecom.ulg.ac.be/
 * IVUL research group, https://ivul.kaust.edu.sa/
 * VGG research group, https://www.robots.ox.ac.uk/~vgg/
 * All rights reserved.
 * The modifications are under the LICENSE.md file.
 *
 * For inquiries contact jan.held@uliege.be
 */
// Experimental forward-only WebGPU modification of the release triangle splatting formulas.
// CPU preprocess/tile sort + WebGPU per-pixel composition, single center sample.
const TILE = 16;
const compute = /* wgsl */`
struct Triangle { uv01: vec4f, uv2sp: vec4f, e0: vec4f, e1: vec4f, e2: vec4f, c0: vec4f, c1: vec4f, c2: vec4f };
struct Params { dims: vec4u, bg: vec4f };
@group(0) @binding(0) var<storage,read> tris: array<Triangle>;
@group(0) @binding(1) var<storage,read> offsets: array<u32>;
@group(0) @binding(2) var<storage,read> ids: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage,read_write> pixels: array<vec4f>;
@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
 if (gid.x >= params.dims.x || gid.y >= params.dims.y) { return; }
 let p=vec2f(f32(gid.x),f32(gid.y));
 let tile=(gid.y/16u)*params.dims.z+gid.x/16u;
 var T=1.0; var rgb=vec3f(0.0);
 for(var i=offsets[tile]; i<offsets[tile+1u]; i++) {
  let a=tris[ids[i]];
  let d=vec3f(dot(a.e0.xy,p)+a.e0.z, dot(a.e1.xy,p)+a.e1.z, dot(a.e2.xy,p)+a.e2.z);
  if(any(d>vec3f(0.0))) {continue;}
  let phi=max(max(d.x,d.y),d.z)*a.uv2sp.w;
  let alpha=min(0.999,a.e0.w*max(0.0,pow(max(phi,0.0),a.uv2sp.z)));
  if(alpha < 1.0/255.0) {continue;}
  let nextT=T*(1.0-alpha);
  // Match CUDA: do not accumulate the contributor which crosses this cutoff.
  if(nextT < 0.0001) {break;}
  let u0=a.uv01.xy; let u1=a.uv01.zw; let u2=a.uv2sp.xy;
  let v0=u1-u0; let v1=u2-u0; let v2=p-u0;
  let denom=v0.x*v1.y-v1.x*v0.y;
  if(abs(denom)<1e-12) {continue;}
  let b0=(v2.x*v1.y-v1.x*v2.y)/denom;
  let b1=(-v2.x*v0.y+v0.x*v2.y)/denom;
  let c=(1.0-b0-b1)*a.c0.xyz+b0*a.c1.xyz+b1*a.c2.xyz;
  rgb += c*(T*alpha); T=nextT;
 }
 pixels[gid.y*params.dims.x+gid.x]=vec4f(rgb+T*params.bg.xyz,1.0-T);
}`;
const display = /* wgsl */`
struct Params { dims: vec4u, bg: vec4f };
@group(0) @binding(0) var<storage,read> pixels: array<vec4f>;
@group(0) @binding(1) var<uniform> params: Params;
@vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4f {
 var p=array<vec2f,3>(vec2f(-1.0,-1.0),vec2f(3.0,-1.0),vec2f(-1.0,3.0));
 return vec4f(p[i],0.0,1.0);
}
@fragment fn fs(@builtin(position) p:vec4f)->@location(0) vec4f {
 return vec4f(pixels[u32(p.y)*params.dims.x+u32(p.x)].xyz,1.0);
}`;
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));

// viewMatrix and projectionMatrix are column-major; projectionMatrix is the
// full world-to-clip transform, not merely the camera projection matrix.
export function preprocess(scene,width,height) {
 const {vertices,faces,colors,weights,viewMatrix:vm,projectionMatrix:pm}=scene;
 const n=vertices.length/3,count=faces.length/3;
 if(!Number.isInteger(n)||!Number.isInteger(count)||colors.length!==n*3||weights.length!==n||vm.length!==16||pm.length!==16) throw Error('Invalid geometry, color, weight, or matrix dimensions');
 if(!Number.isInteger(width)||!Number.isInteger(height)||width<1||height<1) throw Error('Invalid image dimensions');
 const tx=Math.ceil(width/TILE),ty=Math.ceil(height/TILE),nt=tx*ty;
 const packed=new Float32Array(count*32),depths=new Float64Array(count);
 const bounds=new Uint32Array(count*4),counts=new Uint32Array(nt),idx=new Uint32Array(3),x=new Float64Array(3),y=new Float64Array(3),z=new Float64Array(3),ux=new Float64Array(3),uy=new Float64Array(3),edges=new Float64Array(9);
 const margin=scene.bboxMargin??5, sf=scene.sigmas,of=scene.overlaps;
 let skipped=0,visible=0;
 for(let f=0;f<count;f++) {
  for(let j=0;j<3;j++){const id=faces[3*f+j];if(!Number.isInteger(id)||id<0||id>=n)throw Error(`Invalid face ${f}`);idx[j]=id;const q=3*id;x[j]=vertices[q];y[j]=vertices[q+1];z[j]=vertices[q+2];if(!Number.isFinite(x[j])||!Number.isFinite(y[j])||!Number.isFinite(z[j])||!Number.isFinite(colors[q])||!Number.isFinite(colors[q+1])||!Number.isFinite(colors[q+2]))throw Error(`Invalid face parameters ${f}`);}
  const weight=Math.min(weights[idx[0]],weights[idx[1]],weights[idx[2]]),sigma=typeof sf==='number'?sf:sf?sf[f]:0.5,overlap=typeof of==='number'?of:of?of[f]:0.7;
  if(!Number.isFinite(weight)||!Number.isFinite(sigma)||sigma<=0||!Number.isFinite(overlap)||overlap<0||overlap>=1)throw Error(`Invalid face parameters ${f}`);
  if(weight<0.01){skipped++;continue;}
  const cx=(x[0]+x[1]+x[2])/3,cy=(y[0]+y[1]+y[2])/3,cz=(z[0]+z[1]+z[2])/3;
  const vx=vm[0]*cx+vm[4]*cy+vm[8]*cz+vm[12],vy=vm[1]*cx+vm[5]*cy+vm[9]*cz+vm[13],vz=vm[2]*cx+vm[6]*cy+vm[10]*cz+vm[14];
  if(vz<=0.002){skipped++;continue;}
  const ax=x[1]-x[0],ay=y[1]-y[0],az=z[1]-z[0],bx=x[2]-x[0],by=y[2]-y[0],bz=z[2]-z[0];
  const nx=ay*bz-az*by,ny=az*bx-ax*bz,nz=ax*by-ay*bx;
  const nxv=vm[0]*nx+vm[4]*ny+vm[8]*nz,nyv=vm[1]*nx+vm[5]*ny+vm[9]*nz,nzv=vm[2]*nx+vm[6]*ny+vm[10]*nz;
  const cos=(nxv*vx+nyv*vy+nzv*vz)/(Math.max(Math.hypot(nxv,nyv,nzv),1e-4)*Math.max(Math.hypot(vx,vy,vz),1e-4));
  if(Math.abs(cos)<0.001){skipped++;continue;}
  const scale=1/(1-Math.pow(clamp(overlap,0,0.9999),1/sigma));
  const cw=pm[3]*cx+pm[7]*cy+pm[11]*cz+pm[15]+1e-7;
  const pcx=(((pm[0]*cx+pm[4]*cy+pm[8]*cz+pm[12])/cw+1)*width-1)*0.5,pcy=(((pm[1]*cx+pm[5]*cy+pm[9]*cz+pm[13])/cw+1)*height-1)*0.5;
  let bad=!Number.isFinite(pcx)||!Number.isFinite(pcy),dist=0;
  for(let j=0;j<3;j++){const px=(x[j]-cx)*scale+cx,py=(y[j]-cy)*scale+cy,pz=(z[j]-cz)*scale+cz,pw=pm[3]*px+pm[7]*py+pm[11]*pz+pm[15]+1e-7;ux[j]=(((pm[0]*px+pm[4]*py+pm[8]*pz+pm[12])/pw+1)*width-1)*0.5;uy[j]=(((pm[1]*px+pm[5]*py+pm[9]*pz+pm[13])/pw+1)*height-1)*0.5;if(!Number.isFinite(ux[j])||!Number.isFinite(uy[j]))bad=true;dist=Math.max(dist,Math.hypot(ux[j]-pcx,uy[j]-pcy));}
  if(bad){skipped++;continue;}
  const a=Math.hypot(ux[1]-ux[2],uy[1]-uy[2]),b=Math.hypot(ux[0]-ux[2],uy[0]-uy[2]),c=Math.hypot(ux[0]-ux[1],uy[0]-uy[1]),sum=a+b+c;
  if(!Number.isFinite(sum)||sum<=0){skipped++;continue;}
  const icx=(a*ux[0]+b*ux[1]+c*ux[2])/sum,icy=(a*uy[0]+b*uy[1]+c*uy[2])/sum;
  let minDist=-Infinity,valid=true;
  for(let k=0;k<3;k++){const next=(k+1)%3,ex=uy[next]-uy[k],ey=-(ux[next]-ux[k]),len=Math.hypot(ex,ey);if(len===0){valid=false;break;}let enX=ex/len,enY=ey/len,o=-(enX*ux[k]+enY*uy[k]),d=enX*icx+enY*icy+o;if(d>0){enX=-enX;enY=-enY;o=-o;d=-d;}minDist=Math.max(minDist,d);edges[3*k]=enX;edges[3*k+1]=enY;edges[3*k+2]=o;}
  if(!valid||dist>1600||dist<0.01||minDist> -0.01){skipped++;continue;}
  const x0=clamp(Math.floor((Math.min(ux[0],ux[1],ux[2])-margin)/TILE),0,tx),x1=clamp(Math.ceil((Math.max(ux[0],ux[1],ux[2])+margin)/TILE),0,tx),y0=clamp(Math.floor((Math.min(uy[0],uy[1],uy[2])-margin)/TILE),0,ty),y1=clamp(Math.ceil((Math.max(uy[0],uy[1],uy[2])+margin)/TILE),0,ty);
  if(x0>=x1||y0>=y1){skipped++;continue;}
  const j=visible*32;packed[j]=ux[0];packed[j+1]=uy[0];packed[j+2]=ux[1];packed[j+3]=uy[1];packed[j+4]=ux[2];packed[j+5]=uy[2];packed[j+6]=sigma;packed[j+7]=1/minDist;
  for(let k=0;k<3;k++){const t=j+8+4*k;packed[t]=edges[3*k];packed[t+1]=edges[3*k+1];packed[t+2]=edges[3*k+2];packed[t+3]=k===0?weight:0;const v=3*idx[k],d=j+20+4*k;packed[d]=colors[v];packed[d+1]=colors[v+1];packed[d+2]=colors[v+2];}
  depths[visible]=vz;const bpos=visible*4;bounds[bpos]=x0;bounds[bpos+1]=x1;bounds[bpos+2]=y0;bounds[bpos+3]=y1;
  for(let yy=y0;yy<y1;yy++)for(let xx=x0;xx<x1;xx++)counts[yy*tx+xx]++;
  visible++;
 }
 const offsets=new Uint32Array(nt+1);for(let i=0;i<nt;i++)offsets[i+1]=offsets[i]+counts[i];
 const ids=new Uint32Array(offsets[nt]),cursor=offsets.slice(0,nt);
 for(let i=0;i<visible;i++){const p=i*4;for(let yy=bounds[p+2];yy<bounds[p+3];yy++)for(let xx=bounds[p];xx<bounds[p+1];xx++){const tile=yy*tx+xx;ids[cursor[tile]++]=i;}}
 for(let i=0;i<nt;i++)ids.subarray(offsets[i],offsets[i+1]).sort((a,b)=>depths[a]-depths[b]||a-b);
 return {triangles:packed.subarray(0,visible*32),offsets,ids,tileWidth:tx,skipped,visible};
}

export class TriangleSplatRenderer {
 static async create(canvas) {
  if(!navigator.gpu) throw Error('WebGPU unavailable; use HTTPS or localhost and a supported browser/GPU');
  const adapter=await navigator.gpu.requestAdapter(); if(!adapter) throw Error('No WebGPU adapter available');
  const device=await adapter.requestDevice(); const context=canvas.getContext('webgpu'); if(!context) throw Error('WebGPU canvas unavailable');
  const format=navigator.gpu.getPreferredCanvasFormat();context.configure({device,format,alphaMode:'opaque'});
  const cp=device.createComputePipeline({layout:'auto',compute:{module:device.createShaderModule({code:compute}),entryPoint:'main'}});
  const rp=device.createRenderPipeline({layout:'auto',vertex:{module:device.createShaderModule({code:display}),entryPoint:'vs'},fragment:{module:device.createShaderModule({code:display}),entryPoint:'fs',targets:[{format}]},primitive:{topology:'triangle-list'}});
  return new TriangleSplatRenderer(canvas,device,context,cp,rp);
 }
 constructor(canvas,device,context,cp,rp){Object.assign(this,{canvas,device,context,cp,rp});this.buffers=[];}
 async render(scene){
  const {device:d,canvas}=this,w=canvas.width,h=canvas.height,pre=preprocess(scene,w,h);
  for(const b of this.buffers)b.destroy();this.buffers=[];
  const mk=(data,usage)=>{const size=Math.max(16,Math.ceil(data.byteLength/4)*4);if(size>d.limits.maxStorageBufferBindingSize && (usage&GPUBufferUsage.STORAGE))throw Error(`Buffer ${size} exceeds device storage binding limit`);const b=d.createBuffer({size,usage:usage|GPUBufferUsage.COPY_DST});if(data.byteLength)d.queue.writeBuffer(b,0,data);this.buffers.push(b);return b;};
  const tb=mk(pre.triangles.length ? pre.triangles : new Float32Array(32),GPUBufferUsage.STORAGE),ob=mk(pre.offsets,GPUBufferUsage.STORAGE),ib=mk(pre.ids,GPUBufferUsage.STORAGE);
  const raw=new ArrayBuffer(32);new Uint32Array(raw,0,4).set([w,h,pre.tileWidth,0]);new Float32Array(raw,16,4).set([...(scene.background??[1,1,1]),0]);
  const ub=mk(new Uint8Array(raw),GPUBufferUsage.UNIFORM),outputBytes=w*h*16;
  if(outputBytes>d.limits.maxStorageBufferBindingSize)throw Error(`Output buffer ${outputBytes} exceeds device storage binding limit`);
  const pixels=d.createBuffer({size:outputBytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});this.buffers.push(pixels);this.pixelBuffer=pixels;
  const cb=d.createBindGroup({layout:this.cp.getBindGroupLayout(0),entries:[tb,ob,ib,ub,pixels].map((buffer,binding)=>({binding,resource:{buffer}}))});
  const rb=d.createBindGroup({layout:this.rp.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:pixels}},{binding:1,resource:{buffer:ub}}]});
  const enc=d.createCommandEncoder(),pass=enc.beginComputePass();pass.setPipeline(this.cp);pass.setBindGroup(0,cb);pass.dispatchWorkgroups(Math.ceil(w/8),Math.ceil(h/8));pass.end();
  const render=enc.beginRenderPass({colorAttachments:[{view:this.context.getCurrentTexture().createView(),loadOp:'clear',storeOp:'store',clearValue:{r:0,g:0,b:0,a:1}}]});render.setPipeline(this.rp);render.setBindGroup(0,rb);render.draw(3);render.end();d.queue.submit([enc.finish()]);
  return {visibleFaces:pre.visible,skippedFaces:pre.skipped,tileReferences:pre.ids.length,adaptive:'off',sampling:'single-center'};
 }
 async readPixels(){const d=this.device,size=this.canvas.width*this.canvas.height*16;if(!this.pixelBuffer)throw Error('Render first');const b=d.createBuffer({size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});const e=d.createCommandEncoder();e.copyBufferToBuffer(this.pixelBuffer,0,b,0,size);d.queue.submit([e.finish()]);await b.mapAsync(GPUMapMode.READ);const out=new Float32Array(b.getMappedRange().slice(0));b.unmap();b.destroy();return out;}
 destroy(){for(const b of this.buffers)b.destroy();this.buffers=[];}
}
