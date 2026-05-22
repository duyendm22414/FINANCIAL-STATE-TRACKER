import { Canvas } from '@react-three/fiber';
import { Float, OrbitControls } from '@react-three/drei';
import { useEffect, useMemo, useState } from 'react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import Lenis from 'lenis';
import { Link } from 'react-router-dom';
import { ProceduralLantern } from '../three/ProceduralLantern';

gsap.registerPlugin(ScrollTrigger);
const colors = ['#8f1b1b','#d6a645','#2f8c60','#233f8b','#d887aa','#ffb347'];

export function ExperiencePage(){
  const [idx,setIdx]=useState(0); const [shape,setShape]=useState<'round'|'octagonal'|'gourd'>('round'); const [size,setSize]=useState(1);
  const price = useMemo(()=>Math.round(180 + idx*18 + (shape==='octagonal'?40:shape==='gourd'?26:20) + size*40),[idx,shape,size]);
  useEffect(()=>{ const lenis = new Lenis(); const raf=(t:number)=>{lenis.raf(t); requestAnimationFrame(raf)}; requestAnimationFrame(raf);
    gsap.utils.toArray<HTMLElement>('.section').forEach((s)=>gsap.fromTo(s,{opacity:0.35},{opacity:1,scrollTrigger:{trigger:s,start:'top 70%',end:'top 20%',scrub:true}})); return ()=>lenis.destroy();},[]);
  return <main>{Array.from({length:8}).map((_,i)=><section className='section' key={i}><div className='overlay'><h2>{['Hội An','Craft Reveal','Color Bloom','Riverside','Customizer','Collection','Atelier','EndCard'][i]}</h2>{i===0&&<p>Mỗi chiếc đèn là một câu chuyện · Every lantern carries a story.</p>}{i===4&&<div className='panel'><button onClick={()=>setIdx((idx+1)%colors.length)}>Cycle Silk</button><button onClick={()=>setShape(shape==='round'?'octagonal':shape==='octagonal'?'gourd':'round')}>Shape: {shape}</button><input type='range' min='0.8' max='1.4' step='0.01' value={size} onChange={e=>setSize(Number(e.target.value))}/><div>${price}</div></div>}{i===7&&<div><Link to='/shop'>Shop Collection</Link> <Link to='/commission'>Custom Order</Link> <Link to='/contact'>Visit Hội An</Link></div>}</div><Canvas dpr={[1,2]} camera={{position:[0,0,3.4]}}><color attach='background' args={['#07050a']}/><ambientLight intensity={0.5}/><Float floatIntensity={0.6}><ProceduralLantern color={colors[idx]} shape={shape} scale={size} /></Float><OrbitControls enableZoom={false} enablePan={false}/></Canvas></section>)}</main>
}
