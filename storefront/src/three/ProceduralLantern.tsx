import { useMemo, useRef } from 'react';
import * as THREE from 'three';
import { useFrame } from '@react-three/fiber';

export function ProceduralLantern({ color='#e0a43a', shape='round', scale=1 }: { color?: string; shape?: 'round'|'octagonal'|'gourd'; scale?: number; }) {
  const group = useRef<THREE.Group>(null);
  const geometry = useMemo(() => {
    const points: THREE.Vector2[] = [];
    for (let i = 0; i <= 32; i++) {
      const t = i / 32;
      const y = (t - 0.5) * 2;
      const r = shape === 'gourd' ? 0.45 + 0.18 * Math.sin(t * Math.PI * 2) : 0.5 * Math.sin(t * Math.PI);
      points.push(new THREE.Vector2(Math.max(0.1, r), y));
    }
    return new THREE.LatheGeometry(points, shape === 'octagonal' ? 8 : 32);
  }, [shape]);
  useFrame((state) => { if(group.current){ group.current.rotation.y += 0.002; group.current.position.x = Math.sin(state.clock.elapsedTime*0.3)*0.08; }});
  return <group ref={group} scale={scale}><mesh geometry={geometry}><meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.25} roughness={0.25} metalness={0.4} /></mesh><pointLight color={color} intensity={6} distance={4} /></group>;
}
