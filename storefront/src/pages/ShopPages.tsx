import { Link, useParams } from 'react-router-dom';
import { products } from '../data/catalog';
import { Canvas } from '@react-three/fiber';
import { ProceduralLantern } from '../three/ProceduralLantern';
import { useShopStore } from '../store/shopStore';

export function ListingPage(){ const add=useShopStore(s=>s.add); return <div className='page'><h1>Lantern Collection</h1>{products.map(p=><article key={p.id}><Canvas style={{height:180}}><ambientLight intensity={0.4}/><ProceduralLantern shape={p.shape} /></Canvas><h3>{p.vnName} · {p.enName}</h3><p>{p.description}</p><p>${p.price}</p><button onClick={()=>add(p)}>Add to Cart</button><Link to={`/product/${p.id}`}>Detail</Link></article>)}</div>; }
export function ProductPage(){ const {id}=useParams(); const p=products.find(x=>x.id===id); const add=useShopStore(s=>s.add); if(!p) return <div>Not found</div>; return <div className='page'><h1>{p.vnName}</h1><Canvas style={{height:320}}><ambientLight intensity={0.5}/><ProceduralLantern shape={p.shape} /></Canvas><p>{p.description}</p><p>{p.dimensions} · {p.silk}</p><button onClick={()=>add(p)}>Add ${p.price}</button></div>; }
export function CheckoutPage(){ return <form className='page'><h1>Checkout</h1><input placeholder='Full name'/><input placeholder='Email'/><input placeholder='Address'/><textarea placeholder='Delivery notes'/><button>Submit Order</button></form>; }
export function CommissionPage(){ return <form className='page'><h1>Custom Commission</h1><select><option>Round</option><option>Octagonal</option><option>Gourd</option></select><select><option>Crimson Ember</option><option>Imperial Gold</option><option>Jade Mist</option></select><textarea placeholder='Describe your dream lantern'/><button>Send request</button></form>; }
export function ContactPage(){ return <div className='page'><h1>Visit Hội An Atelier</h1><p>Moon Courtyard, Ancient Town District, Hội An.</p><input placeholder='Newsletter email'/><button>Subscribe</button></div>; }
