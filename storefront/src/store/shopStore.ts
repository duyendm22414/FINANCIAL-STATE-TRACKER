import { create } from 'zustand';
import { Product } from '../data/catalog';

interface CartItem { product: Product; quantity: number }
interface ShopState {
  cartOpen:boolean; cart:CartItem[];
  toggleCart:()=>void; add:(p:Product)=>void; updateQty:(id:string, q:number)=>void;
}
export const useShopStore = create<ShopState>((set)=>({
  cartOpen:false, cart:[],
  toggleCart:()=>set((s)=>({cartOpen:!s.cartOpen})),
  add:(p)=>set((s)=>{ const e=s.cart.find(i=>i.product.id===p.id); if(e){return {cart:s.cart.map(i=>i.product.id===p.id?{...i,quantity:i.quantity+1}:i)};} return {cart:[...s.cart,{product:p,quantity:1}],cartOpen:true};}),
  updateQty:(id,q)=>set((s)=>({cart:s.cart.map(i=>i.product.id===id?{...i,quantity:Math.max(1,q)}:i)}))
}));
