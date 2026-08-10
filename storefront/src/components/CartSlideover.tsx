import { Link } from 'react-router-dom';
import { useShopStore } from '../store/shopStore';

export function CartSlideover(){
  const {cartOpen,toggleCart,cart,updateQty}=useShopStore();
  const total = cart.reduce((s,i)=>s+i.product.price*i.quantity,0);
  return <aside className={`cart ${cartOpen?'open':''}`}><button onClick={toggleCart}>Cart ({cart.length})</button>{cart.map(i=><div key={i.product.id}><span>{i.product.enName}</span><input type='number' value={i.quantity} onChange={e=>updateQty(i.product.id,Number(e.target.value))}/></div>)}<strong>${total}</strong><Link to='/checkout'>Checkout</Link></aside>
}
