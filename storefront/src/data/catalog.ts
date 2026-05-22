export type Shape = 'round' | 'octagonal' | 'gourd';
export type Color = 'crimson' | 'imperialGold' | 'jade' | 'indigo' | 'lotus' | 'amber';
export interface Product { id:string; vnName:string; enName:string; shape:Shape; color:Color; size:'S'|'M'|'L'; silk:string; dimensions:string; price:number; description:string; }

export const products: Product[] = [
  { id:'lua-song-hoai', vnName:'Lụa Sông Hoài', enName:'Hoài River Silk', shape:'round', color:'amber', size:'M', silk:'Hand-dyed mulberry silk', dimensions:'45cm x 70cm', price:220, description:'Warm amber glow inspired by lantern trails along the Thu Bồn river.' },
  { id:'bach-kim-vang', vnName:'Bách Kim Vàng', enName:'Imperial Gold Hundred', shape:'octagonal', color:'imperialGold', size:'L', silk:'Golden lacquered silk', dimensions:'60cm x 90cm', price:340, description:'Architectural octagonal frame with regal golden silk and deep radiance.' },
  { id:'sen-dem', vnName:'Sen Đêm', enName:'Night Lotus', shape:'gourd', color:'lotus', size:'S', silk:'Lotus-pink satin silk', dimensions:'38cm x 64cm', price:195, description:'Soft pink transitions that mimic lotus petals in moonlit water.' },
  { id:'ngoc-pho-co', vnName:'Ngọc Phố Cổ', enName:'Old Town Jade', shape:'round', color:'jade', size:'M', silk:'Jade woven silk', dimensions:'48cm x 72cm', price:245, description:'Luminous jade silk with quiet luxury and traditional bamboo ribs.' },
  { id:'cham-indigo', vnName:'Chàm Hội', enName:'Hội Indigo', shape:'octagonal', color:'indigo', size:'M', silk:'Natural indigo silk', dimensions:'50cm x 80cm', price:260, description:'Deep indigo artisan dye with shadow-rich tonal variation.' },
  { id:'do-hoang-hon', vnName:'Đỏ Hoàng Hôn', enName:'Twilight Crimson', shape:'gourd', color:'crimson', size:'L', silk:'Crimson brushed silk', dimensions:'58cm x 96cm', price:315, description:'Signature crimson statement lantern with cinematic evening warmth.' }
];
