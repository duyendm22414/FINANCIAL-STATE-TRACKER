import { Suspense } from 'react';
import { Navigate, Route, Routes } from 'react-router-dom';
import { ExperiencePage } from './pages/ExperiencePage';
import { ListingPage, ProductPage, CheckoutPage, CommissionPage, ContactPage } from './pages/ShopPages';
import { CartSlideover } from './components/CartSlideover';

export function App() {
  return (
    <>
      <Routes>
        <Route path="/" element={<ExperiencePage />} />
        <Route path="/shop" element={<ListingPage />} />
        <Route path="/product/:id" element={<ProductPage />} />
        <Route path="/checkout" element={<CheckoutPage />} />
        <Route path="/commission" element={<CommissionPage />} />
        <Route path="/contact" element={<ContactPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
      <Suspense fallback={null}><CartSlideover /></Suspense>
    </>
  );
}
