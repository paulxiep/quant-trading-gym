/**
 * App component with routing
 *
 * SoC: Only handles routing, no business logic
 */

import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ErrorBoundary } from './components';
import { LandingPage, ConfigPage, SimulationPage } from './pages';

export function App() {
  return (
    <ErrorBoundary>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/config" element={<ConfigPage />} />
          <Route path="/sim" element={<SimulationPage />} />
        </Routes>
      </BrowserRouter>
    </ErrorBoundary>
  );
}
