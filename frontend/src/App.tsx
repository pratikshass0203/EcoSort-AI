// src/App.tsx
import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Landing from './pages/Landing/Landing';
import Dashboard from './pages/Dashboard/Dashboard';
import Scanner from './pages/Scanner/Scanner';
import Result from './pages/Result/Result';
import Categories from './pages/Categories/Categories';
import DisposalGuide from './pages/DisposalGuide/DisposalGuide';
import History from './pages/History/History';
import Analytics from './pages/Analytics/Analytics';
import Assistant from './pages/Assistant/Assistant';
import Settings from './pages/Settings/Settings';
import Layout from './components/layout/Layout';

const App: React.FC = () => {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/scanner" element={<Scanner />} />
          <Route path="/result" element={<Result />} />
          <Route path="/categories" element={<Categories />} />
          <Route path="/disposal-guide" element={<DisposalGuide />} />
          <Route path="/history" element={<History />} />
          <Route path="/analytics" element={<Analytics />} />
          <Route path="/assistant" element={<Assistant />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Layout>
    </Router>
  );
};

export default App;

