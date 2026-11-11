// src/components/layout/Layout.tsx
import React from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { LucideIcon } from 'lucide-react';
import { Home, LayoutDashboard, Camera, FileText, BarChart2, Bot, Settings, List, Trash2 } from 'lucide-react';

const navItems: { to: string; label: string; icon: LucideIcon }[] = [
  { to: '/', label: 'Home', icon: Home },
  { to: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/scanner', label: 'Scanner', icon: Camera },
  { to: '/history', label: 'History', icon: List },
  { to: '/analytics', label: 'Analytics', icon: BarChart2 },
  { to: '/assistant', label: 'Assistant', icon: Bot },
  { to: '/settings', label: 'Settings', icon: Settings },
];

const Layout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const location = useLocation();
  return (
    <div className="flex h-screen bg-gray-900 text-white">
      {/* Sidebar */}
      <aside className="w-64 bg-forest-600 p-4 flex flex-col">
        <h1 className="text-2xl font-bold mb-6 text-emerald-500">EcoSort AI</h1>
        <nav className="flex-1 space-y-2">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                `flex items-center p-2 rounded transition-colors ${
                  isActive ? 'bg-emerald-500 text-black' : 'hover:bg-forest-500'
                }`
              }
            >
              <item.icon className="mr-2 w-5 h-5" />
              {item.label}
            </NavLink>
          ))}
        </nav>
        <div className="mt-auto text-sm opacity-70">v0.1.0</div>
      </aside>
      {/* Main content */}
      <main className="flex-1 overflow-y-auto p-6">
        {children}
      </main>
    </div>
  );
};

export default Layout;

