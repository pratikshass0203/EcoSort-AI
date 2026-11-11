import React from "react";

const Settings: React.FC = () => {
  return (
    <div className="min-h-screen p-8">
      <h1 className="text-3xl font-bold">Settings</h1>
      <p className="mt-3">
        Manage your EcoSort AI application settings.
      </p>

      <div className="mt-6 rounded-xl border p-6">
        <h2 className="text-xl font-semibold">Backend Connection</h2>
        <p>Backend API: http://127.0.0.1:8000/api/v1</p>
      </div>

      <div className="mt-6 rounded-xl border p-6">
        <h2 className="text-xl font-semibold">Language</h2>
        <p>English | Hindi | Tamil</p>
      </div>
    </div>
  );
};

export default Settings;
