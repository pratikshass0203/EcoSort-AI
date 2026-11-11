// src/pages/Assistant/Assistant.tsx
import React, { useState } from 'react';
import { assistantChat } from '../../services/api';
import type { AssistantMessage, AssistantResponse } from '../../types';
import LoadingSpinner from '../../components/common/LoadingSpinner';
import { useLanguage } from '../../context/LanguageContext';

const Assistant: React.FC = () => {
  const { language, setLanguage } = useLanguage();
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<AssistantResponse[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const sendMessage = async () => {
    if (!input.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const payload: AssistantMessage = { message: input, language };
      const resp = await assistantChat(payload);
      setMessages((prev) => [...prev, resp]);
      setInput('');
    } catch (e:any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <h2 className="text-2xl font-bold text-emerald-500">AI Assistant</h2>
      <div className="flex items-center space-x-4">
        <label className="text-sm text-gray-200">Language:</label>
        <select
          value={language}
          onChange={(e) => setLanguage(e.target.value as any)}
          className="bg-forest-600 text-white p-1 rounded"
        >
          <option value="en">English</option>
          <option value="hi">Hindi</option>
          <option value="ta">Tamil</option>
        </select>
      </div>
      <div className="border border-forest-500 p-4 rounded h-64 overflow-y-auto bg-forest-600 glassmorphism">
        {messages.length === 0 ? (
          <p className="text-gray-300">Ask a question about waste disposal...</p>
        ) : (
          messages.map((msg, idx) => (
            <p key={idx} className="mb-2 text-white">{msg.response}</p>
          ))
        )}
      </div>
      {error && <div className="text-red-500">{error}</div>}
      <div className="flex space-x-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          className="flex-1 p-2 bg-forest-600 text-white rounded"
          placeholder="Type your question..."
        />
        <button
          onClick={sendMessage}
          disabled={loading}
          className="px-4 py-2 bg-emerald-500 text-black rounded hover:bg-emerald-600"
        >
          {loading ? 'Sending...' : 'Send'}
        </button>
      </div>
    </div>
  );
};

export default Assistant;

