"use client";
import React, { useState, useRef, useEffect } from 'react';

const BOT_NAME = "VishalBot";

const initialMessages = [
  {
    role: "bot",
    content: `Welcome! I'm Vishal Krishna Kumar's AI. Ask me about my experience, projects, or anything technical.`
  }
];

export default function VishalChatbot() {
  const [messages, setMessages] = useState(initialMessages);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function sendMessage(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (!input.trim()) return;
    const userMsg = { role: "user", content: input };
    setMessages(msgs => [...msgs, userMsg]);
    setInput("");
    setLoading(true);
    try {
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: [...messages, userMsg] })
      });
      const data = await res.json();
      setMessages(msgs => [...msgs, { role: "bot", content: data.answer }]);
    } catch {
      setMessages(msgs => [...msgs, { role: "bot", content: "Sorry, something went wrong." }]);
    }
    setLoading(false);
  }

  return (
    <div className="fixed bottom-6 right-6 z-50 w-96 max-w-full bg-white/90 rounded-xl shadow-2xl border border-gray-200 flex flex-col">
      <div className="px-4 py-2 bg-blue-700 text-white rounded-t-xl font-bold text-lg">{BOT_NAME}</div>
      <div className="flex-1 overflow-y-auto px-4 py-2 space-y-2 h-80">
        {messages.map((msg, i) => (
          <div key={i} className={msg.role === "user" ? "text-right" : "text-left"}>
            <span className={msg.role === "user" ? "inline-block bg-blue-100 text-blue-900 px-3 py-2 rounded-lg" : "inline-block bg-gray-200 text-gray-800 px-3 py-2 rounded-lg"}>
              {msg.content}
            </span>
          </div>
        ))}
        <div ref={messagesEndRef as React.RefObject<HTMLDivElement>} />
      </div>
      <form onSubmit={sendMessage} className="flex border-t border-gray-200">
        <input
          className="flex-1 px-3 py-2 rounded-bl-xl focus:outline-none"
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="Ask Vishal anything..."
          disabled={loading}
        />
        <button type="submit" className="px-4 py-2 bg-blue-700 text-white rounded-br-xl disabled:opacity-50" disabled={loading || !input.trim()}>
          {loading ? "..." : "Send"}
        </button>
      </form>
    </div>
  );
}
