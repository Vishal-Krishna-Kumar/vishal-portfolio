"use client";
import React, { useEffect, useRef, useState } from 'react';
import gsap from 'gsap';

// Helper to get greeting based on hour
function getTimeGreeting(hour: number) {
  if (hour < 5) return 'Good night';
  if (hour < 12) return 'Good morning';
  if (hour < 18) return 'Good afternoon';
  return 'Good evening';
}

// Helper to get location (city, country) using a free API
async function fetchLocation() {
  try {
    const res = await fetch('https://ipapi.co/json/');
    if (!res.ok) throw new Error('Failed to fetch location');
    const data = await res.json();
    return `${data.city ? data.city + ', ' : ''}${data.country_name || ''}`;
  } catch {
    return '';
  }
}

const AnimatedGreeting: React.FC = () => {
  const [name, setName] = useState('');
  const [inputName, setInputName] = useState('');
  const [location, setLocation] = useState('');
  const [greeting, setGreeting] = useState('');
  const greetingRef = useRef<HTMLDivElement>(null);
  const [showInput, setShowInput] = useState(true);

  useEffect(() => {
    if (name) {
      // Get time-based greeting
      const hour = new Date().getHours();
      setGreeting(getTimeGreeting(hour));
      // Get location
      fetchLocation().then(setLocation);
    }
  }, [name]);

  useEffect(() => {
    if (greeting && name && greetingRef.current) {
      gsap.fromTo(
        greetingRef.current.children,
        { opacity: 0, y: 40 },
        { opacity: 1, y: 0, stagger: 0.2, duration: 1, ease: 'power3.out' }
      );
    }
  }, [greeting, name]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputName.trim()) {
      setName(inputName.trim());
      setShowInput(false);
    }
  };

  return (
    <div className="w-full flex flex-col items-center mt-8">
      {showInput ? (
        <form onSubmit={handleSubmit} className="flex flex-col items-center gap-2 bg-white/80 p-4 rounded-xl shadow-lg">
          <label htmlFor="visitor-name" className="text-lg font-semibold">What's your name?</label>
          <input
            id="visitor-name"
            type="text"
            value={inputName}
            onChange={e => setInputName(e.target.value)}
            className="px-3 py-2 rounded border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-400"
            placeholder="Enter your name"
            required
            autoFocus
          />
          <button type="submit" className="mt-2 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition">Greet Me!</button>
        </form>
      ) : (
        <div ref={greetingRef} className="text-center">
          <h1 className="text-3xl md:text-4xl font-bold mb-2">
            {greeting}, {name}{location ? ` from ${location}` : ''}!
          </h1>
          <p className="text-lg text-gray-600">Welcome to my portfolio 👋</p>
        </div>
      )}
    </div>
  );
};

export default AnimatedGreeting;
