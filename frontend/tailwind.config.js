/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        crimson: {
          DEFAULT: '#8b5cf6',
          dark:    '#7c3aed',
          light:   '#a78bfa',
          glow:    'rgba(139,92,246,0.15)',
        },
        violet: {
          DEFAULT: '#8b5cf6',
          light:   '#a78bfa',
          dark:    '#7c3aed',
          glow:    'rgba(139,92,246,0.15)',
        },
        fuchsia: {
          DEFAULT: '#c026d3',
          light:   '#d946ef',
          dark:    '#a21caf',
          glow:    'rgba(192,38,211,0.2)',
        },
        ink: {
          DEFAULT: '#000000',
          light:   '#0a0a0f',
          card:    'rgba(139,92,246,0.03)',
        },
      },
      fontFamily: {
        display: ['Bebas Neue', 'Impact', 'sans-serif'],
        body:    ['Outfit', 'system-ui', 'sans-serif'],
        mono:    ['Space Mono', 'Courier New', 'monospace'],
      },
      animation: {
        'pulse-violet': 'pulse-violet 2s ease-in-out infinite',
        'flicker':      'flicker 3s linear infinite',
        'slide-up':     'slide-up 0.6s ease-out forwards',
        'fade-in':      'fade-in 0.8s ease-out forwards',
        'float':        'float 4s ease-in-out infinite',
        'spin-slow':    'spin 8s linear infinite',
      },
      keyframes: {
        'pulse-violet': {
          '0%, 100%': { boxShadow: '0 0 0px rgba(139,92,246,0)' },
          '50%':      { boxShadow: '0 0 40px rgba(139,92,246,0.5)' },
        },
        'flicker': {
          '0%, 100%': { opacity: '1' },
          '92%':      { opacity: '1' },
          '93%':      { opacity: '0.8' },
          '94%':      { opacity: '1' },
          '96%':      { opacity: '0.9' },
          '97%':      { opacity: '1' },
        },
        'slide-up': {
          from: { opacity: '0', transform: 'translateY(40px)' },
          to:   { opacity: '1', transform: 'translateY(0)' },
        },
        'fade-in': {
          from: { opacity: '0' },
          to:   { opacity: '1' },
        },
        'float': {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%':      { transform: 'translateY(-10px)' },
        },
      },
    },
  },
  plugins: [],
}
