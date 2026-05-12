/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        // Gold replaces crimson as the primary dragon accent
        crimson: {
          DEFAULT: '#C9A227',
          dark:    '#8B6914',
          light:   '#FFD700',
          glow:    'rgba(201,162,39,0.15)',
        },
        gold: {
          DEFAULT: '#C9A227',
          light:   '#FFD700',
          dark:    '#8B6914',
          glow:    'rgba(201,162,39,0.15)',
        },
        fire: {
          DEFAULT: '#FF6B00',
          light:   '#FF9500',
          dark:    '#CC4400',
          glow:    'rgba(255,107,0,0.2)',
        },
        dragon: {
          scale:   '#1A0E00',
          spine:   '#3D2800',
        },
        ink: {
          DEFAULT: '#050505',
          light:   '#0D0D0D',
          card:    'rgba(255,255,255,0.02)',
        },
      },
      fontFamily: {
        display: ['Impact', 'Arial Black', 'sans-serif'],
        body:    ['Inter', 'system-ui', 'sans-serif'],
        mono:    ['JetBrains Mono', 'Courier New', 'monospace'],
      },
      animation: {
        'pulse-gold':  'pulse-gold 2s ease-in-out infinite',
        'fire-flicker':'fire-flicker 0.15s ease-in-out infinite',
        'flicker':     'flicker 3s linear infinite',
        'slide-up':    'slide-up 0.6s ease-out forwards',
        'fade-in':     'fade-in 0.8s ease-out forwards',
        'spin-slow':   'spin 3s linear infinite',
        'float':       'float 4s ease-in-out infinite',
      },
      keyframes: {
        'pulse-gold': {
          '0%, 100%': { boxShadow: '0 0 0px rgba(201,162,39,0)' },
          '50%':      { boxShadow: '0 0 30px rgba(201,162,39,0.5)' },
        },
        'fire-flicker': {
          '0%, 100%': { opacity: '1', transform: 'scaleY(1)' },
          '50%':      { opacity: '0.85', transform: 'scaleY(0.97)' },
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
          '50%':      { transform: 'translateY(-8px)' },
        },
      },
    },
  },
  plugins: [],
}
