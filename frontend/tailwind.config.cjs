/**
 * Tailwind configuration
 */
module.exports = {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        forest: {
          500: '#2c5f2d', // dark forest green
          600: '#244c23',
        },
        emerald: {
          500: '#50c878',
        },
      },
    },
  },
  plugins: [],
};

