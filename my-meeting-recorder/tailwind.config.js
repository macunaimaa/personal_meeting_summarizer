/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}", // This line ensures Tailwind scans your React components
  ],
  theme: {
    extend: {
      // You can add custom theme extensions here if needed
    },
  },
  plugins: [
    // You can add Tailwind plugins here if needed
  ],
}
