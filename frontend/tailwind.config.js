/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,ts,jsx,tsx,mdx}",
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          bg: "#FFF9FC",
          white: "#FFFFFF",
          pastel: "#F8C8DC",
          light: "#FCE7F3",
          primary: "#E88FB3",
          active: "#C45F88",
          text: "#3F2933",
          secondary: "#8A6875",
          border: "#F3D5E2",
        },
      },
    },
  },
  plugins: [],
};
