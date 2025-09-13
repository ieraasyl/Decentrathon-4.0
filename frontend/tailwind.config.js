/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        indrive: {
          green: "#00C853",
          gray: "#F5F5F5",
          dark: "#212121",
        },
      },
    },
  },
  plugins: [],
};
