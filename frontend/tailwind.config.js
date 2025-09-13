/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        indrive: {
          green: "#00C853", // primary brand color
          gray: "#F5F5F5",
          dark: "#212121",
        },
      },
    },
  },
  plugins: [],
};
