import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "var(--background)",
        foreground: "var(--foreground)",
        // Sky Blue palette
        sky: {
          50:  "#f0f9ff",
          100: "#e0f2fe",
          200: "#bae6fd",
          300: "#7dd3fc",
          400: "#38bdf8",
          500: "#0ea5e9",
          600: "#0284c7",
          700: "#0369a1",
          800: "#075985",
          900: "#0c4a6e",
        },
        // Amber / Red accent palette
        brand: {
          amber:  "#f59e0b",
          orange: "#f97316",
          red:    "#ef4444",
          crimson:"#dc2626",
        },
        // Dark navy background shades
        navy: {
          950: "#07111f",
          900: "#0c1a2e",
          800: "#102236",
          700: "#163045",
        },
      },
    },
  },
  plugins: [],
};
export default config;
