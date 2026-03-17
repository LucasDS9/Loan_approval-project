// API base URL – update to your Render URL when deployed
window.API_BASE = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
  ? "http://127.0.0.1:8000"
  : "https://your-app.onrender.com"; // ← substitua pela URL do Render quando fizer deploy