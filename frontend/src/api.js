import axios from "axios";

// replace with your Render backend URL later
const API_BASE = "https://decentrathon-4-0.onrender.com";

export const analyzeCarImage = async (file) => {
  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await axios.post(`${API_BASE}/trust-score`, formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    return response.data;
  } catch (err) {
    console.error(err);
    return { error: "API call failed" };
  }
};