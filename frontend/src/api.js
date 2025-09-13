import axios from "axios";

const API_BASE = import.meta.env.VITE_API_BASE;

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
