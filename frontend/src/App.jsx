import { useState } from "react";
import { analyzeCarImage } from "./api";
import TrustResult from "./components/TrustResult";

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;

    setLoading(true);
    const analysis = await analyzeCarImage(file);
    setResult(analysis);
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center justify-center p-6">
      <h1 className="text-3xl font-bold mb-6">
        🚘 inDrive AI Car Image Analysis
      </h1>

      <form onSubmit={handleSubmit} className="flex flex-col items-center">
        <input
          type="file"
          accept="image/*"
          onChange={(e) => setFile(e.target.files[0])}
          className="mb-4"
        />
        <button
          type="submit"
          disabled={!file || loading}
          className="bg-blue-600 text-white px-6 py-2 rounded-md shadow-md hover:bg-blue-700 disabled:opacity-50"
        >
          {loading ? "Analyzing..." : "Analyze"}
        </button>
      </form>

      <TrustResult result={result} />
    </div>
  );
}

export default App;
