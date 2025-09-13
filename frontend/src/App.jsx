import { useState } from "react";
import { analyzeCarImage } from "./api";
import Header from "./components/Header";
import UploadForm from "./components/UploadForm";
import TrustResult from "./components/TrustResult";
import ErrorMessage from "./components/ErrorMessage";

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError("Please upload an image before analyzing.");
      return;
    }
    setError("");
    setLoading(true);
    const analysis = await analyzeCarImage(file);
    if (analysis.error) {
      setError(analysis.error);
    } else {
      setResult(analysis);
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-indrive-gray flex flex-col items-center">
      <Header />
      <main className="flex flex-col items-center justify-center flex-1 p-6">
        <UploadForm onFileChange={setFile} onSubmit={handleSubmit} loading={loading} />
        <ErrorMessage message={error} />
        <TrustResult result={result} />
      </main>
    </div>
  );
}

export default App;
