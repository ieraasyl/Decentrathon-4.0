/* eslint-disable react/prop-types */
export default function TrustResult({ result }) {
  if (!result) return null;

  if (result.error) {
    return <p className="text-red-500">Error: {result.error}</p>;
  }

  return (
    <div className="p-4 border rounded-md shadow-md bg-white mt-4">
      <h2 className="text-xl font-bold mb-2">Analysis Result</h2>
      <p><strong>Predicted Class:</strong> {result.predicted_class}</p>
      <p><strong>Trust Score:</strong> {result.trust_score}%</p>
      <p><strong>Explanation:</strong> {result.explanation}</p>
    </div>
  );
}
