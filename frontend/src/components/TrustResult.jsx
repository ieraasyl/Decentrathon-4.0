/* eslint-disable react/prop-types */
export default function TrustResult({ result }) {
  if (!result) return null;

  const trustScore = parseFloat(result.trust_score || 0);
  let color = "bg-red-500";
  if (trustScore >= 70) color = "bg-green-500";
  else if (trustScore >= 40) color = "bg-yellow-500";

  return (
    <div className="mt-6 p-6 bg-white rounded-xl shadow-lg w-full max-w-md">
      <h2 className="text-xl font-bold mb-4 text-indrive-dark">Analysis Result</h2>

      <div className="mb-3">
        <p><strong>Predicted Class:</strong> {result.predicted_class}</p>
        <p><strong>Explanation:</strong> {result.explanation}</p>
      </div>

      <div className="mt-4">
        <p className="mb-1 font-semibold">Trust Score</p>
        <div className="w-full bg-gray-200 rounded-full h-4">
          <div
            className={`h-4 rounded-full ${color}`}
            style={{ width: `${trustScore}%` }}
          ></div>
        </div>
        <p className="text-sm mt-1">{trustScore}%</p>
      </div>
    </div>
  );
}
