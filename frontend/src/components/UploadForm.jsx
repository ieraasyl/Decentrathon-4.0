/* eslint-disable react/prop-types */
export default function UploadForm({ onFileChange, onSubmit, loading }) {
  return (
    <form
      onSubmit={onSubmit}
      className="flex flex-col items-center p-6 bg-white shadow-lg rounded-xl w-full max-w-md"
    >
      <input
        type="file"
        accept="image/*"
        onChange={(e) => onFileChange(e.target.files[0])}
        className="mb-4"
      />
      <button
        type="submit"
        disabled={loading}
        className="w-full bg-indrive-green text-white py-2 rounded-md font-semibold hover:bg-green-600 disabled:opacity-50"
      >
        {loading ? "Analyzing..." : "Analyze Image"}
      </button>
    </form>
  );
}
