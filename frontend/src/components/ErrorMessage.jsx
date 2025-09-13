/* eslint-disable react/prop-types */
export default function ErrorMessage({ message }) {
  if (!message) return null;
  return (
    <div className="mt-4 p-4 bg-red-100 text-red-700 border border-red-400 rounded-md">
      <p>{message}</p>
    </div>
  );
}
