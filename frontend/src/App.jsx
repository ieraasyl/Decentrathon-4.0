import React, { useState, useRef, useCallback } from 'react';
import { Camera, Upload, CheckCircle, AlertTriangle, X, Star, Shield, Clock, Users, ArrowLeft } from 'lucide-react';

const App = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysis, setAnalysis] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);
  const cameraInputRef = useRef(null);

  // Replace with your actual Render deployment URL
  const API_BASE_URL = import.meta.env.VITE_API_BASE;

  const handleImageSelect = useCallback((file) => {
    if (!file) return;
    
    if (!file.type.match(/^image\/(jpeg|jpg|png)$/)) {
      setError('Please select a JPG or PNG image file.');
      return;
    }
    
    if (file.size > 5 * 1024 * 1024) {
      setError('File size must be less than 5MB.');
      return;
    }

    setError(null);
    setSelectedImage(file);
    
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
  }, []);

  const handleFileInput = (e) => {
    const file = e.target.files?.[0];
    if (file) handleImageSelect(file);
  };

  const analyzeImage = async () => {
    if (!selectedImage) return;

    setIsAnalyzing(true);
    setError(null);
    
    try {
      const formData = new FormData();
      formData.append('file', selectedImage);

      const response = await fetch(`${API_BASE_URL}/trust-score`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      
      if (data.error) {
        throw new Error(data.error);
      }

      setAnalysis(data);
    } catch (err) {
      setError(err.message || 'Failed to analyze image. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const resetForm = () => {
    setSelectedImage(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    setAnalysis(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
    if (cameraInputRef.current) cameraInputRef.current.value = '';
  };

  const getTrustScoreColor = (score) => {
    if (score >= 80) return 'text-white bg-lime-400';
    if (score >= 60) return 'text-gray-900 bg-yellow-400';
    return 'text-white bg-red-500';
  };

  const getConditionDisplay = (condition) => {
    const displays = {
      'clean': { text: 'Clean Vehicle', color: 'text-lime-400', bg: 'bg-lime-400' },
      'dirty': { text: 'Needs Cleaning', color: 'text-yellow-400', bg: 'bg-yellow-400' },
      'scratchless': { text: 'Excellent Condition', color: 'text-lime-400', bg: 'bg-lime-400' },
      'scratched': { text: 'Minor Damage', color: 'text-red-400', bg: 'bg-red-400' }
    };
    return displays[condition] || displays['clean'];
  };

  return (
    <div className="min-h-screen bg-gray-900">
      {/* Header */}
      <div className="bg-gray-800 px-4 py-4 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <button className="p-2 hover:bg-gray-700 rounded-lg">
            <ArrowLeft className="w-5 h-5 text-white" />
          </button>
          <div>
            <h1 className="text-white font-semibold text-lg">Vehicle Inspector</h1>
            <p className="text-gray-400 text-sm">Check your car condition</p>
          </div>
        </div>
      </div>

      <div className="p-4 space-y-4 max-w-md mx-auto">
        {/* Main Card */}
        <div className="bg-gray-800 rounded-2xl overflow-hidden">
          {!previewUrl ? (
            <div className="p-6">
              <div className="text-center mb-6">
                <div className="w-16 h-16 bg-gray-700 rounded-2xl flex items-center justify-center mx-auto mb-4">
                  <Camera className="w-8 h-8 text-lime-400" />
                </div>
                <h2 className="text-white text-xl font-semibold mb-2">Upload Vehicle Photo</h2>
                <p className="text-gray-400 text-sm">Take a clear photo to check your vehicle condition</p>
              </div>

              <div className="space-y-3">
                <button
                  onClick={() => cameraInputRef.current?.click()}
                  className="w-full bg-lime-400 text-gray-900 font-semibold py-4 rounded-2xl flex items-center justify-center space-x-2 hover:bg-lime-500 transition-colors"
                >
                  <Camera className="w-5 h-5" />
                  <span>Take Photo</span>
                </button>
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="w-full bg-gray-700 text-white font-semibold py-4 rounded-2xl flex items-center justify-center space-x-2 hover:bg-gray-600 transition-colors"
                >
                  <Upload className="w-5 h-5" />
                  <span>Choose from Gallery</span>
                </button>
              </div>
              <p className="text-gray-500 text-xs text-center mt-4">Supports JPG, PNG up to 5MB</p>
            </div>
          ) : (
            <div>
              {/* Image Preview */}
              <div className="relative">
                <div className="aspect-video">
                  <img
                    src={previewUrl}
                    alt="Vehicle preview"
                    className="w-full h-full object-cover"
                  />
                </div>
                <button
                  onClick={resetForm}
                  className="absolute top-3 right-3 p-2 bg-black bg-opacity-60 text-white rounded-full hover:bg-opacity-80"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>

              <div className="p-6">
                {!analysis && !isAnalyzing && (
                  <button
                    onClick={analyzeImage}
                    className="w-full bg-lime-400 text-gray-900 font-semibold py-4 rounded-2xl hover:bg-lime-500 transition-colors"
                  >
                    Check Vehicle Condition
                  </button>
                )}

                {isAnalyzing && (
                  <div className="text-center py-8">
                    <div className="w-8 h-8 border-2 border-lime-400 border-t-transparent rounded-full animate-spin mx-auto mb-3"></div>
                    <p className="text-gray-400">Analyzing your vehicle...</p>
                  </div>
                )}

                {analysis && (
                  <div className="space-y-4">
                    {/* Trust Score */}
                    <div className="text-center">
                      <div className={`inline-flex items-center px-6 py-3 rounded-2xl font-bold text-lg ${getTrustScoreColor(analysis.trust_score)}`}>
                        Trust Score: {analysis.trust_score}%
                      </div>
                    </div>

                    {/* Condition Result */}
                    <div className="bg-gray-700 rounded-xl p-4">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-gray-300 text-sm">Condition</span>
                        <div className={`w-3 h-3 rounded-full ${getConditionDisplay(analysis.predicted_class).bg}`}></div>
                      </div>
                      <p className={`font-semibold ${getConditionDisplay(analysis.predicted_class).color}`}>
                        {getConditionDisplay(analysis.predicted_class).text}
                      </p>
                    </div>

                    {/* Action Buttons */}
                    <div className="space-y-3">
                      {analysis.trust_score >= 80 && (
                        <div className="bg-lime-400 bg-opacity-10 border border-lime-400 border-opacity-30 rounded-xl p-4">
                          <div className="flex items-start space-x-3">
                            <CheckCircle className="w-5 h-5 text-lime-400 mt-0.5" />
                            <div>
                              <p className="text-lime-400 font-semibold text-sm">Perfect!</p>
                              <p className="text-gray-300 text-xs">Your vehicle meets quality standards</p>
                            </div>
                          </div>
                        </div>
                      )}
                      
                      <button
                        onClick={resetForm}
                        className="w-full bg-gray-700 text-white font-semibold py-3 rounded-xl hover:bg-gray-600 transition-colors"
                      >
                        Check Another Photo
                      </button>
                      <button
                        className="w-full bg-lime-400 text-gray-900 font-semibold py-3 rounded-xl hover:bg-lime-500 transition-colors"
                        onClick={() => window.open('https://indrive.com', '_blank')}
                      >
                        Continue to inDrive
                      </button>
                    </div>
                  </div>
                )}

                {error && (
                  <div className="bg-red-500 bg-opacity-10 border border-red-500 border-opacity-30 rounded-xl p-4 mt-4">
                    <div className="flex items-start space-x-3">
                      <AlertTriangle className="w-5 h-5 text-red-400 mt-0.5" />
                      <div>
                        <p className="text-red-400 font-semibold text-sm">Error</p>
                        <p className="text-gray-300 text-xs">{error}</p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-2 gap-3">
          <div className="bg-gray-800 rounded-xl p-4">
            <div className="flex items-center space-x-2 mb-2">
              <div className="w-8 h-8 bg-gray-700 rounded-lg flex items-center justify-center">
                <CheckCircle className="w-4 h-4 text-lime-400" />
              </div>
              <span className="text-lime-400 font-bold text-lg">98.5%</span>
            </div>
            <p className="text-gray-400 text-xs">Accuracy</p>
          </div>
          
          <div className="bg-gray-800 rounded-xl p-4">
            <div className="flex items-center space-x-2 mb-2">
              <div className="w-8 h-8 bg-gray-700 rounded-lg flex items-center justify-center">
                <Clock className="w-4 h-4 text-lime-400" />
              </div>
              <span className="text-lime-400 font-bold text-lg">&lt;3s</span>
            </div>
            <p className="text-gray-400 text-xs">Analysis time</p>
          </div>
        </div>

        {/* Info Card */}
        <div className="bg-gray-800 rounded-xl p-4">
          <div className="flex items-start space-x-3">
            <div className="w-8 h-8 bg-gray-700 rounded-lg flex items-center justify-center mt-1">
              <Shield className="w-4 h-4 text-lime-400" />
            </div>
            <div>
              <h3 className="text-white font-semibold text-sm mb-1">Why check your vehicle?</h3>
              <p className="text-gray-400 text-xs leading-relaxed">
                Clean vehicles get 23% more ride requests and higher passenger ratings. 
                Keep your car in great condition to maximize your earnings.
              </p>
            </div>
          </div>
        </div>

        {/* Bottom Notice */}
        <div className="text-center pt-4">
          <p className="text-gray-500 text-xs">
            Trusted by <span className="text-lime-400 font-semibold">150M+ users</span> worldwide
          </p>
        </div>
      </div>

      {/* Hidden File Inputs */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/jpeg,image/jpg,image/png"
        onChange={handleFileInput}
        className="hidden"
      />
      <input
        ref={cameraInputRef}
        type="file"
        accept="image/*"
        capture="environment"
        onChange={handleFileInput}
        className="hidden"
      />
    </div>
  );
};

export default App;