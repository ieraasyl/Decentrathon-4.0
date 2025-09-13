import React, { useState, useRef, useCallback } from 'react';
import { Camera, Upload, CheckCircle, AlertTriangle, X, Star, Shield, Clock, Users, ArrowLeft, Plus } from 'lucide-react';

const App = () => {
  const [photos, setPhotos] = useState({
    front: null,
    back: null,
    left: null,
    right: null
  });
  const [previewUrls, setPreviewUrls] = useState({
    front: null,
    back: null,
    left: null,
    right: null
  });
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysis, setAnalysis] = useState(null);
  const [error, setError] = useState(null);
  const [currentSide, setCurrentSide] = useState(null);
  const fileInputRef = useRef(null);
  const cameraInputRef = useRef(null);

  // Replace with your actual Render deployment URL
  const API_BASE_URL = import.meta.env.VITE_API_BASE;

  const carSides = [
    { key: 'front', label: 'Front Side', icon: '🚗', description: 'Front bumper and headlights' },
    { key: 'back', label: 'Back Side', icon: '🚙', description: 'Rear bumper and taillights' },
    { key: 'left', label: 'Left Side', icon: '🚘', description: 'Driver side doors and windows' },
    { key: 'right', label: 'Right Side', icon: '🚖', description: 'Passenger side doors and windows' }
  ];

  const handleImageSelect = useCallback((file, side) => {
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
    
    // Clean up previous preview URL for this side
    if (previewUrls[side]) {
      URL.revokeObjectURL(previewUrls[side]);
    }
    
    const url = URL.createObjectURL(file);
    
    setPhotos(prev => ({ ...prev, [side]: file }));
    setPreviewUrls(prev => ({ ...prev, [side]: url }));
    setCurrentSide(null);
  }, [previewUrls]);

  const handleFileInput = (e) => {
    const file = e.target.files?.[0];
    if (file && currentSide) {
      handleImageSelect(file, currentSide);
    }
  };

  const openCamera = (side) => {
    setCurrentSide(side);
    cameraInputRef.current?.click();
  };

  const openGallery = (side) => {
    setCurrentSide(side);
    fileInputRef.current?.click();
  };

  const removePhoto = (side) => {
    if (previewUrls[side]) {
      URL.revokeObjectURL(previewUrls[side]);
    }
    setPhotos(prev => ({ ...prev, [side]: null }));
    setPreviewUrls(prev => ({ ...prev, [side]: null }));
  };

  const analyzeAllPhotos = async () => {
    const uploadedPhotos = Object.values(photos).filter(Boolean);
    if (uploadedPhotos.length === 0) {
      setError('Please upload at least one photo.');
      return;
    }

    setIsAnalyzing(true);
    setError(null);
    
    try {
      const formData = new FormData();
      
      // Add all photos to the form data as a list
      carSides.forEach(({ key }) => {
        if (photos[key]) {
          formData.append('files', photos[key], `${key}_side.jpg`);
        }
      });

      const response = await fetch(`${API_BASE_URL}/inspect-vehicle`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      
      if (data.error) {
        throw new Error(data.error);
      }

      setAnalysis(data);
    } catch (err) {
      setError(err.message || 'Failed to analyze images. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const resetAll = () => {
    // Clean up all preview URLs
    Object.values(previewUrls).forEach(url => {
      if (url) URL.revokeObjectURL(url);
    });
    
    setPhotos({ front: null, back: null, left: null, right: null });
    setPreviewUrls({ front: null, back: null, left: null, right: null });
    setAnalysis(null);
    setError(null);
    setCurrentSide(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
    if (cameraInputRef.current) cameraInputRef.current.value = '';
  };

  const getOverallTrustScore = () => {
    if (!analysis || !analysis.results) return 0;
    const scores = analysis.results.map(result => result.trust_score);
    return Math.round(scores.reduce((sum, score) => sum + score, 0) / scores.length);
  };

  const getTrustScoreColor = (score) => {
    if (score >= 80) return 'text-white bg-lime-400';
    if (score >= 60) return 'text-gray-900 bg-yellow-400';
    return 'text-white bg-red-500';
  };

  const getConditionDisplay = (condition) => {
    const displays = {
      'clean': { text: 'Clean', color: 'text-lime-400', bg: 'bg-lime-400' },
      'dirty': { text: 'Needs Cleaning', color: 'text-yellow-400', bg: 'bg-yellow-400' },
      'scratchless': { text: 'Excellent', color: 'text-lime-400', bg: 'bg-lime-400' },
      'scratched': { text: 'Minor Damage', color: 'text-red-400', bg: 'bg-red-400' }
    };
    return displays[condition] || displays['clean'];
  };

  const uploadedCount = Object.values(photos).filter(Boolean).length;

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
            <p className="text-gray-400 text-sm">Upload 4 photos of your car</p>
          </div>
        </div>
        <div className="text-lime-400 font-semibold text-sm">
          {uploadedCount}/4
        </div>
      </div>

      <div className="p-4 space-y-4 max-w-md mx-auto">
        {/* Photo Upload Grid */}
        <div className="grid grid-cols-2 gap-3">
          {carSides.map(({ key, label, icon, description }) => (
            <div key={key} className="bg-gray-800 rounded-xl overflow-hidden">
              {previewUrls[key] ? (
                <div className="relative">
                  <div className="aspect-square">
                    <img
                      src={previewUrls[key]}
                      alt={`${label} view`}
                      className="w-full h-full object-cover"
                    />
                  </div>
                  <button
                    onClick={() => removePhoto(key)}
                    className="absolute top-2 right-2 p-1.5 bg-black bg-opacity-60 text-white rounded-full hover:bg-opacity-80"
                  >
                    <X className="w-3 h-3" />
                  </button>
                  <div className="absolute bottom-2 left-2 bg-black bg-opacity-60 text-white px-2 py-1 rounded text-xs">
                    {label}
                  </div>
                </div>
              ) : (
                <div className="aspect-square p-4 flex flex-col items-center justify-center text-center">
                  <div className="text-2xl mb-2">{icon}</div>
                  <h3 className="text-white font-medium text-sm mb-1">{label}</h3>
                  <p className="text-gray-400 text-xs mb-3">{description}</p>
                  <div className="space-y-2 w-full">
                    <button
                      onClick={() => openCamera(key)}
                      className="w-full bg-lime-400 text-gray-900 font-semibold py-2 rounded-lg text-xs flex items-center justify-center space-x-1"
                    >
                      <Camera className="w-3 h-3" />
                      <span>Camera</span>
                    </button>
                    <button
                      onClick={() => openGallery(key)}
                      className="w-full bg-gray-700 text-white font-semibold py-2 rounded-lg text-xs flex items-center justify-center space-x-1"
                    >
                      <Upload className="w-3 h-3" />
                      <span>Gallery</span>
                    </button>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Analysis Button */}
        {uploadedCount > 0 && !analysis && !isAnalyzing && (
          <button
            onClick={analyzeAllPhotos}
            className="w-full bg-lime-400 text-gray-900 font-semibold py-4 rounded-2xl hover:bg-lime-500 transition-colors"
          >
            Analyze Vehicle Condition ({uploadedCount} photo{uploadedCount !== 1 ? 's' : ''})
          </button>
        )}

        {/* Loading */}
        {isAnalyzing && (
          <div className="bg-gray-800 rounded-2xl p-6 text-center">
            <div className="w-8 h-8 border-2 border-lime-400 border-t-transparent rounded-full animate-spin mx-auto mb-3"></div>
            <p className="text-gray-400">Analyzing your vehicle...</p>
            <p className="text-gray-500 text-sm mt-1">Processing {uploadedCount} photo{uploadedCount !== 1 ? 's' : ''}...</p>
          </div>
        )}

        {/* Results */}
        {analysis && (
          <div className="space-y-4">
            {/* Overall Trust Score */}
            <div className="bg-gray-800 rounded-2xl p-6 text-center">
              <h2 className="text-white text-lg font-semibold mb-4">Overall Assessment</h2>
              <div className={`inline-flex items-center px-6 py-3 rounded-2xl font-bold text-xl ${getTrustScoreColor(getOverallTrustScore())}`}>
                Trust Score: {getOverallTrustScore()}%
              </div>
            </div>

            {/* Individual Results */}
            <div className="space-y-3">
              <h3 className="text-white font-semibold">Detailed Results</h3>
              {analysis.results && analysis.results.map((result, index) => {
                const sideInfo = carSides[index];
                if (!sideInfo || !photos[sideInfo.key]) return null;
                
                return (
                  <div key={sideInfo.key} className="bg-gray-800 rounded-xl p-4">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center space-x-3">
                        <span className="text-lg">{sideInfo.icon}</span>
                        <span className="text-white font-medium">{sideInfo.label}</span>
                      </div>
                      <div className={`px-3 py-1 rounded-full text-sm font-medium ${getTrustScoreColor(result.trust_score)}`}>
                        {result.trust_score}%
                      </div>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-300 text-sm">Condition</span>
                      <div className="flex items-center space-x-2">
                        <div className={`w-2 h-2 rounded-full ${getConditionDisplay(result.predicted_class).bg}`}></div>
                        <span className={`text-sm font-medium ${getConditionDisplay(result.predicted_class).color}`}>
                          {getConditionDisplay(result.predicted_class).text}
                        </span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Action Buttons */}
            <div className="space-y-3">
              {getOverallTrustScore() >= 80 && (
                <div className="bg-lime-400 bg-opacity-10 border border-lime-400 border-opacity-30 rounded-xl p-4">
                  <div className="flex items-start space-x-3">
                    <CheckCircle className="w-5 h-5 text-lime-400 mt-0.5" />
                    <div>
                      <p className="text-lime-400 font-semibold text-sm">Excellent Vehicle!</p>
                      <p className="text-gray-300 text-xs">Your car meets all quality standards for premium rides</p>
                    </div>
                  </div>
                </div>
              )}
              
              <button
                onClick={resetAll}
                className="w-full bg-gray-700 text-white font-semibold py-3 rounded-xl hover:bg-gray-600 transition-colors"
              >
                Check Another Vehicle
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

        {/* Error Display */}
        {error && (
          <div className="bg-red-500 bg-opacity-10 border border-red-500 border-opacity-30 rounded-xl p-4">
            <div className="flex items-start space-x-3">
              <AlertTriangle className="w-5 h-5 text-red-400 mt-0.5" />
              <div>
                <p className="text-red-400 font-semibold text-sm">Error</p>
                <p className="text-gray-300 text-xs">{error}</p>
              </div>
            </div>
          </div>
        )}

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
              <span className="text-lime-400 font-bold text-lg">&lt;5s</span>
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
              <h3 className="text-white font-semibold text-sm mb-1">Complete vehicle inspection</h3>
              <p className="text-gray-400 text-xs leading-relaxed">
                Upload all 4 sides for a comprehensive assessment. Complete inspections 
                result in 31% higher passenger trust and more ride requests.
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