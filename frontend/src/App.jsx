import React, { useState, useRef, useCallback } from 'react';
import { Camera, Upload, CheckCircle, AlertTriangle, X, Star, Shield, Clock, Users, ArrowLeft, Plus } from 'lucide-react';

// Image validation configuration
const IMAGE_CONFIG = {
  maxSize: 10 * 1024 * 1024, // 10MB
  minSize: 10 * 1024, // 10KB
  allowedTypes: ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'],
  allowedMimeTypes: ['image/jpeg', 'image/png', 'image/webp'],
  maxDimension: 4096,
  minDimension: 224,
  requiredSides: 4,
  maxFileNameLength: 255,
  suspiciousFileNames: [
    'script', 'payload', 'exploit', 'hack', 'malware', 'virus',
    '.exe', '.bat', '.cmd', '.sh', '.ps1', '.scr', '.com', '.pif'
  ]
};

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
  const [isValidating, setIsValidating] = useState(false);
  const [analysis, setAnalysis] = useState(null);
  const [error, setError] = useState(null);
  const [successMessage, setSuccessMessage] = useState(null);
  const [currentSide, setCurrentSide] = useState(null);
  const [sideMismatchWarning, setSideMismatchWarning] = useState(null);
  const fileInputRef = useRef(null);
  const cameraInputRef = useRef(null);

  // Replace with your actual Render deployment URL
  const API_BASE_URL = import.meta.env.VITE_API_BASE;

  const carSides = [
    { key: 'front', label: 'Front Side', icon: '🚗', description: 'Front bumper and headlights', tip: 'Face the front of the car' },
    { key: 'back', label: 'Back Side', icon: '🚙', description: 'Rear bumper and taillights', tip: 'Face the back/rear of the car' },
    { key: 'left', label: 'Left Side', icon: '🚘', description: 'Driver side doors and windows', tip: 'Left side when facing forward' },
    { key: 'right', label: 'Right Side', icon: '🚖', description: 'Passenger side doors and windows', tip: 'Right side when facing forward' }
  ];

  // Helper function to get image dimensions
  const getImageDimensions = (file) => {
    return new Promise((resolve, reject) => {
      const img = new Image();
      const url = URL.createObjectURL(file);
      
      img.onload = () => {
        URL.revokeObjectURL(url);
        resolve({
          width: img.naturalWidth,
          height: img.naturalHeight
        });
      };
      
      img.onerror = () => {
        URL.revokeObjectURL(url);
        reject(new Error('Failed to load image'));
      };
      
      img.src = url;
    });
  };

  // Simple file hash for duplicate detection
  const getFileHash = async (file) => {
    const buffer = await file.arrayBuffer();
    const hashBuffer = await crypto.subtle.digest('SHA-256', buffer);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray.map(b => b.toString(16).padStart(2, '0')).join('').substring(0, 16);
  };

  // Enhanced image validation function
  const validateImage = async (file) => {
    const errors = [];

    // Check file type
    if (!IMAGE_CONFIG.allowedTypes.includes(file.type.toLowerCase())) {
      errors.push(`Invalid file type. Please use JPG, PNG, or WebP format.`);
    }

    // Check file size
    if (file.size > IMAGE_CONFIG.maxSize) {
      errors.push(`File size too large (${(file.size / 1024 / 1024).toFixed(2)}MB). Maximum allowed is ${IMAGE_CONFIG.maxSize / 1024 / 1024}MB.`);
    }

    if (file.size < IMAGE_CONFIG.minSize) {
      errors.push(`File size too small (${(file.size / 1024).toFixed(2)}KB). Minimum required is ${IMAGE_CONFIG.minSize / 1024}KB.`);
    }

    // Check image dimensions
    try {
      const dimensions = await getImageDimensions(file);
      
      if (dimensions.width > IMAGE_CONFIG.maxDimension || dimensions.height > IMAGE_CONFIG.maxDimension) {
        errors.push(`Image dimensions too large (${dimensions.width}x${dimensions.height}). Maximum allowed is ${IMAGE_CONFIG.maxDimension}x${IMAGE_CONFIG.maxDimension}.`);
      }

      if (dimensions.width < IMAGE_CONFIG.minDimension || dimensions.height < IMAGE_CONFIG.minDimension) {
        errors.push(`Image dimensions too small (${dimensions.width}x${dimensions.height}). Minimum required is ${IMAGE_CONFIG.minDimension}x${IMAGE_CONFIG.minDimension}.`);
      }

      // Check aspect ratio (should be reasonable for vehicle photos)
      const aspectRatio = dimensions.width / dimensions.height;
      if (aspectRatio > 3 || aspectRatio < 0.33) {
        errors.push(`Unusual aspect ratio (${aspectRatio.toFixed(2)}:1). Please use standard vehicle photo proportions.`);
      }
    } catch (error) {
      errors.push('Unable to read image dimensions. File may be corrupted.');
    }

    return {
      isValid: errors.length === 0,
      errors: errors
    };
  };

  // Check for duplicate images (simple implementation)
  const checkForDuplicateImage = async (newFile, excludeSide) => {
    const newFileHash = await getFileHash(newFile);
    
    for (const [side, existingFile] of Object.entries(photos)) {
      if (side === excludeSide || !existingFile) continue;
      
      try {
        const existingHash = await getFileHash(existingFile);
        if (newFileHash === existingHash) {
          return true;
        }
      } catch (error) {
        console.warn('Error comparing file hashes:', error);
      }
    }
    
    return false;
  };

  // Enhanced handleImageSelect function
  const handleImageSelect = useCallback(async (file, side) => {
    if (!file) return;
    
    setError(null);
    setSuccessMessage(null);
    
    // Show loading state while validating
    setIsValidating(true);
    
    try {
      // Comprehensive validation
      const validation = await validateImage(file);
      
      if (!validation.isValid) {
        setError(validation.errors.join(' '));
        return;
      }

      // Check for duplicate images (basic hash comparison)
      const isDuplicate = await checkForDuplicateImage(file, side);
      if (isDuplicate) {
        setError('This image appears to be a duplicate of another side. Please use different photos for each vehicle side.');
        return;
      }

      // Clean up previous preview URL for this side
      if (previewUrls[side]) {
        URL.revokeObjectURL(previewUrls[side]);
      }
      
      const url = URL.createObjectURL(file);
      
      setPhotos(prev => ({ ...prev, [side]: file }));
      setPreviewUrls(prev => ({ ...prev, [side]: url }));
      setCurrentSide(null);
      
      // Success feedback
      setSuccessMessage(`${carSides.find(s => s.key === side)?.label} photo uploaded successfully!`);
      setTimeout(() => setSuccessMessage(null), 3000);
      
    } catch (error) {
      setError(`Failed to validate image: ${error.message}`);
    } finally {
      setIsValidating(false);
    }
  }, [previewUrls, photos]);

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
    setSuccessMessage(null);
  };

  // Enhanced analyzeAllPhotos function
  const analyzeAllPhotos = async () => {
    const uploadedPhotos = Object.values(photos).filter(Boolean);
    const uploadedSides = Object.entries(photos).filter(([_, file]) => file).map(([side, _]) => side);
    
    // Validation checks
    if (uploadedPhotos.length === 0) {
      setError('Please upload at least one photo to continue.');
      return;
    }

    if (uploadedPhotos.length < IMAGE_CONFIG.requiredSides) {
      const missingSides = carSides
        .filter(side => !photos[side.key])
        .map(side => side.label)
        .join(', ');
      
      setError(`Please upload all ${IMAGE_CONFIG.requiredSides} required photos. Missing: ${missingSides}`);
      return;
    }

    // Check if all uploaded images are still valid (in case of corruption)
    for (const [side, file] of Object.entries(photos)) {
      if (!file) continue;
      
      try {
        const validation = await validateImage(file);
        if (!validation.isValid) {
          setError(`${carSides.find(s => s.key === side)?.label} photo is invalid: ${validation.errors[0]}`);
          return;
        }
      } catch (error) {
        setError(`Failed to validate ${carSides.find(s => s.key === side)?.label} photo: ${error.message}`);
        return;
      }
    }

    setIsAnalyzing(true);
    setError(null);
    
    try {
      const formData = new FormData();
      
      // Add photos in the correct order (front, back, left, right)
      carSides.forEach(({ key }) => {
        if (photos[key]) {
          // Add metadata to filename for better tracking
          const timestamp = new Date().getTime();
          const filename = `${key}_side_${timestamp}.jpg`;
          formData.append('files', photos[key], filename);
        }
      });

      // Add metadata
      formData.append('metadata', JSON.stringify({
        uploadedSides: uploadedSides,
        totalPhotos: uploadedPhotos.length,
        timestamp: new Date().toISOString(),
        clientVersion: '3.0.0'
      }));

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 60000); // 60 second timeout

      const response = await fetch(`${API_BASE_URL}/inspect-vehicle`, {
        method: 'POST',
        body: formData,
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.details || errorData.error || `Server error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      
      if (data.error) {
        throw new Error(data.details || data.error);
      }

      // Validate response structure
      if (!data.results || !Array.isArray(data.results)) {
        throw new Error('Invalid response format from server');
      }

      // Check for vehicle side mismatches using the new API format
      const sideMismatches = [];
      
      if (data.side_results) {
        Object.entries(data.side_results).forEach(([side, result]) => {
          if (result.validation_status === "failed" && result.validation_error?.error_type === "wrong_side") {
            const expectedLabel = carSides.find(s => s.key === side)?.label;
            const detectedSide = result.detected_vehicle_side;
            const detectedLabel = carSides.find(s => s.key === detectedSide)?.label;
            
            console.log('Mismatch detected:', {
              side,
              detectedSide,
              expectedLabel,
              detectedLabel,
              result: result.validation_error
            });
            
            sideMismatches.push({
              expectedSide: side,
              detectedSide: detectedSide,
              expectedLabel: expectedLabel,
              detectedLabel: detectedLabel || detectedSide, // Fallback to detected side
              confidence: result.detection_confidence,
              message: result.validation_error.message,
              suggestion: result.validation_error.suggestion
            });
          }
        });
      }

      // If there are side mismatches, show warning
      if (sideMismatches.length > 0) {
        setSideMismatchWarning({
          mismatches: sideMismatches,
          analysisData: data
        });
        return; // Stop analysis and show warning modal
      }

      setAnalysis(data);
      
    } catch (err) {
      console.error('Analysis error:', err);
      
      if (err.name === 'AbortError') {
        setError('Analysis timed out. Please check your connection and try again.');
      } else if (err.message.includes('Failed to fetch')) {
        setError('Network error. Please check your internet connection and try again.');
      } else {
        setError(err.message || 'Failed to analyze images. Please try again.');
      }
      
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
    setSuccessMessage(null);
    setCurrentSide(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
    if (cameraInputRef.current) cameraInputRef.current.value = '';
  };

  const getOverallTrustScore = () => {
    if (!analysis?.summary?.overall_cleanliness_score) return 0;
    return Math.round(analysis.summary.overall_cleanliness_score);
  };

  const getTrustScoreColor = (score) => {
    if (score >= 80) return 'text-white bg-lime-400';
    if (score >= 60) return 'text-gray-900 bg-yellow-400';
    return 'text-white bg-red-500';
  };

  const getConditionDisplay = (condition) => {
    const displays = {
      'very clean': { text: 'Excellent', color: 'text-lime-400', bg: 'bg-lime-400' },
      'clean': { text: 'Clean', color: 'text-lime-400', bg: 'bg-lime-400' },
      'slightly dirty': { text: 'Minor Cleaning', color: 'text-yellow-400', bg: 'bg-yellow-400' },
      'dirty': { text: 'Needs Cleaning', color: 'text-orange-400', bg: 'bg-orange-400' },
      'very dirty': { text: 'Deep Cleaning Required', color: 'text-red-400', bg: 'bg-red-400' },
      // Legacy support
      'scratchless': { text: 'Excellent', color: 'text-lime-400', bg: 'bg-lime-400' },
      'scratched': { text: 'Minor Damage', color: 'text-red-400', bg: 'bg-red-400' }
    };
    return displays[condition] || displays['clean'];
  };

  // Validation indicator component for the upload cards
  const ValidationIndicator = ({ side }) => {
    if (isValidating && currentSide === side) {
      return (
        <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center">
          <div className="w-6 h-6 border-2 border-lime-400 border-t-transparent rounded-full animate-spin"></div>
        </div>
      );
    }
    return null;
  };

  // Success message component
  const SuccessMessage = () => (
    successMessage && (
      <div className="bg-green-500 bg-opacity-10 border border-green-500 border-opacity-30 rounded-xl p-4">
        <div className="flex items-start space-x-3">
          <CheckCircle className="w-5 h-5 text-green-400 mt-0.5" />
          <div>
            <p className="text-green-400 font-semibold text-sm">Success</p>
            <p className="text-gray-300 text-xs">{successMessage}</p>
          </div>
        </div>
      </div>
    )
  );

  // Handle cancelling analysis due to side mismatch
  const fixSideMismatch = () => {
    setSideMismatchWarning(null);
    setError('Analysis cancelled. Please upload images to their correct vehicle side slots for accurate results.');
    setIsAnalyzing(false);
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
          {carSides.map(({ key, label, icon, description, tip }) => (
            <div key={key} className="bg-gray-800 rounded-xl overflow-hidden relative">
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
                  <p className="text-gray-400 text-xs mb-2">{description}</p>
                  <p className="text-lime-400 text-xs mb-3 font-medium">📍 {tip}</p>
                  <div className="space-y-2 w-full">
                    <button
                      onClick={() => openCamera(key)}
                      disabled={isValidating}
                      className="w-full bg-lime-400 text-gray-900 font-semibold py-2 rounded-lg text-xs flex items-center justify-center space-x-1 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      <Camera className="w-3 h-3" />
                      <span>Camera</span>
                    </button>
                    <button
                      onClick={() => openGallery(key)}
                      disabled={isValidating}
                      className="w-full bg-gray-700 text-white font-semibold py-2 rounded-lg text-xs flex items-center justify-center space-x-1 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      <Upload className="w-3 h-3" />
                      <span>Gallery</span>
                    </button>
                  </div>
                </div>
              )}
              <ValidationIndicator side={key} />
            </div>
          ))}
        </div>

        {/* Success Message */}
        <SuccessMessage />

        {/* Side Mismatch Warning Modal */}
        {sideMismatchWarning && (
          <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center p-4 z-50">
            <div className="bg-gray-800 rounded-2xl p-6 max-w-md w-full">
              <div className="flex items-start space-x-3 mb-4">
                <AlertTriangle className="w-6 h-6 text-red-400 mt-0.5 flex-shrink-0" />
                <div>
                  <h3 className="text-red-400 font-semibold text-lg mb-2">⛔ Incorrect Image Placement</h3>
                  <p className="text-gray-300 text-sm mb-4">
                    Our AI detected incorrect vehicle side placements. You must fix these before proceeding:
                  </p>
                </div>
              </div>
              
              <div className="space-y-3 mb-6">
                {sideMismatchWarning.mismatches.map((mismatch, index) => (
                  <div key={index} className="bg-yellow-500 bg-opacity-10 border border-yellow-500 border-opacity-30 rounded-lg p-3">
                    <div className="flex items-center justify-between">
                      <span className="text-white text-sm">
                        <span className="font-medium">{mismatch.expectedLabel}</span> slot
                      </span>
                      <span className="text-yellow-400 text-xs">
                        {Math.round(mismatch.confidence * 100)}% confidence
                      </span>
                    </div>
                    <p className="text-gray-300 text-xs mt-1">
                      Contains <span className="text-yellow-400 font-medium">{mismatch.detectedLabel || mismatch.detectedSide}</span> view instead
                    </p>
                  </div>
                ))}
              </div>

              <div className="text-gray-300 text-sm mb-6">
                <p className="mb-2">🎯 <strong>For best results:</strong></p>
                <ul className="text-xs space-y-1 ml-4">
                  <li>• Upload images to their correct vehicle side slots</li>
                  <li>• This ensures accurate damage detection and scoring</li>
                  <li>• Wrong placement may affect your vehicle trust score</li>
                </ul>
              </div>

              <div className="space-y-3">
                <button
                  onClick={fixSideMismatch}
                  className="w-full bg-lime-400 text-gray-900 font-semibold py-3 rounded-xl hover:bg-lime-500 transition-colors"
                >
                  Fix Images Now
                </button>
                <p className="text-center text-gray-400 text-xs">
                  Analysis blocked until all images are correctly placed
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Analysis Button */}
        {uploadedCount > 0 && !analysis && !isAnalyzing && (
          <button
            onClick={analyzeAllPhotos}
            disabled={isValidating}
            className="w-full bg-lime-400 text-gray-900 font-semibold py-4 rounded-2xl hover:bg-lime-500 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Analyze Vehicle Condition ({uploadedCount} photo{uploadedCount !== 1 ? 's' : ''})
          </button>
        )}

        {/* Loading */}
        {(isAnalyzing || isValidating) && (
          <div className="bg-gray-800 rounded-2xl p-6 text-center">
            <div className="w-8 h-8 border-2 border-lime-400 border-t-transparent rounded-full animate-spin mx-auto mb-3"></div>
            <p className="text-gray-400">
              {isValidating ? 'Validating image...' : 'Analyzing your vehicle...'}
            </p>
            {isAnalyzing && (
              <p className="text-gray-500 text-sm mt-1">Processing {uploadedCount} photo{uploadedCount !== 1 ? 's' : ''}...</p>
            )}
          </div>
        )}

        {/* Results */}
        {analysis && (
          <div className="space-y-4">
            {/* Overall Trust Score */}
            <div className="bg-gray-800 rounded-2xl p-6 text-center">
              <h2 className="text-white text-lg font-semibold mb-4">Overall Assessment</h2>
              <div className={`inline-flex items-center px-6 py-3 rounded-2xl font-bold text-xl ${getTrustScoreColor(getOverallTrustScore())}`}>
                Cleanliness Score: {getOverallTrustScore()}%
              </div>
              {analysis.summary?.assessment_message && (
                <p className="text-gray-300 text-sm mt-3">{analysis.summary.assessment_message}</p>
              )}
            </div>

            {/* Validation Warnings */}
            {analysis.validation?.warnings && analysis.validation.warnings.length > 0 && (
              <div className="bg-yellow-500 bg-opacity-10 border border-yellow-500 border-opacity-30 rounded-xl p-4">
                <div className="flex items-start space-x-3">
                  <AlertTriangle className="w-5 h-5 text-yellow-400 mt-0.5" />
                  <div>
                    <p className="text-yellow-400 font-semibold text-sm">Photo Quality Notes</p>
                    <ul className="text-gray-300 text-xs mt-1 space-y-1">
                      {analysis.validation.warnings.map((warning, index) => (
                        <li key={index}>• {warning}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            )}

            {/* Individual Results */}
            <div className="space-y-3">
              <h3 className="text-white font-semibold">Detailed Results</h3>
              {analysis.side_results && Object.entries(analysis.side_results).map(([side, result]) => {
                const sideInfo = carSides.find(s => s.key === side);
                if (!sideInfo || !photos[side]) return null;
                
                // Show error results
                if (result.processing_status === "failed") {
                  return (
                    <div key={`error-${side}`} className="bg-red-500 bg-opacity-10 border border-red-500 border-opacity-30 rounded-xl p-4">
                      <div className="flex items-start space-x-3">
                        <AlertTriangle className="w-5 h-5 text-red-400 mt-0.5" />
                        <div>
                          <p className="text-red-400 font-semibold text-sm">{sideInfo.label} - Processing Failed</p>
                          <p className="text-gray-300 text-xs">{result.processing_error}</p>
                        </div>
                      </div>
                    </div>
                  );
                }

                // Show validation error results
                if (result.validation_status === "failed") {
                  return (
                    <div key={`validation-${side}`} className="bg-yellow-500 bg-opacity-10 border border-yellow-500 border-opacity-30 rounded-xl p-4">
                      <div className="flex items-start space-x-3">
                        <AlertTriangle className="w-5 h-5 text-yellow-400 mt-0.5" />
                        <div>
                          <p className="text-yellow-400 font-semibold text-sm">{sideInfo.label} - Validation Issue</p>
                          <p className="text-gray-300 text-xs">{result.validation_error?.message}</p>
                          {result.validation_error?.suggestion && (
                            <p className="text-gray-400 text-xs mt-1">💡 {result.validation_error.suggestion}</p>
                          )}
                        </div>
                      </div>
                    </div>
                  );
                }
                
                return (
                  <div key={side} className="bg-gray-800 rounded-xl p-4">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center space-x-3">
                        <span className="text-lg">{sideInfo.icon}</span>
                        <span className="text-white font-medium">{sideInfo.label}</span>
                      </div>
                      <div className={`px-3 py-1 rounded-full text-sm font-medium ${getTrustScoreColor(result.cleanliness_score || 0)}`}>
                        {result.cleanliness_score || 0}%
                      </div>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-300 text-sm">Condition</span>
                      <div className="flex items-center space-x-2">
                        <div className={`w-2 h-2 rounded-full ${getConditionDisplay(result.cleanliness_category).bg}`}></div>
                        <span className={`text-sm font-medium ${getConditionDisplay(result.cleanliness_category).color}`}>
                          {getConditionDisplay(result.cleanliness_category).text}
                        </span>
                      </div>
                    </div>
                    {result.detection_confidence && (
                      <div className="flex items-center justify-between mt-2">
                        <span className="text-gray-400 text-xs">Detection Confidence</span>
                        <span className="text-gray-300 text-xs">{Math.round(result.detection_confidence * 100)}%</span>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>

            {/* Recommendations */}
            {analysis.recommendations && analysis.recommendations.length > 0 && (
              <div className="bg-blue-500 bg-opacity-10 border border-blue-500 border-opacity-30 rounded-xl p-4">
                <div className="flex items-start space-x-3">
                  <Shield className="w-5 h-5 text-blue-400 mt-0.5" />
                  <div>
                    <p className="text-blue-400 font-semibold text-sm">Recommendations</p>
                    <ul className="text-gray-300 text-xs mt-2 space-y-1">
                      {analysis.recommendations.map((recommendation, index) => (
                        <li key={index}>• {recommendation}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            )}

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
        accept="image/jpeg,image/jpg,image/png,image/webp"
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