import React, { useState, useRef } from 'react';
import { UploadCloud, X, AlertTriangle, Activity } from 'lucide-react';

export default function GlaucomaDashboard() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState('');
  const [isDragging, setIsDragging] = useState(false);
  
  const fileInputRef = useRef(null);

  const handleFileChange = (file) => {
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResults(null);
      setError('');
    } else {
      setError('Please select a valid image file (JPG, PNG, TIF)');
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const clearSelection = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setResults(null);
    setError('');
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const analyzeImage = async (e) => {
    e.preventDefault();
    if (!selectedFile) return;

    setIsLoading(true);
    setError('');
    setResults(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      // Assuming the API is hosted on the same domain or CORS is enabled
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to process image. Make sure the API is running.');
      }

      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError(err.message || 'An error occurred during prediction.');
    } finally {
      setIsLoading(false);
    }
  };

  const getPredictionColor = (prediction) => {
    if (prediction === 'NORMAL') return 'text-emerald-600';
    if (prediction.includes('SUSPECT')) return 'text-amber-500';
    return 'text-rose-600';
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 p-4 md:p-8 font-sans selection:bg-blue-100 flex items-center justify-center">
      <div className="w-full max-w-4xl space-y-6">
        
        {/* Header */}
        <div className="text-center space-y-2">
          <h1 className="text-3xl font-bold tracking-tight text-slate-800">Glaucoma Screening AI</h1>
          <p className="text-slate-500 font-light">Upload a retinal fundus image for instant U-Net segmentation and Random Forest risk analysis.</p>
        </div>

        {/* Upload Card */}
        <div className="bg-white/90 backdrop-blur-md p-8 rounded-2xl shadow-sm border border-slate-200/80">
          <form onSubmit={analyzeImage} className="space-y-6">
            
            {/* Drop Zone */}
            <div 
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => !selectedFile && fileInputRef.current.click()}
              className={`border-2 border-dashed rounded-xl p-10 text-center transition-colors relative
                ${isDragging ? 'border-blue-500 bg-blue-50' : 'border-slate-300 hover:bg-slate-50/50'}
                ${!selectedFile ? 'cursor-pointer' : ''}`}
            >
              <input
                type="file"
                ref={fileInputRef}
                onChange={(e) => handleFileChange(e.target.files[0])}
                accept="image/*"
                className="hidden"
              />

              {!selectedFile ? (
                <div className="space-y-4 pointer-events-none">
                  <UploadCloud className="mx-auto h-12 w-12 text-slate-400" strokeWidth={1.5} />
                  <div className="text-sm text-slate-600">
                    <span className="font-semibold text-blue-600">Click to upload</span> or drag and drop
                  </div>
                  <p className="text-xs text-slate-400">JPG, PNG or TIF</p>
                </div>
              ) : (
                <div className="relative inline-block">
                  <img src={previewUrl} alt="Preview" className="h-48 rounded shadow-sm object-cover mx-auto" />
                  <button 
                    type="button" 
                    onClick={(e) => { e.stopPropagation(); clearSelection(); }}
                    className="absolute -top-3 -right-3 bg-white text-slate-600 rounded-full p-1 shadow hover:bg-slate-100 border border-slate-200"
                  >
                    <X size={16} />
                  </button>
                  <p className="mt-3 text-sm text-slate-500 truncate max-w-xs mx-auto">{selectedFile.name}</p>
                </div>
              )}
            </div>

            {/* Error Message */}
            {error && (
              <div className="text-red-500 text-sm text-center bg-red-50 p-2 rounded-lg border border-red-100">
                {error}
              </div>
            )}

            {/* Action Bar */}
            <div className="flex justify-end pt-2">
              <button 
                type="submit" 
                disabled={!selectedFile || isLoading}
                className="flex items-center gap-2 bg-slate-800 text-white px-6 py-2.5 rounded-lg font-medium hover:bg-slate-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
              >
                {isLoading ? (
                  <>
                    <Activity className="animate-spin" size={18} />
                    Processing...
                  </>
                ) : (
                  'Analyze Image'
                )}
              </button>
            </div>
          </form>
        </div>

        {/* Results Dashboard */}
        {results && (
          <div className="bg-white/90 backdrop-blur-md p-8 rounded-2xl shadow-sm border border-slate-200/80 space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
            
            {/* Result Header */}
            <div className="flex flex-col md:flex-row items-center justify-between border-b border-slate-100 pb-6 gap-4">
              <div className="space-y-1 text-center md:text-left">
                <p className="text-sm text-slate-500 font-medium uppercase tracking-wider">AI Diagnosis</p>
                <h2 className={`text-3xl font-bold ${getPredictionColor(results.prediction)}`}>
                  {results.prediction.replace('_', ' ')}
                </h2>
                <p className="text-slate-500">
                  {(results.confidence * 100).toFixed(1)}% Confidence
                </p>
              </div>

              {/* Warning Badge */}
              {results.anomaly_flag && (
                <div className="flex items-center gap-2 bg-amber-50 text-amber-700 px-4 py-2 rounded-lg text-sm border border-amber-200">
                  <AlertTriangle size={18} />
                  <span>Unusual image dimensions detected</span>
                </div>
              )}
            </div>

            {/* Content Split */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-start">
              
              {/* Annotated Image */}
              <div className="space-y-3">
                <p className="text-sm font-semibold text-slate-700">Segmentation Mask</p>
                <div className="rounded-xl overflow-hidden shadow-sm border border-slate-200 bg-white">
                  <img 
                    src={`data:image/${results.annotated_image_format};base64,${results.annotated_image_base64}`} 
                    className="w-full h-auto object-cover" 
                    alt="Annotated Result" 
                  />
                </div>
              </div>

              {/* Clinical Measurements */}
              <div className="space-y-6">
                <div className="space-y-3">
                  <p className="text-sm font-semibold text-slate-700">Clinical Measurements</p>
                  
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-white p-4 rounded-xl border border-slate-100 shadow-sm">
                      <p className="text-xs text-slate-400 font-medium uppercase mb-1">VCDR</p>
                      <p className="text-xl font-semibold text-slate-800">{results.features.VCDR.toFixed(3)}</p>
                    </div>
                    <div className="bg-white p-4 rounded-xl border border-slate-100 shadow-sm">
                      <p className="text-xs text-slate-400 font-medium uppercase mb-1">ACDR</p>
                      <p className="text-xl font-semibold text-slate-800">{results.features.ACDR.toFixed(3)}</p>
                    </div>
                    <div className="bg-white p-4 rounded-xl border border-slate-100 shadow-sm">
                      <p className="text-xs text-slate-400 font-medium uppercase mb-1">HCDR</p>
                      <p className="text-xl font-semibold text-slate-800">{results.features.HCDR.toFixed(3)}</p>
                    </div>
                    <div className="bg-slate-50 p-4 rounded-xl border border-slate-100">
                      <p className="text-xs text-slate-400 font-medium uppercase mb-1">Model Engine</p>
                      <p className="text-sm font-semibold text-slate-600 capitalize">
                        {results.model_mode.replace('_', ' ')}
                      </p>
                    </div>
                  </div>
                </div>
                
                <div className="bg-blue-50 text-blue-800 text-sm p-4 rounded-lg border border-blue-100">
                  <p className="font-medium mb-1">Interpretation Guide</p>
                  <p className="font-light text-blue-700 opacity-90">
                    Glaucoma Suspect logic triggers if VCDR &gt; 0.50 or ACDR &gt; 0.30. The final decision relies on a trained Random Forest bounding classifier to map exact thresholds.
                  </p>
                </div>
              </div>

            </div>
          </div>
        )}
      </div>
    </div>
  );
}