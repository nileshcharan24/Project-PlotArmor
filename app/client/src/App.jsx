import React, { useState } from 'react';
import HemisphereSlider from './components/HemisphereSlider';
import SplitView from './components/SplitView';

const App = () => {
  const [prompt, setPrompt] = useState('');
  const [sliderValue, setSliderValue] = useState(50);
  const [loading, setLoading] = useState(false);
  const [bdhOutput, setBdhOutput] = useState('');
  const [gpt2Output, setGpt2Output] = useState('');

  const handleGenerate = () => {
    setLoading(true);
    setBdhOutput('');
    setGpt2Output('');
    setTimeout(() => {
      // Mock outputs
      setBdhOutput(
        <p>
          BDH output for prompt: <strong>"{prompt}"</strong> with logic-creativity balance at <strong>{sliderValue}</strong>
        </p>
      );
      setGpt2Output(
        <p>
          GPT-2 output for prompt: <strong>"{prompt}"</strong> with logic-creativity balance at <strong>{sliderValue}</strong>
        </p>
      );
      setLoading(false);
    }, 1000);
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <header className="max-w-4xl mx-auto mb-8">
        <h1 className="text-4xl font-bold text-center mb-2">Project PlotArmor</h1>
        <p className="text-center text-gray-600">Beyond the Black Box: Controllable Neural Storytelling</p>
      </header>
      <main className="max-w-4xl mx-auto flex flex-col gap-6">
        <textarea
          className="w-full p-3 border rounded shadow-sm resize-none"
          rows={4}
          placeholder="Enter your story prompt here..."
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          disabled={loading}
        />
        <HemisphereSlider value={sliderValue} onChange={(e) => setSliderValue(Number(e.target.value))} />
        <button
          className="px-6 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
          onClick={handleGenerate}
          disabled={loading || prompt.trim() === ''}
        >
          {loading ? 'Generating...' : 'Generate'}
        </button>
        <SplitView leftContent={bdhOutput} rightContent={gpt2Output} />
      </main>
    </div>
  );
};

export default App;
