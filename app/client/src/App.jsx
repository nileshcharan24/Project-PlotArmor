import React, { useState } from 'react';
import HemisphereSlider from './components/HemisphereSlider';
import SplitView from './components/SplitView';
import LogicValidator from './components/LogicValidator';
import Card from './components/Card';
import Textarea from './components/Textarea';
import Button from './components/Button';

const App = () => {
  const [prompt, setPrompt] = useState('');
  const [sliderValue, setSliderValue] = useState(50);
  const [loading, setLoading] = useState(false);
  const [bdhOutput, setBdhOutput] = useState('');
  const [gpt2Output, setGpt2Output] = useState('');

  const handleGenerate = async () => {
    setLoading(true);
    // Clear previous outputs before generating
    setBdhOutput('');
    setGpt2Output('');

    try {
      // NOTE: Using '/generate' to match your main.py configuration. 
      // If you changed main.py to prefix '/api', change this URL to 'http://localhost:8000/api/generate'
      const response = await fetch('http://localhost:8000/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          context: prompt,
          slider_value: sliderValue, // Passing the current slider value
          max_tokens: 50,
          temperature: 1.0,
          top_k: 50
        }),
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`);
      }

      const data = await response.json();
      
      // Update state with the real data from the backend
      setBdhOutput(data.bdh_text);
      setGpt2Output(data.gpt_text);

    } catch (error) {
      console.error("Failed to generate text:", error);
      alert("Failed to connect to the backend. Is the server running?");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0b1220] text-gray-300 p-8">
      <header className="text-center mb-12">
        <h1 className="text-5xl font-bold text-white">Project PlotArmor</h1>
        <p className="text-gray-400 mt-2">Beyond the Black Box: Controllable Neural Storytelling</p>
      </header>

      <main className="grid grid-cols-1 md:grid-cols-2 gap-8 max-w-7xl mx-auto">
        {/* Story Generation Card */}
        <Card title="Story Generation">
          <HemisphereSlider
            value={sliderValue}
            onChange={(e) => setSliderValue(Number(e.target.value))}
          />
          <Textarea
            rows={4}
            placeholder="Enter your story prompt here..."
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            disabled={loading}
          />
          <Button
            onClick={handleGenerate}
            disabled={loading || prompt.trim() === ''}
          >
            {loading ? 'Generating...' : 'Generate'}
          </Button>
          <SplitView
            leftContent={bdhOutput}
            rightContent={gpt2Output}
            loading={loading}
            sliderValue={sliderValue}
          />
        </Card>

        {/* Logic Validator Card */}
        <Card title="Logic Validator">
          <LogicValidator />
        </Card>
      </main>
    </div>
  );
};

export default App;
