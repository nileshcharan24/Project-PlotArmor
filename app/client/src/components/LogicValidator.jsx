import React, { useState } from 'react';
import Textarea from './Textarea';
import Button from './Button';

const LogicValidator = () => {
  const [context, setContext] = useState('');
  const [draft, setDraft] = useState('');
  const [validationResult, setValidationResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleValidate = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/validate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ context, draft }),
      });
      const data = await response.json();
      setValidationResult(data);
    } catch (error) {
      console.error('Error validating logic:', error);
    }
    setLoading(false);
  };

  const renderHighlightedDraft = () => {
    if (!validationResult || !validationResult.contradictions || validationResult.contradictions.length === 0) {
      return draft;
    }

    let lastIndex = 0;
    const parts = [];
    validationResult.contradictions.forEach((contradiction, i) => {
      const { start, end } = contradiction.span;
      if (start > lastIndex) {
        parts.push(draft.substring(lastIndex, start));
      }
      parts.push(
        <span key={i} className="bg-red-500 text-white px-1 rounded">
          {draft.substring(start, end)}
        </span>
      );
      lastIndex = end;
    });

    if (lastIndex < draft.length) {
      parts.push(draft.substring(lastIndex));
    }

    return parts;
  };
  
  return (
    <div>
      <label htmlFor="context" className="block text-sm font-medium text-gray-400">
        Context (Premise)
      </label>
      <Textarea
        id="context"
        value={context}
        onChange={(e) => setContext(e.target.value)}
        rows={4}
        placeholder="Enter context..."
      />

      <label htmlFor="draft" className="block text-sm font-medium text-gray-400 mt-4">
        Draft (New Text)
      </label>
      <Textarea
        id="draft"
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        rows={4}
        placeholder="Enter draft to validate..."
      />
      <Button
        onClick={handleValidate}
        disabled={loading || !context || !draft}
        className="mt-4"
        variant="secondary"
      >
        {loading ? 'Validating...' : 'Validate Logic'}
      </Button>

      {validationResult && (
        <div className="mt-4">
          <h3 className="font-semibold text-white">Validation Result</h3>
          <div className="mt-2 p-3 bg-[#0d1117] border border-gray-600 rounded-md min-h-[100px] text-gray-300 whitespace-pre-wrap">
            {renderHighlightedDraft()}
          </div>
          {validationResult.consistent === false && (
            <div className="mt-2 p-2 bg-red-900 border border-red-700 rounded-md text-red-300">
              Contradiction detected!
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default LogicValidator;
