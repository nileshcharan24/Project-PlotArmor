/**
 * Basic Express server boilerplate for the MERN application.
 * Handles API requests and serves the backend for model interactions.
 */

const express = require('express');
const app = express();
const port = process.env.PORT || 3000;

// Middleware
app.use(express.json());

// Basic route
app.get('/', (req, res) => {
  res.json({ message: 'BDH vs GPT-2 Benchmark API' });
});

// Placeholder for model switching endpoint
app.post('/switch-model', (req, res) => {
  const { model } = req.body;
  // Logic to switch model dynamically
  res.json({ currentModel: model });
});

app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});