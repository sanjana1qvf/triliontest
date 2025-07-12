// Vercel API route - Main backend entry point
const express = require('express');
const cors = require('cors');
const OpenAI = require('openai');
const Anthropic = require('@anthropic-ai/sdk');

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY
});

const app = express();

// Enhanced CORS configuration for Vercel
app.use(cors({
  origin: ['http://localhost:3000', 'http://localhost:3001', 'http://127.0.0.1:3000', 'http://127.0.0.1:3001', 'https://trilionclips.vercel.app', 'https://trilionclips-git-main-sanjana1qvf.vercel.app'],
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'Range']
}));

app.use(express.json());

// Test endpoint
app.get('/', (req, res) => {
  res.json({ 
    message: 'TrilionClips API is running on Vercel!',
    status: 'active',
    timestamp: new Date().toISOString(),
    endpoints: {
      analyze: '/api/analyze',
      analyzeViral: '/api/analyze-viral',
      test: '/api/test'
    }
  });
});

// Test endpoint
app.get('/test', (req, res) => {
  res.json({ 
    message: 'Backend is working on Vercel!',
    timestamp: new Date().toISOString()
  });
});

// Simple viral analysis endpoint (mock for now)
app.post('/analyze-viral', async (req, res) => {
  try {
    const { ytLink, numClips = 3, clipDuration = 30 } = req.body;
    
    if (!ytLink) {
      return res.status(400).json({ error: 'No YouTube link provided' });
    }

    console.log('Analyzing video for viral clips:', ytLink);

    // Generate mock viral clips (since Vercel is stateless, we can't process videos)
    const mockClips = [];
    for (let i = 0; i < numClips; i++) {
      const startTime = Math.floor(Math.random() * 300) + 60;
      const endTime = startTime + clipDuration;
      
      mockClips.push({
        id: `clip_${Date.now()}_${i}`,
        start_time: `${Math.floor(startTime / 60)}:${(startTime % 60).toString().padStart(2, '0')}`,
        end_time: `${Math.floor(endTime / 60)}:${(endTime % 60).toString().padStart(2, '0')}`,
        title: `Viral Moment ${i + 1} - This Will Blow Your Mind! 😱`,
        description: `This clip has viral potential due to its emotional impact and relatability.`,
        viral_score: 8.5 + (Math.random() * 1.5),
        predicted_views: Math.floor(Math.random() * 1000000) + 100000,
        thumbnail_url: `https://via.placeholder.com/300x200/FF6B6B/FFFFFF?text=Viral+Clip+${i + 1}`,
        download_url: `https://example.com/clip_${i}.mp4`
      });
    }

    res.json({
      success: true,
      message: 'Viral analysis completed on Vercel!',
      video_url: ytLink,
      clips: mockClips,
      analysis_summary: {
        total_clips: mockClips.length,
        average_viral_score: (mockClips.reduce((sum, clip) => sum + clip.viral_score, 0) / mockClips.length).toFixed(1),
        recommended_platforms: ['TikTok', 'Instagram Reels', 'YouTube Shorts']
      }
    });

  } catch (error) {
    console.error('Viral analysis error:', error);
    res.status(500).json({ 
      error: 'Failed to analyze video for viral clips',
      details: error.message 
    });
  }
});

// Simple analysis endpoint (mock for now)
app.post('/analyze', async (req, res) => {
  try {
    const { ytLink, numClips = 3, clipDuration = 30 } = req.body;
    
    if (!ytLink) {
      return res.status(400).json({ error: 'No YouTube link provided' });
    }

    console.log('Analyzing video:', ytLink);

    // Generate mock clips
    const mockClips = [];
    for (let i = 0; i < numClips; i++) {
      const startTime = Math.floor(Math.random() * 300) + 60;
      const endTime = startTime + clipDuration;
      
      mockClips.push({
        id: `clip_${Date.now()}_${i}`,
        start_time: `${Math.floor(startTime / 60)}:${(startTime % 60).toString().padStart(2, '0')}`,
        end_time: `${Math.floor(endTime / 60)}:${(endTime % 60).toString().padStart(2, '0')}`,
        title: `Clip ${i + 1} - Interesting Moment`,
        description: `This clip contains engaging content.`,
        thumbnail_url: `https://via.placeholder.com/300x200/4ECDC4/FFFFFF?text=Clip+${i + 1}`,
        download_url: `https://example.com/clip_${i}.mp4`
      });
    }

    res.json({
      success: true,
      message: 'Analysis completed on Vercel!',
      video_url: ytLink,
      clips: mockClips
    });

  } catch (error) {
    console.error('Analysis error:', error);
    res.status(500).json({ 
      error: 'Failed to analyze video',
      details: error.message 
    });
  }
});

// Export for Vercel
module.exports = app;
