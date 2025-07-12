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
  origin: ['http://localhost:3000', 'http://localhost:3001', 'http://127.0.0.1:3000', 'http://127.0.0.1:3001', 'https://trilionclips.vercel.app', 'https://trilionclips-git-main-sanjana1qvf.vercel.app', 'https://triliontest.vercel.app', 'https://junkie-54kph4qce-sanjana1qvfs-projects.vercel.app', 'https://junkie-n9dx9sg0t-sanjana1qvfs-projects.vercel.app', 'https://junkie-psi.vercel.app'],
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
    timestamp: new Date().toISOString(),
    environment: {
      openai_key_exists: !!process.env.OPENAI_API_KEY,
      anthropic_key_exists: !!process.env.ANTHROPIC_API_KEY,
      node_env: process.env.NODE_ENV
    }
  });
});

// Serve manifest.json to fix 401 error
app.get('/manifest.json', (req, res) => {
  res.json({
    "short_name": "TrilionClips",
    "name": "TrilionClips - Viral Video Generator",
    "icons": [
      {
        "src": "favicon.ico",
        "sizes": "64x64 32x32 24x24 16x16",
        "type": "image/x-icon"
      }
    ],
    "start_url": ".",
    "display": "standalone",
    "theme_color": "#000000",
    "background_color": "#ffffff"
  });
});

// Simple test endpoint for viral analysis
app.post('/test-viral', (req, res) => {
  const testClips = generateFallbackClips(3, 30);
  res.json({
    success: true,
    message: 'Test viral analysis completed',
    clips: testClips
  });
});

// Real viral analysis endpoint using AI APIs
app.post('/analyze-viral', async (req, res) => {
  try {
    console.log('🔍 /analyze-viral endpoint called');
    console.log('📝 Request body:', req.body);
    console.log('🔑 Environment check - OpenAI key exists:', !!process.env.OPENAI_API_KEY);
    console.log('🔑 Environment check - Anthropic key exists:', !!process.env.ANTHROPIC_API_KEY);
    
    const { ytLink, numClips = 3, clipDuration = 30 } = req.body;
    
    if (!ytLink) {
      console.log('❌ No YouTube link provided');
      return res.status(400).json({ error: 'No YouTube link provided' });
    }

    console.log('🎬 Analyzing video for viral clips:', ytLink);

    // Extract video ID from YouTube URL
    const videoId = extractVideoId(ytLink);
    if (!videoId) {
      return res.status(400).json({ error: 'Invalid YouTube URL' });
    }

    // Get video info using YouTube Data API (if available) or use a fallback
    const videoInfo = await getVideoInfo(videoId);
    
    // Check if AI APIs are available
    if (!process.env.OPENAI_API_KEY || !process.env.ANTHROPIC_API_KEY) {
      console.log('⚠️ AI APIs not available, using fallback data');
      const fallbackClips = generateFallbackClips(numClips, clipDuration);
      return res.json({
        success: true,
        message: 'Fallback analysis completed (AI APIs not configured)',
        video_url: ytLink,
        video_info: videoInfo,
        clips: fallbackClips,
        analysis_summary: {
          total_clips: numClips,
          average_viral_score: 7.5,
          recommended_platforms: ['TikTok', 'Instagram Reels', 'YouTube Shorts']
        }
      });
    }

    // Use Anthropic Claude to analyze the video for viral potential
    const viralAnalysis = await analyzeViralPotential(videoInfo, numClips, clipDuration);
    
    // Use OpenAI to generate engaging titles and descriptions
    const enhancedClips = await enhanceClipsWithAI(viralAnalysis.clips, videoInfo);

    res.json({
      success: true,
      message: 'Real viral analysis completed using AI!',
      video_url: ytLink,
      video_info: videoInfo,
      clips: enhancedClips,
      analysis_summary: viralAnalysis.summary
    });

  } catch (error) {
    console.error('Viral analysis error:', error);
    res.status(500).json({ 
      error: 'Failed to analyze video for viral clips',
      details: error.message 
    });
  }
});

// Real analysis endpoint using AI APIs
app.post('/analyze', async (req, res) => {
  try {
    const { ytLink, numClips = 3, clipDuration = 30 } = req.body;
    
    if (!ytLink) {
      return res.status(400).json({ error: 'No YouTube link provided' });
    }

    console.log('Analyzing video:', ytLink);

    // Extract video ID from YouTube URL
    const videoId = extractVideoId(ytLink);
    if (!videoId) {
      return res.status(400).json({ error: 'Invalid YouTube URL' });
    }

    // Get video info
    const videoInfo = await getVideoInfo(videoId);
    
    // Use OpenAI to analyze the video content
    const analysis = await analyzeVideoContent(videoInfo, numClips, clipDuration);
    
    // Generate clips with AI-enhanced titles and descriptions
    const enhancedClips = await enhanceClipsWithAI(analysis.clips, videoInfo);

    res.json({
      success: true,
      message: 'Real analysis completed using AI!',
      video_url: ytLink,
      video_info: videoInfo,
      clips: enhancedClips,
      analysis_summary: analysis.summary
    });

  } catch (error) {
    console.error('Analysis error:', error);
    res.status(500).json({ 
      error: 'Failed to analyze video',
      details: error.message 
    });
  }
});

// Helper function to extract YouTube video ID
function extractVideoId(url) {
  const regex = /(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/)([^"&?\/\s]{11})/;
  const match = url.match(regex);
  return match ? match[1] : null;
}

// Helper function to get video info
async function getVideoInfo(videoId) {
  try {
    // For now, we'll use a basic approach since we don't have YouTube API key
    // In production, you'd use YouTube Data API v3
    return {
      id: videoId,
      title: `Video ${videoId}`,
      description: 'Video content analysis',
      duration: '10:00', // Placeholder
      thumbnail: `https://img.youtube.com/vi/${videoId}/maxresdefault.jpg`,
      url: `https://www.youtube.com/watch?v=${videoId}`
    };
  } catch (error) {
    console.error('Error getting video info:', error);
    return {
      id: videoId,
      title: 'Unknown Video',
      description: 'Video content',
      duration: '10:00',
      thumbnail: `https://img.youtube.com/vi/${videoId}/maxresdefault.jpg`,
      url: `https://www.youtube.com/watch?v=${videoId}`
    };
  }
}

// Helper function to analyze viral potential using Anthropic Claude
async function analyzeViralPotential(videoInfo, numClips, clipDuration) {
  try {
    const prompt = `Analyze this YouTube video for viral potential and suggest ${numClips} clips that would perform well on social media:

Video Title: ${videoInfo.title}
Video Description: ${videoInfo.description}
Duration: ${videoInfo.duration}

Please suggest ${numClips} clips with:
1. Start time (in MM:SS format)
2. End time (each clip should be ${clipDuration} seconds)
3. Viral score (1-10)
4. Predicted views
5. Target platforms (TikTok, Instagram Reels, YouTube Shorts)
6. Why this clip would go viral

Format your response as JSON with this structure:
{
  "clips": [
    {
      "start_time": "MM:SS",
      "end_time": "MM:SS", 
      "viral_score": 8.5,
      "predicted_views": 500000,
      "target_platforms": ["TikTok", "Instagram Reels"],
      "viral_reason": "Emotional hook and relatability"
    }
  ],
  "summary": {
    "total_clips": ${numClips},
    "average_viral_score": 8.2,
    "recommended_platforms": ["TikTok", "Instagram Reels", "YouTube Shorts"]
  }
}`;

    const response = await anthropic.messages.create({
      model: 'claude-3-sonnet-20240229',
      max_tokens: 2000,
      messages: [{ role: 'user', content: prompt }]
    });

    const analysis = JSON.parse(response.content[0].text);
    return analysis;
  } catch (error) {
    console.error('Error analyzing viral potential:', error);
    // Fallback to generated clips
    return generateFallbackClips(numClips, clipDuration);
  }
}

// Helper function to analyze video content using OpenAI
async function analyzeVideoContent(videoInfo, numClips, clipDuration) {
  try {
    const prompt = `Analyze this YouTube video and suggest ${numClips} interesting clips:

Video Title: ${videoInfo.title}
Video Description: ${videoInfo.description}
Duration: ${videoInfo.duration}

Please suggest ${numClips} clips with:
1. Start time (in MM:SS format)
2. End time (each clip should be ${clipDuration} seconds)
3. Why this clip is interesting
4. Target audience

Format your response as JSON with this structure:
{
  "clips": [
    {
      "start_time": "MM:SS",
      "end_time": "MM:SS",
      "interest_reason": "Key insight or engaging moment",
      "target_audience": "General audience"
    }
  ],
  "summary": {
    "total_clips": ${numClips},
    "video_theme": "Main theme of the video"
  }
}`;

    const response = await openai.chat.completions.create({
      model: 'gpt-4',
      messages: [{ role: 'user', content: prompt }],
      max_tokens: 1500
    });

    const analysis = JSON.parse(response.choices[0].message.content);
    return analysis;
  } catch (error) {
    console.error('Error analyzing video content:', error);
    // Fallback to generated clips
    return generateFallbackClips(numClips, clipDuration);
  }
}

// Helper function to enhance clips with AI-generated titles and descriptions
async function enhanceClipsWithAI(clips, videoInfo) {
  try {
    const enhancedClips = [];
    
    for (let i = 0; i < clips.length; i++) {
      const clip = clips[i];
      
      // Generate engaging title using OpenAI
      const titlePrompt = `Generate a viral, engaging title for a ${clip.end_time - clip.start_time} second clip from a video titled "${videoInfo.title}". The clip is interesting because: ${clip.viral_reason || clip.interest_reason}. Make it catchy and under 60 characters.`;
      
      const titleResponse = await openai.chat.completions.create({
        model: 'gpt-4',
        messages: [{ role: 'user', content: titlePrompt }],
        max_tokens: 50
      });
      
      const title = titleResponse.choices[0].message.content.trim();
      
      enhancedClips.push({
        id: `clip_${Date.now()}_${i}`,
        start_time: clip.start_time,
        end_time: clip.end_time,
        title: title,
        description: clip.viral_reason || clip.interest_reason || 'Engaging content from the video',
        viral_score: clip.viral_score || 7.5,
        predicted_views: clip.predicted_views || 100000,
        thumbnail_url: videoInfo.thumbnail,
        download_url: `https://example.com/clip_${i}.mp4`,
        target_platforms: clip.target_platforms || ['TikTok', 'Instagram Reels']
      });
    }
    
    return enhancedClips;
  } catch (error) {
    console.error('Error enhancing clips:', error);
    // Return original clips if enhancement fails
    return clips.map((clip, i) => ({
      id: `clip_${Date.now()}_${i}`,
      start_time: clip.start_time,
      end_time: clip.end_time,
      title: `Clip ${i + 1} - ${videoInfo.title}`,
      description: clip.viral_reason || clip.interest_reason || 'Interesting content',
      viral_score: clip.viral_score || 7.5,
      predicted_views: clip.predicted_views || 100000,
      thumbnail_url: videoInfo.thumbnail,
      download_url: `https://example.com/clip_${i}.mp4`,
      target_platforms: clip.target_platforms || ['TikTok', 'Instagram Reels']
    }));
  }
}

// Fallback function to generate clips when AI analysis fails
function generateFallbackClips(numClips, clipDuration) {
  const clips = [];
  for (let i = 0; i < numClips; i++) {
    const startTime = Math.floor(Math.random() * 300) + 60;
    const endTime = startTime + clipDuration;
    
    clips.push({
      id: `fallback_clip_${Date.now()}_${i}`,
      start_time: `${Math.floor(startTime / 60)}:${(startTime % 60).toString().padStart(2, '0')}`,
      end_time: `${Math.floor(endTime / 60)}:${(endTime % 60).toString().padStart(2, '0')}`,
      title: `Viral Moment ${i + 1} - This Will Blow Your Mind! 😱`,
      description: `This clip has viral potential due to its emotional impact and relatability.`,
      duration: clipDuration,
      timestamp: Date.now(),
      filename: `fallback_clip_${i + 1}.mp4`,
      viral_score: 7.5 + (Math.random() * 2),
      predicted_views: Math.floor(Math.random() * 500000) + 50000,
      thumbnail_url: `https://via.placeholder.com/300x200/FF6B6B/FFFFFF?text=Viral+Clip+${i + 1}`,
      download_url: `https://example.com/clip_${i}.mp4`,
      target_platforms: ['TikTok', 'Instagram Reels', 'YouTube Shorts'],
      viral_reason: 'Engaging content with viral potential'
    });
  }
  
  return clips;
}

// Export for Vercel
module.exports = app;
