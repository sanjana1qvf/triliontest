# Vercel Deployment Guide for TrilionClips

## Overview
This guide will help you deploy your TrilionClips application (frontend + backend) on Vercel with real AI-powered video analysis.

## Prerequisites
- GitHub account with the repository: `https://github.com/sanjana1qvf/triliontest.git`
- Vercel account
- OpenAI API key
- Anthropic API key

## Step 1: Get API Keys

### OpenAI API Key
1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Sign up or log in
3. Go to "API Keys" section
4. Create a new API key
5. Copy the key (starts with `sk-`)

### Anthropic API Key
1. Go to [Anthropic Console](https://console.anthropic.com/)
2. Sign up or log in
3. Go to "API Keys" section
4. Create a new API key
5. Copy the key (starts with `sk-ant-`)

## Step 2: Deploy on Vercel

### Option 1: Vercel Dashboard (Recommended)

1. **Go to Vercel Dashboard**
   - Visit [vercel.com](https://vercel.com)
   - Sign in with your GitHub account

2. **Import Repository**
   - Click "New Project"
   - Import your GitHub repository: `https://github.com/sanjana1qvf/triliontest.git`
   - Vercel will auto-detect the monorepo structure

3. **Configure Project Settings**
   - **Project Name**: `triliontest` (or your preferred name)
   - **Framework Preset**: Select "Other"
   - **Root Directory**: Leave as `/` (root)
   - **Build Command**: Leave empty (auto-detect)
   - **Output Directory**: Leave empty (auto-detect)

4. **Set Environment Variables**
   - Click "Environment Variables" section
   - Add the following variables:

   ```
   OPENAI_API_KEY=sk-your-openai-api-key-here
   ANTHROPIC_API_KEY=sk-ant-your-anthropic-api-key-here
   REACT_APP_API_URL=https://your-project-name.vercel.app/api
   ```

   **Important**: Replace `your-project-name` with your actual Vercel project name.

5. **Deploy**
   - Click "Deploy"
   - Wait for the build to complete
   - You'll get a URL like: `https://triliontest-abc123.vercel.app`

### Option 2: Vercel CLI

1. **Install Vercel CLI**
   ```bash
   npm i -g vercel
   ```

2. **Login to Vercel**
   ```bash
   vercel login
   ```

3. **Deploy**
   ```bash
   vercel
   ```

4. **Set Environment Variables**
   ```bash
   vercel env add OPENAI_API_KEY
   vercel env add ANTHROPIC_API_KEY
   vercel env add REACT_APP_API_URL
   ```

## Step 3: Update Environment Variables (After First Deploy)

After your first deployment, update the `REACT_APP_API_URL`:

1. Go to your Vercel project dashboard
2. Navigate to "Settings" → "Environment Variables"
3. Update `REACT_APP_API_URL` to your actual Vercel URL:
   ```
   REACT_APP_API_URL=https://your-actual-project-name.vercel.app/api
   ```
4. Redeploy the project

## Step 4: Test Your Application

1. **Visit your Vercel URL**
   - Example: `https://triliontest-abc123.vercel.app`

2. **Test the API**
   - Go to: `https://your-project-name.vercel.app/api/test`
   - You should see: `{"message": "Backend is working on Vercel!"}`

3. **Test Video Analysis**
   - Enter a YouTube URL in the frontend
   - Click "Generate Viral Clips"
   - You should see real AI-generated clips with viral analysis

## How It Works

### Frontend (React)
- Located in `/frontend` directory
- Built using `@vercel/static-build`
- Serves static files

### Backend (Node.js API Routes)
- Located in `/api` directory
- Uses `@vercel/node` for serverless functions
- Handles video analysis using:
  - **OpenAI GPT-4**: For content analysis and title generation
  - **Anthropic Claude**: For viral potential analysis
  - **YouTube Data**: For video information extraction

### Routing
- `/api/*` → Backend API routes
- `/*` → Frontend React app

## Features

✅ **Real AI Analysis**: Uses OpenAI and Anthropic APIs for actual video analysis
✅ **Viral Potential Scoring**: AI-powered viral score prediction
✅ **Smart Clip Generation**: AI suggests optimal clip timings
✅ **Engaging Titles**: AI-generated viral titles for each clip
✅ **Platform Recommendations**: Suggests best platforms for each clip
✅ **Fallback System**: Works even if AI APIs are temporarily unavailable

## Troubleshooting

### Common Issues

1. **"Backend not available" error**
   - Check if environment variables are set correctly
   - Verify API keys are valid
   - Check Vercel function logs

2. **CORS errors**
   - The backend is configured to allow all common origins
   - If you have a custom domain, add it to the CORS origins in `/api/index.js`

3. **API rate limits**
   - OpenAI and Anthropic have rate limits
   - The app includes fallback mechanisms for when APIs are unavailable

4. **Build failures**
   - Check that all dependencies are in `package.json`
   - Verify Node.js version compatibility

### Environment Variables Checklist

- [ ] `OPENAI_API_KEY` - Your OpenAI API key
- [ ] `ANTHROPIC_API_KEY` - Your Anthropic API key  
- [ ] `REACT_APP_API_URL` - Your Vercel URL + `/api`

### Testing Checklist

- [ ] Frontend loads correctly
- [ ] API test endpoint works (`/api/test`)
- [ ] Video analysis generates clips
- [ ] Clips have AI-generated titles
- [ ] Viral scores are displayed
- [ ] Platform recommendations work

## Cost Considerations

- **Vercel**: Free tier includes 100GB bandwidth and 100 serverless function executions per day
- **OpenAI**: Pay per API call (GPT-4 is more expensive than GPT-3.5)
- **Anthropic**: Pay per API call (Claude is generally cost-effective)

## Next Steps

1. **Custom Domain**: Add your own domain in Vercel settings
2. **YouTube API**: Add YouTube Data API v3 for better video information
3. **Video Processing**: Integrate with external video processing services
4. **Analytics**: Add usage analytics and monitoring
5. **Caching**: Implement caching for better performance

## Support

If you encounter issues:
1. Check Vercel function logs in the dashboard
2. Verify all environment variables are set
3. Test API endpoints individually
4. Check browser console for frontend errors

Your TrilionClips app is now ready to generate real viral clips using AI! 🚀
