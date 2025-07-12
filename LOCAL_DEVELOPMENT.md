# Local Development Guide

## Quick Start

### 1. Install Dependencies
```bash
npm run install-all
```

### 2. Set Up Environment Variables
Create a `.env` file in the root directory:
```bash
cp env.example .env
```

Then edit `.env` and add your API keys:
```env
OPENAI_API_KEY=sk-your-openai-api-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-api-key-here
REACT_APP_API_URL=http://localhost:3001/api
```

### 3. Run the Backend API
```bash
npm run dev
```
This will start the API on `http://localhost:3001`

### 4. Run the Frontend (in a new terminal)
```bash
npm run frontend
```
This will start the React app on `http://localhost:3000`

### 5. Test the Application
- Open `http://localhost:3000` in your browser
- Enter a YouTube URL
- Click "Generate Viral Clips"
- You should see real AI-generated clips!

## API Endpoints

- `GET http://localhost:3001/` - API status
- `GET http://localhost:3001/test` - Test endpoint
- `POST http://localhost:3001/analyze-viral` - Generate viral clips

## Troubleshooting

### "OPENAI_API_KEY environment variable is missing"
- Make sure you have a `.env` file in the root directory
- Check that your API keys are correct
- Restart the server after changing environment variables

### "Backend not available" in frontend
- Make sure the backend is running on port 3001
- Check that `REACT_APP_API_URL` is set to `http://localhost:3001/api`
- Check browser console for CORS errors

### API rate limits
- OpenAI and Anthropic have rate limits
- The app includes fallback mechanisms when APIs are unavailable

## Getting API Keys

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

## Features Available Locally

✅ **Real AI Analysis**: Uses OpenAI and Anthropic APIs
✅ **Viral Potential Scoring**: AI-powered viral score prediction
✅ **Smart Clip Generation**: AI suggests optimal clip timings
✅ **Engaging Titles**: AI-generated viral titles for each clip
✅ **Platform Recommendations**: Suggests best platforms for each clip
✅ **Fallback System**: Works even if AI APIs are temporarily unavailable

## Development Notes

- The local API uses the same code as the Vercel deployment
- All AI features work locally with proper API keys
- The frontend connects to the local API via `REACT_APP_API_URL`
- Changes to the API code require restarting the server
- Changes to the frontend code auto-reload in development

## Next Steps

1. **Deploy to Vercel**: Follow the `VERCEL_DEPLOYMENT_GUIDE.md`
2. **Add YouTube API**: For better video information
3. **Add Video Processing**: Integrate with external services
4. **Add Analytics**: Monitor usage and performance

Happy coding! 🚀 