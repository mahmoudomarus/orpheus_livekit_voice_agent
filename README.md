# Orpheus TTS with Livekit Voice Agent

This project demonstrates how to use Orpheus TTS (hosted on Baseten) with Livekit to create an interactive voice agent.

## Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your API keys:
   ```
   ORPHEUS_API_KEY=your_orpheus_api_key
   LIVEKIT_URL=wss://bignoodle-bwhouzvj.livekit.cloud
   LIVEKIT_API_KEY=APIFQrCdYBAuxeE
   LIVEKIT_API_SECRET=KW6GPLsgQ0YBgOib1GJCaMzTLcR7Tf1ebRlPMD69BQk
   OPENAI_API_KEY=your_openai_api_key
   ASSEMBLYAI_API_KEY=your_assemblyai_api_key # if using AssemblyAI for STT
   ```
   
   Make sure to replace `your_orpheus_api_key` with your actual Baseten API key for Orpheus TTS.

4. Run the voice agent:
   ```bash
   python voice_agent_openai.py
   ```
   
   Or if you want to use the Ollama version:
   ```bash
   python "voice_agent (1).py"
   ```

## Components

- **orpheus_tts.py**: Custom implementation of the Orpheus TTS using the Baseten API
- **voice_agent_openai.py**: Voice agent using OpenAI for speech-to-text and Orpheus for text-to-speech
- **voice_agent (1).py**: Voice agent using AssemblyAI for speech-to-text and Orpheus for text-to-speech

## Configuration Options

The Orpheus TTS plugin accepts the following parameters:

- `api_key`: Your Baseten API key for Orpheus
- `voice`: The voice to use (default: "tara")
- `max_tokens`: Maximum number of tokens to generate (default: 10000)
- `endpoint_url`: The Baseten API endpoint (default: "https://model-2qjn02j3.api.baseten.co/environments/production/predict")
- `sample_rate`: Audio sample rate (default: 24000)

## About Orpheus TTS

Orpheus is a high-quality text-to-speech model hosted on Baseten. It provides natural-sounding voices and is accessible via a simple API.

To get your own API key, sign up on Baseten and deploy the Orpheus model to your account.
