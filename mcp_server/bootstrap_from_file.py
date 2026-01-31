"""
Bootstrap PLTM from exported conversation data

Usage:
    python bootstrap_from_file.py conversations.json
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_server.bootstrap_pltm import PLTMBootstrapper
from loguru import logger


async def main():
    if len(sys.argv) < 2:
        print("Usage: python bootstrap_from_file.py <conversations.json>")
        print("\nExpected JSON format:")
        print("""
[
  {
    "title": "Conversation title",
    "messages": [
      {"role": "user", "content": "Your message"},
      {"role": "assistant", "content": "AI response"}
    ]
  }
]
        """)
        return
    
    # Load conversations from file
    conv_file = Path(sys.argv[1])
    if not conv_file.exists():
        print(f"Error: File not found: {conv_file}")
        return
    
    with open(conv_file, 'r', encoding='utf-8') as f:
        conversations = json.load(f)
    
    print(f"Loaded {len(conversations)} conversations from {conv_file}")
    
    # Bootstrap
    bootstrapper = PLTMBootstrapper()
    await bootstrapper.initialize()
    
    results = await bootstrapper.bootstrap_from_conversations(conversations)
    
    print("\n" + "="*60)
    print("BOOTSTRAP COMPLETE")
    print("="*60)
    print(f"Conversations: {results['conversations_processed']}")
    print(f"Atoms created: {results['atoms_created']}")
    print(f"Styles: {results['styles_extracted']}")
    print(f"Unique styles: {results['unique_styles']}")
    print("="*60)
    
    # Show personality
    from src.personality.personality_synthesizer import PersonalitySynthesizer
    synth = PersonalitySynthesizer(bootstrapper.store)
    personality = await synth.synthesize_personality("alby")
    
    print("\nPERSONALITY PROFILE:")
    print(json.dumps(personality, indent=2))
    
    await bootstrapper.close()


if __name__ == "__main__":
    asyncio.run(main())
