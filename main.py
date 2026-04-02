import argparse
from src.infer_pipeline import synthesize_text_to_emotional_speech
import sys
import io

# Fix Unicode printing on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True, help='Input text or file')
    parser.add_argument('--gender', choices=['female', 'male'], required=True)
    parser.add_argument('--accent', choices=['american', 'british', 'australian', 'indian', 'neutral'], required=True)
    parser.add_argument('--out', required=True, help='Output WAV file path')
    args = parser.parse_args()

    # Call TTS function with gender and accent
    synthesize_text_to_emotional_speech(
        text_or_path=args.input,
        gender=args.gender,
        accent=args.accent,
        out_path=args.out
    )

    print(f"Emotional speech saved at: {args.out}")

if __name__ == "__main__":
    main()
