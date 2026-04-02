# src/style_mapper.py
import math

EMOTION_TO_STYLE = {
    'joy': 'happy',
    'happy': 'happy',
    'sadness': 'sad',
    'sad': 'sad',
    'anger': 'angry',
    'angry': 'angry',
    'fear': 'fear',
    'surprise': 'surprised',
    'disgust': 'disgust',
    'neutral': 'neutral'
}

# Base modifiers per style (f0_scale, rate_scale, energy_scale)
BASE_MODIFIERS = {
    'happy':    {'f0_scale': 1.20, 'rate_scale': 1.05, 'energy_scale': 1.15},
    'sad':      {'f0_scale': 0.90, 'rate_scale': 0.92, 'energy_scale': 0.85},
    'angry':    {'f0_scale': 1.25, 'rate_scale': 1.08, 'energy_scale': 1.30},
    'fear':     {'f0_scale': 1.12, 'rate_scale': 1.00, 'energy_scale': 1.05},
    'surprised':{'f0_scale': 1.30, 'rate_scale': 1.10, 'energy_scale': 1.18},
    'disgust':  {'f0_scale': 0.95, 'rate_scale': 0.95, 'energy_scale': 0.9},
    'neutral':  {'f0_scale': 1.00, 'rate_scale': 1.00, 'energy_scale': 1.00},
}

class StyleMapper:
    def __init__(self):
        pass

    def map(self, label: str, intensity: float = 0.6):
        """
        label: detected label (e.g., 'happy' or 'joy')
        intensity: 0.0 - 1.0 controlling strength
        Returns: dict with style_name and prosody modifiers
        """
        label = label.lower()
        style = EMOTION_TO_STYLE.get(label, 'neutral')
        base = BASE_MODIFIERS.get(style, BASE_MODIFIERS['neutral'])

        # intensity adjusts how strong the modifier is (linear blend with neutral)
        blend = 0.4 + 0.6 * max(0.0, min(1.0, intensity))  # between 0.4..1.0
        prosody = {
            'f0_scale': float(base['f0_scale'] * blend + 1.0 * (1.0 - blend)),
            'rate_scale': float(base['rate_scale'] * blend + 1.0 * (1.0 - blend)),
            'energy_scale': float(base['energy_scale'] * blend + 1.0 * (1.0 - blend))
        }
        return {
            'style_name': f"{style}",
            'prosody': prosody,
            'style': style,
            'intensity': float(intensity)
        }
