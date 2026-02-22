import os
import sys

# Force Python to load local project folder first
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from era import create_app

app = create_app()
