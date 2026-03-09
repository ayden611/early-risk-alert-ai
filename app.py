from era import create_app
from flask import send_from_directory

@app.route("/pitch-deck")
def pitch_deck():
    return send_from_directory(
        "static",
        "Early_Risk_Alert_AI_Pitch_Deck.pdf",
        as_attachment=True
    )

app = create_app()
