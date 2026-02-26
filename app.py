from era import create_app

app = create_app()

@app.get("/")
def root():
    return {
        "status": "running",
        "service": "early-risk-alert-ai-prod"
    }

@app.get("/healthz")
def healthz():
    return {
        "status": "ok"
    }

if __name__ == "__main__":
    app.run(debug=True)