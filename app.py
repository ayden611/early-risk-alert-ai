from era import create_app

try:
    app = create_app()
    print("APP CREATED OK:", app)
except Exception as e:
    print("CREATE_APP FAILED:", repr(e))
    raise
