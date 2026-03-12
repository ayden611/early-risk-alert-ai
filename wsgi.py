try:
    from era import create_app
    app = create_app()
    print("WSGI APP:", app, type(app), flush=True)
except Exception as e:
    print("WSGI IMPORT FAILED:", repr(e), flush=True)
    raise
