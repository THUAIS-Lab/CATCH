import uvicorn
if __name__ == "__main__":
    uvicorn.run("v_app.server:app", host="127.0.0.1", port=8080, reload=False)
