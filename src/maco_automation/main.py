import uvicorn
from maco_automation.api.main import app

if __name__ == "__main__":
    # Host="0.0.0.0" is essential for Docker
    # It tells uvicorn to listen on all network interfaces
    uvicorn.run(app, host="0.0.0.0", port=8000)