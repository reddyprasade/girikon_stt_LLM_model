import logging
import os
import sys
import uvicorn
from dotenv import load_dotenv

load_dotenv()

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    uvicorn.run("driver:app",
                host=os.getenv("APP_HOST", "0.0.0.0"),
                port=int(os.getenv("APP_PORT") or 4000),
                reload=False,
                # use_colors=True,
                # access_log=True,
                # log_level="info",
                # forwarded_allow_ips="*",
                # proxy_headers=True,
                # ssl_keyfile="./ssl/key.pem",
                # ssl_certfile="./ssl/cert.pem",
                headers=[
                    ("server", "Girikon AI")
                ],
                timeout_keep_alive=10000,
                # reload_delay=0.10,
                #reload_includes=["./app.py", "./server.py", "./.env"],
                #reload_excludes=["./.idea", "./logs", "./data", ".gitignore", "ecosystem.config.json", "README.md", "requirements.txt"],

                )
