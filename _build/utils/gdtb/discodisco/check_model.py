import os, sys

if not os.path.exists("models/eng.pdtb.pdtb_flair_clone/model.tar.gz"):
    response = input("! discodisco model.tar.gz missing - do you wish to download it?\n")
    if response.lower() in ["yes", "y"]:
        import requests
        import shutil
        url = "https://gucorpling.org/amir/download/gdtb/model.tar.gz"
        r = requests.get(url, stream=True)
        with open("models/eng.pdtb.pdtb_flair_clone/model.tar.gz", "wb") as f:
            shutil.copyfileobj(r.raw, f)
        sys.stderr.write("Downloaded model.tar.gz\n")
    else:
        sys.exit(1)
