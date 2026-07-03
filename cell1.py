import os 

# Append Deno binary to the system PATH so yt-dlp can access it automatically
os.environ["PATH"] = "/root/.deno/bin:" + os.environ.get("PATH", "")

# Initialize the protected local scratch directory (No Google Drive dependencies)
WORK_DIR = "/content/Automated_Shorts"
os.makedirs(WORK_DIR, exist_ok=True)

# Generate a placeholder cookiefile if one does not exist (cookies are needed for age-restricted content / to avoid anti-bot related problems)
cookie_path = '/content/cookies.txt'
if not os.path.exists(cookie_path):
    with open(cookie_path, 'w') as f:
        f.write("# Netscape HTTP Cookie File\n")

print("="*60)
print("[*] Env setup done !")
print(f"[*] Working Directory: {WORK_DIR}")
print("[!] IMPORTANT: If processing age-restricted videos, drag and drop your exported 'cookies.txt' into the main /content/ folder")
print("="*60)
