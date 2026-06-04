import yt_dlp

def download_video(url, output_path='./'):
    """
    Downloads a YouTube video as an MP4 file.
    """
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': f'{output_path}/%(title)s.%(ext)s',  # Save with video title
        'merge_output_format': 'mp4',  # Ensure final output is MP4
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

# --- Example Usage ---
video_url = input("Enter the YouTube URL: ")
download_video(video_url)
print("Download complete!")