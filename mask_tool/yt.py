import yt_dlp

def download_youtube_video(url, start_time=None, end_time=None, output_file="output.mp4"):
    # Set the full path to the ffmpeg binary
    ffmpeg_path = 'C:\\Users\\anawi\\Downloads\\ffmpeg-master-latest-win64-gpl\\ffmpeg-master-latest-win64-gpl\\bin'  # Modify this to the actual path to ffmpeg

    ydl_opts = {
        'outtmpl': output_file,
        'format': 'bestvideo+bestaudio/best',  # Select the best video and audio quality
        'merge_output_format': 'mp4',  # Combine video and audio into mp4 format
        'ffmpeg_location': ffmpeg_path  # Add the ffmpeg path if necessary
    }

    # Add start and end time options if provided
    if start_time or end_time:
        time_range = []
        if start_time:
            time_range.extend(['-ss', start_time])
        if end_time:
            time_range.extend(['-to', end_time])

        ydl_opts['postprocessor_args'] = time_range

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

# Example usage
url = 'https://www.youtube.com/watch?v=xbBKbDwlR0E'
start_time = '00:01:00'
end_time = '00:05:00'

download_youtube_video(url, start_time, end_time)
