import requests
import json

# Your YouTube API Key
API_KEY = 'AIzaSyC7sqOcDUCCPVbu_rRSX0nvxxs-3rbJREY'

def get_livestream_url(video_id):
    """Fetch the live stream URL for a YouTube video."""
    url = f"https://www.googleapis.com/youtube/v3/videos?part=liveStreamingDetails&id={video_id}&key={API_KEY}"
    response = requests.get(url)
    data = response.json()

    # Print the entire response to inspect what data is being returned
    print(json.dumps(data, indent=2))  # To see the full API response
    
    if 'items' in data and len(data['items']) > 0:
        live_details = data['items'][0].get('liveStreamingDetails', {})
        
        # Check if there is an actual start time and the stream is live
        if 'actualStartTime' in live_details:
            print(f"Stream Start Time: {live_details['actualStartTime']}")
        
        # Now check for the stream URL (HLS)
        if 'hlsManifestUrl' in live_details:
            return live_details['hlsManifestUrl']  # Return the HLS stream URL
        else:
            print("HLS URL not found.")
    else:
        print("No stream details found.")

    raise ValueError("Live stream URL not found.")

# Example usage
video_url = "https://www.youtube.com/watch?v=U1LMXaV3sYI"  # Replace with your YouTube live stream URL
video_id = video_url.split("v=")[-1]
get_livestream_url(video_id)
