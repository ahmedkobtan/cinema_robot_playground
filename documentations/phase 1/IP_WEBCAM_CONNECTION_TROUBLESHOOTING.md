# IP Webcam Connection Troubleshooting

## Common Error: "Stream ends prematurely"

### What It Means

The error `[http @ ...] Stream ends prematurely at 40812` indicates that OpenCV successfully connected to the IP Webcam server, but the stream was cut off before it could read a complete frame.

### Why It Happens

1. **URL Format Issue**: Some IP Webcam apps require `/video` suffix, others don't
2. **Backend Issue**: OpenCV's default backend may not handle HTTP streams well
3. **Network Instability**: Connection drops during initial handshake
4. **IP Webcam App Settings**: Stream format or encoding may have changed

### Solutions

#### 1. Try Different URL Formats

The code now automatically tries multiple URL formats:
- Original URL: `http://192.168.0.220:8080`
- With /video: `http://192.168.0.220:8080/video`

If one doesn't work, the other will be tried automatically.

#### 2. Verify URL in Browser

Before running the script, test the URL in a web browser:

```bash
# Try both formats:
http://192.168.0.220:8080
http://192.168.0.220:8080/video
```

If the browser shows video, the URL is correct. If not, check:
- IP Webcam app is running
- Phone and PC are on same Wi-Fi network
- IP address hasn't changed

#### 3. Check IP Webcam App Settings

Different IP Webcam apps have different URL formats:

**IP Webcam (Android)**:
- Default: `http://IP:PORT/video`
- Alternative: `http://IP:PORT/videofeed`

**DroidCam**:
- Default: `http://IP:PORT/video`

**EpocCam**:
- Different format, check app documentation

#### 4. Restart IP Webcam App

Sometimes the app needs to be restarted:
1. Close IP Webcam app completely
2. Restart the app
3. Note the new IP address (it may have changed)
4. Update config file with new IP
5. Try again

#### 5. Check Network Connection

```bash
# Test network connectivity
ping 192.168.0.220

# Test HTTP connection
curl -I http://192.168.0.220:8080
```

#### 6. Update Config File

If the IP address changed, update `config/dev.json`:

```json
{
  "configResolution": {
    "resolved": {
      "env": "DEV",
      "ip_webcam_url": "http://NEW_IP:8080/video"
    }
  }
}
```

### Code Improvements

The updated `VideoStream.connect()` method now:

1. **Tries Multiple URL Formats**: Automatically tries with and without `/video` suffix
2. **Uses FFMPEG Backend**: More reliable for HTTP streams
3. **Sets Buffer Size**: Reduces latency and connection issues
4. **Better Error Messages**: Provides specific troubleshooting steps

### Manual Testing

If automatic URL detection doesn't work, specify the URL explicitly:

```bash
# Try with /video suffix
poetry run python scripts/test_video_stream.py \
  --stream-url "http://192.168.0.220:8080/video" \
  --test-display

# Try without /video suffix
poetry run python scripts/test_video_stream.py \
  --stream-url "http://192.168.0.220:8080" \
  --test-display
```

### Expected Behavior

When connection succeeds, you should see:
```
INFO | Connecting to stream: http://192.168.0.220:8080
INFO | ✓ Connected successfully using: http://192.168.0.220:8080/video
INFO |   (Original URL: http://192.168.0.220:8080)
```

This indicates the code automatically found the correct URL format.

### Still Not Working?

If none of the above works:

1. **Check OpenCV Version**: Ensure you have OpenCV 4.12+ with FFMPEG support
   ```bash
   poetry run python -c "import cv2; print(cv2.__version__)"
   ```

2. **Try Different IP Webcam App**: Some apps are more reliable than others

3. **Use Test Mode**: Test the tracking logic without live stream:
   ```bash
   poetry run python scripts/test_object_tracking_live.py \
     --prompt "desk lamp" \
     --method fan \
     --device cuda \
     --test-mode
   ```

4. **Check Firewall**: Ensure firewall isn't blocking the connection

5. **Network Debugging**: Use Wireshark or tcpdump to see what's happening at network level
