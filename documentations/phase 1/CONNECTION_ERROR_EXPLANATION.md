# Connection Error Explanation

## Error: "No route to host" / "Failed to connect to video stream"

### What This Means

This error indicates a **network connectivity issue**, not a code bug. The script is working correctly:

1. ✅ **URL is preserved exactly as provided** - No automatic modifications
2. ✅ **Connection attempt is made** - `cv2.VideoCapture` is called with your exact URL
3. ✅ **Error is handled gracefully** - Clear error messages and troubleshooting tips

### Why This Happens

The error `Connection to tcp://192.168.0.220:8080 failed: No route to host` means:

- The IP address `192.168.0.220:8080` is not reachable from your current machine
- Possible reasons:
  1. **Phone not connected** - IP Webcam app is not running
  2. **Different network** - Phone and PC are on different Wi-Fi networks
  3. **Wrong IP address** - Phone's IP address may have changed
  4. **Firewall blocking** - Network firewall may be blocking the connection
  5. **Wrong endpoint** - Some IP Webcam apps require `/video` or other endpoints

### How to Fix

1. **Check IP Webcam app is running** on your phone
2. **Verify phone and PC are on same Wi-Fi network**
3. **Get current IP address** from IP Webcam app (it may have changed)
4. **Try with `/video` suffix** if your app requires it:
   ```bash
   poetry run python scripts/test_object_tracking_live.py \
     --prompt "desk lamp" \
     --method fan \
     --device cpu \
     --stream-url "http://192.168.0.220:8080/video"
   ```
5. **Test connection in browser first** - Open `http://192.168.0.220:8080` (or with `/video`) in a web browser to verify it works

### Testing Without Phone

Use `--test-mode` to test the script logic without requiring a live phone connection:

```bash
poetry run python scripts/test_object_tracking_live.py \
  --prompt "desk lamp" \
  --method fan \
  --device cpu \
  --test-mode \
  --max-frames 10
```

### Verification

The code is working correctly. Tests verify:
- ✅ URL is preserved exactly as provided
- ✅ Connection attempt uses exact URL (no modifications)
- ✅ Error handling works correctly
- ✅ All connection logic is tested

**The connection failure is expected when the phone isn't reachable** - this is normal network behavior, not a code issue.
