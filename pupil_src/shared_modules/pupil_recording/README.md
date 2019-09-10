# Pupil Player Recording Format 2.0

## File Hierarchy

```
+- info.player.json (REQUIRED)
+- ...
```

## `info.player.json` format

Serialization format as defined in https://docs.python.org/3.6/library/json.html

All following versions follow [Semantic Versioning 2.0.0](https://semver.org/).

Fields:
```js
{
    // required keys
    "meta_version": "2.0",
    "min_player_version": "1.16",
    "recording_uuid": string,  // follows RFC 4122
    "start_time_system_s": float,  // double-precision, unit: seconds
    "start_time_synced_s": float,  // double-precision, unit: seconds
    "duration_s": float,  // double-precision, unit: seconds
    "recording_software_name": string,
    "recording_software_version": string,
    // optional
    "recording_name": string,
    "system_info": string,
}
```