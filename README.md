# Face Recognition and Object Detection on a Raspberry Pi 4

This was my individual project for the Computer Vision Systems module (CS6461) at the University of Limerick, December 2025. Every student worked on their own Raspberry Pi.

## What it does

A Raspberry Pi 4 with a camera watches a room. It detects everyday objects with YOLOv5, recognises up to five known people by their faces, and writes down who came into view, who left and when. The aim was to get this working on a small computer with no GPU, and to understand what makes it fail.

## Hardware

- Raspberry Pi 4 Model B, 4 GB RAM
- Raspberry Pi Camera Module v2 (Sony IMX219, 8 MP)
- Operated over Remote Desktop on Wi-Fi

## How it works

1. **Capture.** Frames are captured at 1280x720 with Picamera2.
2. **Known faces.** Each known person has a folder of photos. The face_recognition library (built on dlib) turns every face into a 128-number embedding. The embeddings are cached in a pickle file, and an MD5 hash of the photo folder tells the program when it needs to rebuild the cache.
3. **Detection on every third frame.** Object and face detection only run on one frame in three. The two frames in between reuse the last results, so the expensive work happens a third as often.
4. **Objects.** YOLOv5n (the smallest YOLOv5 model) through Ultralytics, with a 416 px input.
5. **Faces.** Faces are found on a half-size copy of the frame to save time, and the boxes are scaled back up. Each face is compared with the known embeddings. A distance below 0.6 counts as a match, and the confidence shown on screen is 1 minus the distance.
6. **Logging.** `movement_log.csv` stores each person's entry time, exit time and average confidence. `object_log.csv` stores each detected object with its confidence and time.
7. **Live controls.** Keyboard keys adjust brightness and sharpness while it runs, which slows it down a little.

## Lighting tests

I tested the system in three lighting conditions. I measured the light with a phone light-meter app, not a proper lux meter, so the values are approximate.

| Scenario | Set-up | What happened |
|---|---|---|
| Normal (about 150 to 250 lux) | Living room, overhead LEDs and a warm bulb | Everyone was recognised, with some ups and downs in confidence |
| Low light (below about 50 lux) | Main lights off | The image became grainy, known people were often labelled "Unknown" and small objects were missed |
| Directional | A torch and a phone light moved around | Shadows caused some false object detections. Face recognition handled angled light better than darkness |

Lighting was the main reason recognition failed. In low light the camera raises its gain to brighten the picture, and that also amplifies the sensor noise, which blurs the facial detail the embeddings depend on.

## Results from my logs

In my main logged session (about 4 minutes):

- 179 object detections and 16 face entry/exit intervals were logged
- a person was detected in 91 of 92 detection passes
- known faces were matched with an average confidence of about 0.42 to 0.48

## Notes on speed

I did not benchmark the frame rate properly. The on-screen FPS counter measured single loop iterations, which reads high on the frames that skip detection. The logs show roughly one detection pass every 2.6 seconds in the main session. If I did this again I would log the time of every frame and test with a recorded video so runs can be compared.

## How to run

On Raspberry Pi OS:

```
sudo apt install python3-picamera2
pip install -r requirements.txt
```

Add photos of the people you want to recognise:

```
known_faces/
    person_one/   photo1.jpg, photo2.jpg, ...
    person_two/   photo1.jpg, ...
```

Then run:

```
python cam_yolo.py
```

The face photos and recorded videos from my tests are not included, because they show real people.

## Project structure

```
.
├── cam_yolo.py
├── results/
│   ├── movement_log.csv
│   └── object_log.csv
├── requirements.txt
└── README.md
```

## Tools used

Python 3.9, Picamera2, OpenCV, Ultralytics YOLOv5n, face_recognition (dlib), NumPy.

## What I would improve

- Run capture and detection in separate threads
- Use a hardware accelerator for YOLO
- Measure frame rate and latency properly
- Better low-light handling, for example an infrared camera
