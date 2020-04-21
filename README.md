## Lucas-Kanade-Tracker

## PROJECT DESCRIPTION

The aim of this project is to implement the Lucas-Kanade (LK) template tracker. Then to evaluate the code on three video sequences from the Visual Tracker benchmark database: featuring a car on the road, a Bolt, and a  on a dragonbaby.


To initialize the tracker, We define a template by drawing a bounding box around the object to be tracked in the first frame of the video. For each of the subsequent frames the tracker will update an affine transform that warps the current frame so that the template in the first frame is aligned with the warped current frame.
# Run the code

Enter the following to run the code.

For Bolt:
```
python3 trackBolt.py
```
For car:
```
python3 trackCar.py
```
For DragonBaby:
```
python3 trackDragonbaby.py
```


![3xgj1r](https://user-images.githubusercontent.com/55011289/79813555-fb49f200-8349-11ea-9a86-f0db137c8a59.gif)
