## Lucas-Kanade-Tracker

## PROJECT DESCRIPTION

The aim of this project is to implement the Lucas-Kanade (LK) template tracker. Then to evaluate the code on three video sequences from the Visual Tracker benchmark database: featuring a car on the road, a Bolt, and a  on a dragonbaby.


To initialize the tracker, We define a template by drawing a bounding box around the object to be tracked in the first frame of the video. For each of the subsequent frames the tracker will update an affine transform that warps the current frame so that the template in the first frame is aligned with the warped current frame.

## Run the code

Enter the following to run the code.

For Bolt:
```
cd code
python3 trackBolt2.py
```
For car:
```
cd code
python3 trackCar2.py
```
For DragonBaby:
```
cd code
python3 trackDragonbaby.py
```

## Sample Output:
For Bolt

![3xgj1r](https://user-images.githubusercontent.com/55011289/79813555-fb49f200-8349-11ea-9a86-f0db137c8a59.gif)

For Car:

![Alt Text](gif/car.gif)


For DragonBaby

![3xgje8](https://user-images.githubusercontent.com/55011289/79813700-5c71c580-834a-11ea-8566-ec8e4c476efd.gif)




### Robustness to Illumination
In order to increase the robustness of the tracker, we need to scale the brightness of pixels in each frame so that the average brightness of pixels in the tracked region stays the same as the average brightness of pixels in the template. For tracking the car, gamma correction did not give better results. So, instead we employed z-scores method for tracking.

### Run code
To run the robust tracker for car. Please run the following code.
```
cd code
python3 trackCarRobust2.py
```
### Sample output
With and without Z-score output videos. 

![Alt Text](gif/car2.gif)
![Alt Text](gif/car1.gif) 