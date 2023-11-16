Project: ECSE488 Low Cost System for Video Surveillance 
Team 5

This zip file contains all documents for the implementation of a video surveillance system, includes:
1. FinalReport.pdf --- The final project report
2. MultiCam.py --- The implementation of the system
3. DemoSlide.pdf --- presentation slides
4. 2x2demo.pdf -- 2x2 demo slides
5. README

To run and test the model:
Prerequisite library: 
1. Python 3.9
2. Numpy
3. Opencv 4.0

Usage
The program is ready to run with default settings, to run the model, type: 
python3 MultiCam.py
in bash, or open the python code in any IDE and run it

Parameter description:
You can change the global variables to control the behavior of the model, the default setup can run properly in most of the situations.
NUM_CAMERAS --- the number of the cameras connected to the system (from 1 up to 4)
THRESHOLD_MOTION --- threshold for passingby target (reduce this param will increase the sensitivity for passingby 			   target detection)
THRESHOLD_APPROACH --- threshold for approaching target (reduce this param will increase the sensitivity for                			   approaching target detection)
TIMEOUT --- the maximum time before reseting the system
DRAW_TRAINGLE --- to draw a traingle that contains the moving target
IMSHOW --- show the captured video

Contribution:
This project is created and implemented by:
Anbang Chen
Shibo Wang

Contact Information:
For more information, please contact:
anbangchen@case.edu
shibowang@case.edu

References
This code is inspired by the opencv library and the idea of Gausian Mixture Model.
