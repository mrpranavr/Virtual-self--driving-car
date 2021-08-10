# Virtual self-driving-car
Create a virtual self driving car using reinforcement learning with pytorch and Kivy

## Requirements 
- Python 3.6 
- pytorch ( conda install pytorch==0.3.1 -c pytorch )
- kivy ( conda install -c conda-forge/label/cf201901 kivy )

## Initial setup
- Create a virtual environment with python 3.6 ( You can use Anaconda for ease of use )
- Activate the virtual environment
- Install Pytorch and Kivy on the environment using the command lines given in the requirements section
- Copy all the files into single folder
- excecute the map.py file 

## What the output shows
In the beginning the car can be seen moving rapidly here and there. What is happening is that it is learning how to move from the start to the destination using reinforcement Q-learning. After a few seconds the car moves fluently from one end to the other and back again.

**NOTE:** You can save the model by pressing the `save` button at the bottom left of the window. This also gives a learning curve graph that shows the reward or learning rate for the number of times it tried.
Once a model is saved a new file is created named last_brain.pth. This is your trained model. You can load it the next time you launch the window by pressing the `load` button on the bottom left of the window

https://user-images.githubusercontent.com/88646272/128818670-3216569c-90ce-4676-a101-cbce23544b41.mp4

![normal training data](https://user-images.githubusercontent.com/88646272/128819378-af71d07d-ffe6-4fb7-a31e-071f17f89d71.PNG)


Next let us try to draw a new road map from one end to other. Let us see how to car learns to moves through the new path. You can draw by holding the left mouse button and dragging across the screen.

https://user-images.githubusercontent.com/88646272/128819476-bc362bd0-e530-43b6-8bbe-a15436c5fedc.mp4

Learning graph of route 1 :

![route 1 training data](https://user-images.githubusercontent.com/88646272/128819582-b53778d9-8432-42d0-8c3d-2285361e3569.PNG)

Finally let us try a harder path and see how it learns (SPIOLER ALERT: It hits the walls and goes through them, but if we train it for longer or tweak the code a bit we can make it work more efficiently)


https://user-images.githubusercontent.com/88646272/128819799-9a217ab0-c865-4982-969d-167dbb1cfbf2.mp4

Its graph is :

![route 2 training data](https://user-images.githubusercontent.com/88646272/128819933-93702be6-bfcb-42e2-a953-12e0f4ae2f70.PNG)

We can see the reward is lower than the previous ones. But the more we train the more chance it may go up and improve
