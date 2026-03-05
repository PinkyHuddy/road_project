## **Data Extraction and Event Identification**

For the data extraction I started by using the PeMS system or Freeway Performance Measurement System. I accessed this through DoT of California, allowing me to pull 10000 files of data containing 5 minute intervals of sensor data accross various stations in the District 3 region. 

I then explored the data and found the needed variables to gain closure insight, these included average Flow, average Speed, Timestamp, and Occupancy. With this I then created a few iteration of python scripts that would parse through the data and find possible closures. The first few were good attempts but took far to long (many hours). In the end I was able to engineer a pipeline that extracted a possible closures by the length of the closure, the start time, and the station location. 

Parsing the data may be challenging and using predicting the closures can not be done with this data alone. The next goal is to get closures and then narrow the table down to one row per date with a boolean column stating whether there was a closure (this is the target variable), the length of closure, and temperature variables. These will be used in the model to predict whether there is a closure or not and the length of closure.


This did not work, working on model that can be trained so that it correctly classifies tweets with high accuracy.